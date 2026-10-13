"""
Instrument OpenAI Responses API calls for New Relic AI Monitoring.

Wraps client.responses.create() / parse() to emit LlmChatCompletionSummary and
LlmChatCompletionMessage events. Temporary workaround until the New Relic Python
agent supports the Responses API natively (PR #1813).
"""

import functools
import time
import logging
from typing import Any, Optional, Callable
import newrelic.agent
from newrelic.api.function_trace import FunctionTrace

logger = logging.getLogger(__name__)


# ============================================================================
# Decorator Implementation
# ============================================================================


def _enter_llm_span() -> Any:
    """
    Mirror what the agent's native OpenAI hook does so APM's AI Monitoring UI
    recognises the app: tag the model/framework (Supportability metric), set the
    transaction `llm` attribute, and open an `Llm/completion/OpenAI/create` span.
    Returns the open FunctionTrace (or None if there is no transaction).
    """
    try:
        txn = newrelic.agent.current_transaction()
        if not txn:
            return None
        txn.add_ml_model_info("OpenAI", None)
        txn._add_agent_attribute("llm", True)
        ft = FunctionTrace(name="create", group="Llm/completion/OpenAI")
        ft.__enter__()
        return ft
    except Exception as e:
        logger.warning(f"Could not open Llm span: {e}")
        return None


def _exit_llm_span(ft: Any) -> None:
    if ft is not None:
        try:
            ft.__exit__(None, None, None)
        except Exception as e:
            logger.debug(f"Could not close Llm span: {e}")


def _notice_llm_error(ft: Any, exc: Optional[BaseException] = None, response: Any = None) -> None:
    """
    Flag the Llm span (and transaction) as errored. Needed because the span is closed
    with __exit__(None, None, None), so the agent never sees the exception itself.
    Works for raised exceptions (exc) and for "completed but failed" responses.
    """
    if ft is None:
        return
    try:
        attrs: dict[str, Any] = {}
        if exc is not None:
            status_code = getattr(exc, "status_code", None)
            body = getattr(exc, "body", None)
            code = getattr(exc, "code", None) or _get(body, "code")
            param = getattr(exc, "param", None) or _get(body, "param")
        else:
            err = _get(response, "error")
            status_code = None
            code = _get(err, "code") or _get(response, "status")
            param = _get(err, "param")
            attrs["error.message"] = _get(err, "message") or f"Response status: {_get(response, 'status')}"
        if status_code is not None:
            attrs["http.statusCode"] = status_code
        if code is not None:
            attrs["error.code"] = str(code)
        if param is not None:
            attrs["error.param"] = str(param)
        # Custom attributes on the span itself (notice_error attributes only reach the error event)
        for k, v in attrs.items():
            ft.add_custom_attribute(k, v)
        if exc is not None:
            ft.notice_error(attributes=attrs)  # uses the in-flight exception
        else:
            try:  # synthesize an exception so the agent gets a real traceback
                raise RuntimeError(attrs["error.message"])
            except RuntimeError:
                ft.notice_error(attributes=attrs)
    except Exception as e:
        logger.warning(f"Could not notice LLM error: {e}")


def _check_failed_response(ft: Any, response: Any) -> None:
    """Flag responses that returned normally but have status 'failed' or an error object."""
    if _get(response, "status") == "failed" or _get(response, "error"):
        _notice_llm_error(ft, response=response)


def instrument_openai_responses(client: Any) -> None:
    """
    Wrap client.responses.create() to emit LlmChatCompletionMessage and
    LlmChatCompletionSummary events.

    Args:
        client: An OpenAI AsyncOpenAI or OpenAI client instance

    Raises:
        AttributeError: If client.responses doesn't exist
    """
    if not hasattr(client, "responses"):
        raise AttributeError(f"{client} does not have a 'responses' attribute")

    original_create = client.responses.create
    original_parse = client.responses.parse

    @functools.wraps(original_create)
    async def create_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Async wrapper for client.responses.create()"""
        ft = _enter_llm_span()
        start_time = time.time()
        call_metadata = _extract_call_metadata(args, kwargs)

        try:
            # Execute the actual call
            response = await original_create(*args, **kwargs)
            elapsed = time.time() - start_time

            # Record telemetry
            _check_failed_response(ft, response)
            _record_response_completion(response, call_metadata, elapsed, error=None)
            return response

        except Exception as exc:
            elapsed = time.time() - start_time
            _notice_llm_error(ft, exc)
            _record_response_completion(None, call_metadata, elapsed, error=exc)
            raise
        finally:
            _exit_llm_span(ft)

    @functools.wraps(original_parse)
    async def parse_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Async wrapper for client.responses.parse()"""
        ft = _enter_llm_span()
        start_time = time.time()
        call_metadata = _extract_call_metadata(args, kwargs)

        try:
            response = await original_parse(*args, **kwargs)
            elapsed = time.time() - start_time
            _check_failed_response(ft, response)
            _record_response_completion(response, call_metadata, elapsed, error=None)
            return response
        except Exception as exc:
            elapsed = time.time() - start_time
            _notice_llm_error(ft, exc)
            _record_response_completion(None, call_metadata, elapsed, error=exc)
            raise
        finally:
            _exit_llm_span(ft)

    # Monkey-patch the methods
    client.responses.create = create_wrapper
    client.responses.parse = parse_wrapper

    logger.info("Instrumented OpenAI Responses API for NR AI Monitoring")


def _extract_call_metadata(args: Any, kwargs: Any) -> dict[str, Any]:
    """
    Extract call parameters from responses.create() args/kwargs.

    Responses API signature:
        client.responses.create(
            model: str,
            instructions: str,  # Optional: system instructions
            input: str | list,  # Required: user input
            max_output_tokens: int,
            temperature: float,
            ...
        )
    """
    # For Responses API, assume keyword arguments (common pattern with OpenAI SDK)
    metadata = {
        "request_model": kwargs.get("model"),
        "instructions": kwargs.get("instructions"),
        "input": kwargs.get("input"),
        "max_output_tokens": kwargs.get("max_output_tokens"),
        "temperature": kwargs.get("temperature"),
        # Trace context (try to get current transaction)
        "trace_id": None,
        "span_id": None,
        # Custom LLM attributes (e.g., gen_ai.conversation_id)
        "llm_attributes": {},
    }

    # Attempt to get trace linking info
    try:
        linking = newrelic.agent.get_linking_metadata()
        metadata["trace_id"] = linking.get("trace.id") or linking.get("trace_id")
        metadata["span_id"] = linking.get("span.id") or linking.get("span_id")
    except Exception as e:
        logger.warning(f"Could not get trace linking metadata: {e}")

    # Attempt to get custom LLM attributes from transaction context
    try:
        current_transaction = newrelic.agent.current_transaction()
        if current_transaction and hasattr(current_transaction, "custom_attributes"):
            # Copy over any attributes starting with "llm." or "gen_ai."
            for key, value in current_transaction.custom_attributes.items():
                if key.startswith(("llm.", "gen_ai.")):
                    metadata["llm_attributes"][key] = value
    except Exception as e:
        logger.debug(f"Could not extract custom LLM attributes: {e}")

    return metadata


def _record_response_completion(
    response: Any, metadata: dict[str, Any], elapsed: float, error: Optional[Exception] = None
) -> None:
    """
    Record LlmChatCompletionMessage and LlmChatCompletionSummary events.

    Converts Responses API response into Chat Completions event schema and emits:
    - One LlmChatCompletionSummary (call-level metadata)
    - N LlmChatCompletionMessage events (one per message)
    """
    import uuid
    import newrelic.agent

    try:
        # Generate IDs
        completion_id = str(uuid.uuid4())

        # Extract response data
        response_data = _parse_response_data(response, error)

        # Build and emit Summary event
        summary_event = _build_summary_event(
            completion_id=completion_id,
            metadata=metadata,
            response_data=response_data,
            elapsed=elapsed,
            error=error,
        )

        try:
            newrelic.agent.record_custom_event("LlmChatCompletionSummary", summary_event)
            logger.debug(
                f"Recorded LlmChatCompletionSummary: completion_id={completion_id}, "
                f"messages={response_data.get('total_messages', 0)}"
            )
        except Exception as e:
            logger.error(f"Failed to record Summary event: {e}")

        # Build and emit Message events
        message_events = _build_message_events(
            completion_id=completion_id,
            metadata=metadata,
            response_data=response_data,
        )

        for i, msg_event in enumerate(message_events):
            try:
                newrelic.agent.record_custom_event("LlmChatCompletionMessage", msg_event)
            except Exception as e:
                logger.error(f"Failed to record Message event {i}: {e}")

        logger.debug(f"Recorded {len(message_events)} LlmChatCompletionMessage events")

    except Exception as e:
        logger.error(f"Error recording completion events: {e}", exc_info=True)
        # Don't raise — we don't want instrumentation to break the actual LLM call


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-key access for pydantic models and dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_response_data(response: Any, error: Optional[Exception] = None) -> dict[str, Any]:
    """
    Extract usage, content, and status from a Responses API response.

    Responses API response shape (streaming):
        {
            "output": [
                {
                    "type": "message" | "function_call",
                    "text": str,
                    "output_text": {"text": str, "delta": str},
                    ...
                },
                ...
            ],
            "model": str,
            "input_tokens": int,
            "output_tokens": int,
            "status": "completed" | "incomplete" | "failed",
            "incomplete_details": {"reason": str},  # if incomplete
        }
    """
    data = {
        "response_model": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_messages": 0,
        "finish_reason": None,
        "content_parts": [],
        "status": "failed" if error else "completed",
    }

    if error or not response:
        return data

    try:
        # SDK returns pydantic models (no .get); mock/raw paths may return dicts
        usage = _get(response, "usage")
        data["response_model"] = _get(response, "model")
        data["input_tokens"] = _get(usage, "input_tokens", 0) or 0
        data["output_tokens"] = _get(usage, "output_tokens", 0) or 0
        data["status"] = _get(response, "status", "completed")

        if data["status"] == "incomplete":
            data["finish_reason"] = _get(_get(response, "incomplete_details"), "reason")
        else:
            data["finish_reason"] = data["status"]

        # Output items: {"type": "message", "content": [{"type": "output_text", "text": ...}]}
        message_count = 1  # the input message
        for item in _get(response, "output", []) or []:
            if _get(item, "type") == "message":
                for part in _get(item, "content", []) or []:
                    text = _get(part, "text")
                    if text:
                        data["content_parts"].append(text)
                message_count += 1

        data["total_messages"] = message_count

    except Exception as e:
        logger.warning(f"Error parsing response data: {e}")

    return data


def _build_summary_event(
    completion_id: str,
    metadata: dict[str, Any],
    response_data: dict[str, Any],
    elapsed: float,
    error: Optional[Exception] = None,
) -> dict[str, Any]:
    """Build LlmChatCompletionSummary event payload."""
    import time

    event = {
        "id": completion_id,
        "vendor": "openai",
        "ingest_source": "Python",
        "duration": int(elapsed * 1000),  # Convert to ms
        "timestamp": int(time.time() * 1000),
        # Trace linking (required)
        "trace_id": metadata.get("trace_id"),
        "span_id": metadata.get("span_id"),
        # Request fields
        "request.model": metadata.get("request_model"),
        "request.max_tokens": metadata.get("max_output_tokens"),
        "request.temperature": metadata.get("temperature"),
        # Response fields
        "response.model": response_data.get("response_model"),
        "response.number_of_messages": response_data.get("total_messages", 0),
        "response.usage.prompt_tokens": response_data.get("input_tokens") or None,
        "response.usage.completion_tokens": response_data.get("output_tokens") or None,
        "response.choices.finish_reason": response_data.get("finish_reason"),
    }

    # Compute total tokens if both are available
    if (
        event["response.usage.prompt_tokens"]
        and event["response.usage.completion_tokens"]
    ):
        event["response.usage.total_tokens"] = (
            event["response.usage.prompt_tokens"]
            + event["response.usage.completion_tokens"]
        )

    # On error, mark it and omit usage/model details
    if error:
        event["error"] = True
        event["response.model"] = None  # May not have succeeded
        event["response.usage.prompt_tokens"] = None
        event["response.usage.completion_tokens"] = None
        event["response.usage.total_tokens"] = None

    elif response_data.get("status") == "failed":
        event["error"] = True

    # Add custom LLM attributes (e.g., gen_ai.conversation_id)
    if metadata.get("llm_attributes"):
        event.update(metadata["llm_attributes"])

    # Remove None/empty values to keep event clean
    return {k: v for k, v in event.items() if v is not None}


def _build_message_events(
    completion_id: str, metadata: dict[str, Any], response_data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build list of LlmChatCompletionMessage events (one per input/output message)."""
    import uuid
    import time

    events = []
    sequence = 0
    timestamp = int(time.time() * 1000)

    # Emit input message (instructions or input param)
    if metadata.get("instructions") or metadata.get("input"):
        input_content = metadata.get("instructions") or str(metadata.get("input", ""))
        input_msg = {
            "id": f"{completion_id}-{sequence}",
            "completion_id": completion_id,
            "sequence": sequence,
            "role": "system" if metadata.get("instructions") else "user",
            "content": input_content if input_content else None,
            "vendor": "openai",
            "ingest_source": "Python",
            "response.model": response_data.get("response_model"),
            "trace_id": metadata.get("trace_id"),
            "span_id": metadata.get("span_id"),
            "timestamp": timestamp,
        }
        # Remove None/empty fields
        input_msg = {k: v for k, v in input_msg.items() if v is not None}
        events.append(input_msg)
        sequence += 1

    # Emit output messages
    for content in response_data.get("content_parts", []):
        output_msg = {
            "id": f"{completion_id}-{sequence}",
            "completion_id": completion_id,
            "sequence": sequence,
            "role": "assistant",
            "content": content if content else None,
            "is_response": True,
            "vendor": "openai",
            "ingest_source": "Python",
            "response.model": response_data.get("response_model"),
            "trace_id": metadata.get("trace_id"),
            "span_id": metadata.get("span_id"),
        }
        # Remove None/empty fields
        output_msg = {k: v for k, v in output_msg.items() if v is not None}
        events.append(output_msg)
        sequence += 1

    return events


# ============================================================================
# Sync API support
# ============================================================================


def instrument_openai_responses_sync(client: Any) -> None:
    """
    Wrap synchronous client.responses.create() / client.responses.parse().
    Use this if the application uses the sync OpenAI client, not AsyncOpenAI.
    """
    if not hasattr(client, "responses"):
        raise AttributeError(f"{client} does not have a 'responses' attribute")

    original_create = client.responses.create
    original_parse = client.responses.parse

    @functools.wraps(original_create)
    def create_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Sync wrapper for client.responses.create()"""
        ft = _enter_llm_span()
        start_time = time.time()
        call_metadata = _extract_call_metadata(args, kwargs)

        try:
            response = original_create(*args, **kwargs)
            elapsed = time.time() - start_time
            _check_failed_response(ft, response)
            _record_response_completion(response, call_metadata, elapsed, error=None)
            return response
        except Exception as exc:
            elapsed = time.time() - start_time
            _notice_llm_error(ft, exc)
            _record_response_completion(None, call_metadata, elapsed, error=exc)
            raise
        finally:
            _exit_llm_span(ft)

    @functools.wraps(original_parse)
    def parse_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Sync wrapper for client.responses.parse()"""
        ft = _enter_llm_span()
        start_time = time.time()
        call_metadata = _extract_call_metadata(args, kwargs)

        try:
            response = original_parse(*args, **kwargs)
            elapsed = time.time() - start_time
            _check_failed_response(ft, response)
            _record_response_completion(response, call_metadata, elapsed, error=None)
            return response
        except Exception as exc:
            elapsed = time.time() - start_time
            _notice_llm_error(ft, exc)
            _record_response_completion(None, call_metadata, elapsed, error=exc)
            raise
        finally:
            _exit_llm_span(ft)

    client.responses.create = create_wrapper
    client.responses.parse = parse_wrapper

    logger.info("Instrumented OpenAI Responses API (sync) for NR AI Monitoring")
