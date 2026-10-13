# OpenAI Responses API Instrumentation for New Relic (Temporary Workaround)

> **Status: ⚠️ This is a temporary workaround until the NR APM Python agent supports Responses API calls.**
>
> This is provided as-is and is not officially supported by New Relic's support team. For issues or questions, please refer to the documentation or open an issue on GitHub.

## TL;DR
The New Relic Python agent does not yet instrument the OpenAI **Responses API** (`client.responses.create()`), so AI Monitoring shows nothing for apps (e.g. Strands agents) that use it. This small decorator wraps those calls and emits the same AI Monitoring events (`LlmChatCompletionSummary`, `LlmChatCompletionMessage`) plus an `Llm/completion/OpenAI/create` span. A working demo is available on request.

## Requirements
- Python 3.10+
- New Relic Python agent 14.0.0+
- Strands Agents framework using an OpenAI client with the Responses API
- `openai` SDK

## Installation
```bash
pip install git+https://github.com/rossfromwork/newrelic-openai-responses-decorator.git
```

## Usage
```python
from openai import AsyncOpenAI
from newrelic_openai_responses_decorator import instrument_openai_responses

client = AsyncOpenAI()
instrument_openai_responses(client)   # use instrument_openai_responses_sync() for OpenAI()
# pass `client` to Strands as usual; calls inside a New Relic transaction are recorded
```
See `examples/basic_usage.py`.

## Compatibility
Tested with the Python agent 14.x and the Responses API (`create`, `parse`; non-streaming). Streaming responses are not instrumented. Call `instrument_*` once per client, after the agent is initialized.

## Troubleshooting
- **No events in New Relic:** calls must run inside an active transaction (web request or `@background_task`), and AI Monitoring must be enabled (`ai_monitoring.enabled = true`).
- **`AttributeError ... 'responses'`:** upgrade the `openai` SDK.
- **Sync vs async:** use the matching function for your client type.
- **Content missing:** events appear but message content (prompt/response text) is absent. This usually means `ai_monitoring.record_content.enabled` is `false`, so content is not captured. Set it to `true` to capture content.

## Support
Contact your New Relic account team. For more details, see the repository issues or documentation.
