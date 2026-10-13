"""Minimal example: instrument an AsyncOpenAI client before using it with Strands.

Run with:  NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program python examples/basic_usage.py
"""
import asyncio

import newrelic.agent
from openai import AsyncOpenAI

from newrelic_openai_responses_decorator import instrument_openai_responses

client = AsyncOpenAI()  # reads OPENAI_API_KEY
instrument_openai_responses(client)  # patches client.responses.create / parse


@newrelic.agent.background_task(name="responses-demo")
async def main():
    response = await client.responses.create(model="gpt-4o", input="Say hello in one sentence.")
    print(response.output_text)


if __name__ == "__main__":
    newrelic.agent.initialize()
    asyncio.run(main())
