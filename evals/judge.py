
import asyncio
import json
import time
import threading

from groq import Groq, RateLimitError
from deepeval.models import DeepEvalBaseLLM

from config import (
    GROQ_API_KEY,
    GROQ_EVAL_MODEL,
    validate_groq_config,
)


MIN_REQUEST_INTERVAL = 10.0
MAX_RETRIES = 3
MAX_TOKENS = 2048

_request_lock = threading.Lock()
_last_request_time = 0.0


def wait_for_request_slot():
    global _last_request_time

    with _request_lock:
        wait = MIN_REQUEST_INTERVAL - (
            time.monotonic() - _last_request_time
        )

        if wait > 0:
            time.sleep(wait)

        _last_request_time = time.monotonic()


class GroqEvaluationLLM(DeepEvalBaseLLM):

    def __init__(self, model_name=None):
        validate_groq_config()

        self.model_name = model_name or GROQ_EVAL_MODEL

        if self.model_name.lower().startswith("gemini"):
            raise ValueError(
                "Use a Groq-compatible model in GROQ_EVAL_MODEL."
            )

        self.client = Groq(
            api_key=GROQ_API_KEY,
            max_retries=0,
            timeout=90.0,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None):

        schema_json = (
            schema.model_json_schema()
            if schema is not None
            else None
        )

        current_prompt = prompt

        for attempt in range(MAX_RETRIES + 1):
            try:
                wait_for_request_slot()

                system_prompt = (
                    "You are a precise evaluation judge. "
                    "Return valid JSON only. "
                    "Include every required field. "
                    "Never omit required fields or invent information."
                )

                if schema_json is not None:
                    system_prompt += (
                        "\nRequired JSON schema:\n"
                        + json.dumps(schema_json)
                    )

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": current_prompt,
                        },
                    ],
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                )

                choice = response.choices[0]
                result = choice.message.content

                if not result:
                    raise ValueError("Groq returned an empty response.")

                if choice.finish_reason == "length":
                    raise ValueError(
                        "Groq response was truncated. Increase MAX_TOKENS."
                    )

                if schema is None:
                    return result.strip()

                try:
                    return schema.model_validate_json(result)

                except Exception as error:
                    if attempt == MAX_RETRIES:
                        raise ValueError(
                            f"Groq JSON failed schema validation: {error}"
                        ) from error

                    current_prompt = (
                        prompt
                        + "\n\nYour previous JSON response failed "
                        "schema validation."
                        + f"\nValidation error: {error}"
                        + "\nReturn corrected JSON. Include every "
                        "required field in every object."
                    )

            except RateLimitError as error:
                message = str(error).lower()

                if any(term in message for term in (
                    "requests per day",
                    "tokens per day",
                    "daily limit",
                    "rpd",
                    "tpd",
                )):
                    raise RuntimeError(
                        "Groq daily quota exhausted. Wait for quota reset."
                    ) from error

                if attempt == MAX_RETRIES:
                    raise

                headers = getattr(
                    getattr(error, "response", None),
                    "headers",
                    {},
                ) or {}

                try:
                    wait = float(headers.get("retry-after", 0))
                except (TypeError, ValueError):
                    wait = 0

                time.sleep(
                    wait or min(10 * (2 ** attempt), 40)
                )

        raise RuntimeError("Groq evaluation failed.")

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(
            self.generate,
            prompt,
            schema,
        )

    def get_model_name(self):
        return self.model_name
