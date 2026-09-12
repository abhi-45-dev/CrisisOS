from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import settings
from app.exceptions import LLMError
from app.logging_config import logger

T = TypeVar("T", bound=BaseModel)


class LLMProvider:
    name = "base"

    @property
    def available(self) -> bool:
        return False

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise LLMError("LLM provider is not configured")


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self) -> None:
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self._client = None
        if self.api_key:
            try:
                from groq import Groq

                self._client = Groq(api_key=self.api_key, timeout=self.timeout)
            except Exception as exc:  # pragma: no cover - import/runtime guard
                logger.error("Failed to initialize Groq client: %s", exc)
                self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key) and self._client is not None

    def generate(self, prompt: str, system: str | None = None) -> str:
        if not self.available:
            raise LLMError("GROQ_API_KEY is missing or Groq client failed to initialize")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            logger.info("Calling Groq model %s", self.model)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as exc:
            logger.error("Groq API failure: %s", exc)
            raise LLMError("LLM request failed") from exc


class LLMService:
    def __init__(self, provider: LLMProvider | None = None):
        if provider is not None:
            self.provider = provider
        elif settings.LLM_PROVIDER.lower() == "groq":
            self.provider = GroqProvider()
        else:
            self.provider = LLMProvider()

    @property
    def available(self) -> bool:
        return self.provider.available

    def generate(self, prompt: str, system: str | None = None) -> str:
        return self.provider.generate(prompt, system=system)

    def generate_structured(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        raw = self.generate(prompt, system=system)
        payload = extract_json(raw)
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            logger.error("Malformed LLM JSON for %s: %s", schema.__name__, exc)
            raise LLMError("LLM returned malformed structured output") from exc


def extract_json(text: str) -> Any:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM response was not valid JSON") from exc


llm_service = LLMService()
