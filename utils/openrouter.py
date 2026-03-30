from __future__ import annotations

import difflib
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass
class OpenRouterError(RuntimeError):
    message: str
    available_providers: list[str] | None = None

    def __str__(self) -> str:
        if self.available_providers:
            return f"{self.message} | available_providers={','.join(self.available_providers)}"
        return self.message


def _headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_REFERER")
    title = os.getenv("OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _extract_error(payload_text: str) -> tuple[str, list[str] | None]:
    providers: list[str] | None = None
    message = payload_text
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return message, providers

    err = payload.get("error", payload)
    if isinstance(err, dict):
        message = str(err.get("message", payload_text))
        metadata = err.get("metadata", {})
        if isinstance(metadata, dict):
            raw_providers = metadata.get("providers")
            if isinstance(raw_providers, list):
                providers = [str(x) for x in raw_providers]
        if providers is None:
            raw = err.get("providers")
            if isinstance(raw, list):
                providers = [str(x) for x in raw]
    return message, providers


def _request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout: int,
) -> dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        payload_text = exc.read().decode("utf-8", errors="replace")
        message, providers = _extract_error(payload_text)
        raise OpenRouterError(message, providers) from exc
    except error.URLError as exc:
        raise OpenRouterError(f"Network error: {exc}") from exc


def chat(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 60,
    provider: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider:
        payload["provider"] = provider

    return _request_json(
        method="POST",
        url=f"{base_url}/chat/completions",
        headers=_headers(api_key),
        payload=payload,
        timeout=timeout,
    )


def chat_text(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: int = 60,
    system_prompt: str | None = None,
    provider: dict[str, Any] | None = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = chat(
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        provider=provider,
    )
    return (
        response.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


def list_models(api_key: str, timeout: int = 60) -> list[str]:
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    payload = _request_json(
        method="GET",
        url=f"{base_url}/models",
        headers=_headers(api_key),
        payload=None,
        timeout=timeout,
    )
    model_ids: list[str] = []
    for item in payload.get("data", []):
        if isinstance(item, dict):
            model_id = item.get("id")
            if model_id:
                model_ids.append(str(model_id))
    return model_ids


def validate_model_id(api_key: str, model: str) -> tuple[bool, list[str]]:
    try:
        model_ids = list_models(api_key)
    except OpenRouterError:
        # If model listing fails, do not hard-block execution.
        return True, []

    if model in model_ids:
        return True, []
    suggestions = difflib.get_close_matches(model, model_ids, n=5, cutoff=0.45)
    return False, suggestions
