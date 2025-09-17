#!/usr/bin/env python3
"""Smoke tests for Best VEO3 Bot runtime."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List

import requests
from dotenv import load_dotenv
from unittest import mock


@dataclass
class StepResult:
    name: str
    success: bool
    detail: str


class SmokeError(RuntimeError):
    """Raised when a smoke step fails."""


def _load_env() -> None:
    load_dotenv(override=False)


def _healthz_url() -> str:
    explicit = os.getenv("SMOKE_HEALTHZ_URL")
    if explicit:
        return explicit
    port = os.getenv("HEALTHZ_PORT") or os.getenv("PORT") or "8080"
    host = os.getenv("SMOKE_HEALTHZ_HOST") or os.getenv("HEALTHZ_HOST") or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    scheme = os.getenv("SMOKE_HEALTHZ_SCHEME", "http")
    return f"{scheme}://{host}:{port}/healthz"


def _check_healthz() -> str:
    url = _healthz_url()
    try:
        resp = requests.get(url, timeout=5)
    except requests.RequestException as exc:
        raise SmokeError(f"healthz request failed: {exc}") from exc
    if resp.status_code != 200:
        raise SmokeError(f"healthz returned status {resp.status_code}: {resp.text}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise SmokeError(f"healthz invalid JSON: {resp.text}") from exc
    if not payload.get("ok"):
        raise SmokeError(f"healthz responded with ok=false: {json.dumps(payload, ensure_ascii=False)}")
    return json.dumps(payload, ensure_ascii=False)


def _require_env(name: str, *fallbacks: str) -> str:
    keys = (name, *fallbacks)
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            if key != name:
                return value.strip()
            return value.strip()
    raise SmokeError(f"Environment variable {name} is required")


def _send_test_message() -> str:
    token = _require_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN")
    chat_id = _require_env("ADMIN_CHAT_ID", "ADMIN_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"Smoke test ping at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "disable_notification": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as exc:
        raise SmokeError(f"Telegram sendMessage failed: {exc}") from exc
    if resp.status_code != 200:
        raise SmokeError(f"Telegram sendMessage HTTP {resp.status_code}: {resp.text}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise SmokeError(f"Telegram sendMessage invalid JSON: {resp.text}") from exc
    if not data.get("ok"):
        raise SmokeError(f"Telegram sendMessage ok=false: {json.dumps(data, ensure_ascii=False)}")
    return f"message_id={data.get('result', {}).get('message_id')}"


def _exercise_modes() -> str:
    import bot
    import prompt_master
    from kie_banana import create_banana_task, wait_for_banana_result

    class FakeResponse:
        def __init__(self, payload: dict, status: int = 200):
            self._payload = payload
            self.status_code = status
            self.text = json.dumps(payload, ensure_ascii=False)

        def json(self) -> dict:
            return self._payload

    def fake_request(*_args, **_kwargs) -> FakeResponse:
        return FakeResponse({"code": 200, "data": {"taskId": "smoke-task"}})

    with mock.patch("bot.requests.request", side_effect=fake_request):
        veo_ok, veo_task, _ = bot.submit_kie_veo("Smoke test prompt", "16:9", None, "veo3")
        if not veo_ok or not veo_task:
            raise SmokeError("VEO submission failed")
        mj_ok, mj_task, _ = bot.mj_generate("Smoke test prompt", "16:9")
        if not mj_ok or not mj_task:
            raise SmokeError("MJ submission failed")

    with mock.patch("kie_banana._post_json", return_value=(200, {"code": 200, "data": {"taskId": "banana-smoke"}})), \
         mock.patch("kie_banana._get_json", return_value=(200, {"code": 200, "data": {"state": "success", "resultUrls": ["https://example.com/demo.png"]}})):
        task_id = create_banana_task("Smoke banana", ["https://example.com/demo.png"])
        urls = wait_for_banana_result(task_id, timeout_sec=1, poll_sec=0)
        if not urls:
            raise SmokeError("Banana result empty")

    with mock.patch("prompt_master._ask_openai", return_value="Prompt master ok"):
        pm_text = prompt_master.generate_prompt_master("Smoke test scene")
        if not pm_text:
            raise SmokeError("Prompt Master response empty")

    return "veo,mj,banana,prompt_master"


def run_steps(steps: List[tuple[str, Callable[[], str]]]) -> List[StepResult]:
    results: List[StepResult] = []
    for name, func in steps:
        try:
            detail = func()
            results.append(StepResult(name, True, detail))
        except Exception as exc:  # pragma: no cover - runtime errors
            results.append(StepResult(name, False, str(exc)))
    return results


def main() -> int:
    _load_env()

    steps = [
        ("healthz", _check_healthz),
        ("telegram", _send_test_message),
        ("modes", _exercise_modes),
    ]

    results = run_steps(steps)
    all_ok = all(r.success for r in results)

    print("Smoke test report:")
    for result in results:
        status = "OK" if result.success else "FAIL"
        print(f" - {result.name}: {status} ({result.detail})")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
