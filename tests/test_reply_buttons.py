import asyncio
from types import SimpleNamespace

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("REDIS_URL", "memory://")
os.environ.setdefault("KIE_API_KEY", "test-key")

import hub_router
from keyboards import TEXT_ACTION_VARIANTS


class StubBot:
    async def edit_message_text(self, **kwargs):  # pragma: no cover - not used
        return SimpleNamespace(message_id=kwargs.get("message_id"))

    async def edit_message_reply_markup(self, **kwargs):  # pragma: no cover - not used
        return SimpleNamespace(message_id=kwargs.get("message_id"))

    async def send_message(self, **kwargs):  # pragma: no cover - not used
        return SimpleNamespace(message_id=kwargs.get("message_id", 42))


@pytest.fixture(autouse=True)
def _clean_router(monkeypatch):
    monkeypatch.setattr(hub_router, "_ROUTES", {})
    monkeypatch.setattr(hub_router, "_FALLBACK_HANDLER", None)
    yield


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Профиль", ("menu", "profile")),
        ("База знаний", ("menu", "kb")),
        ("Фото", ("menu", "photo")),
        ("Музыка", ("menu", "music")),
        ("Видео", ("menu", "video")),
        ("Диалог", ("menu", "dialog")),
    ],
)
def test_reply_button_routes_text_dispatch(text, expected):
    calls = []

    @hub_router.register(*expected)
    async def _handler(ctx: hub_router.CallbackContext) -> None:  # type: ignore[override]
        calls.append((ctx.namespace, ctx.action))

    ctx = SimpleNamespace(
        bot=StubBot(),
        user_data={},
        application=SimpleNamespace(logger=None),
    )
    message = SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=123),
        message_id=77,
    )
    update = SimpleNamespace(
        effective_message=message,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=999),
        callback_query=None,
    )

    asyncio.run(hub_router.route_text(update, ctx))

    assert calls == [expected]


@pytest.mark.parametrize("label", list(TEXT_ACTION_VARIANTS.keys()))
def test_text_action_variants_cover_reply_labels(label):
    normalized = hub_router.resolve_text_action(label)
    assert normalized is not None
