import asyncio
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("REDIS_URL", "memory://")
os.environ.setdefault("KIE_API_KEY", "test-key")

import hub_router
import bot as bot_module
import handlers.sum as sum_module
import ui.buttons.router as router
from keyboards import TEXT_ACTION_VARIANTS
from utils.input_state import clear_wait_state, set_wait


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


def test_profile_button_debounced_ack(monkeypatch):
    router._LAST_CALLBACK_AT.clear()
    monkeypatch.setattr(router, "DEBOUNCE_SEC", 0.6)

    ack_calls: list[str] = []
    dispatch_calls: list[tuple[str, str]] = []

    async def fake_safe_answer(query, cache_time=0):  # type: ignore[override]
        ack_calls.append(query.data)
        return "ok"

    async def fake_dispatch(action, update, ctx, *, raw_data, payload=None, legacy=False):  # type: ignore[override]
        dispatch_calls.append(("registry", action))

    async def fake_namespace(normalized, update, ctx):  # type: ignore[override]
        dispatch_calls.append(("namespace", f"{normalized.namespace}:{normalized.action}"))

    monkeypatch.setattr(router, "safe_answer", fake_safe_answer)
    monkeypatch.setattr(router, "dispatch_via_registry", fake_dispatch)
    monkeypatch.setattr(router, "should_process", lambda user_id, key: True)
    monkeypatch.setattr(router, "_dispatch_namespace_callback", fake_namespace)

    query = SimpleNamespace(data="btn:profile", _ui_button_handled=False)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=1),
        effective_user=SimpleNamespace(id=42),
    )

    asyncio.run(router.on_callback(update, SimpleNamespace()))
    assert dispatch_calls == [("namespace", "profile:open")]

    dispatch_calls.clear()
    ack_calls.clear()
    router._LAST_CALLBACK_AT[(42, "btn:profile")] = router.time.monotonic()

    asyncio.run(router.on_callback(update, SimpleNamespace()))

    assert dispatch_calls == []
    assert len(ack_calls) == 1


def test_sum_text_routed_to_sum(monkeypatch):
    user_id = 101
    chat_id = 202

    events: list[tuple[str, str, int]] = []

    async def fake_handle(ctx, message, cleaned_text, wait_state, *, user_id=None):  # type: ignore[override]
        events.append((cleaned_text, wait_state.kind.value, int(user_id or 0)))
        return True

    monkeypatch.setattr(sum_module, "handle_wait_input", fake_handle)

    class StubMessage:
        def __init__(self) -> None:
            self.chat_id = chat_id
            self.text = "Привет"
            self.message_id = 77
            self.replies: list[str] = []

        async def reply_text(self, text, **kwargs):  # type: ignore[override]
            self.replies.append(text)

    message = StubMessage()
    update = SimpleNamespace(
        effective_message=message,
        message=message,
        effective_user=SimpleNamespace(id=user_id),
    )

    ctx = SimpleNamespace(bot=None, user_data={})

    set_wait(user_id, "sum_prompt", message.message_id, chat_id=chat_id, meta={})

    try:
        asyncio.run(bot_module.handle_card_input(update, ctx))
    finally:
        clear_wait_state(user_id)

    assert events == [("Привет", "sum_prompt", user_id)]
    assert "✅ Принято" in message.replies


def test_sum_start_without_draft(monkeypatch):
    alert_calls: list[dict[str, object]] = []

    class StubQuery:
        async def answer(self, text=None, show_alert=False):  # type: ignore[override]
            alert_calls.append({"text": text, "show_alert": show_alert})

    callback = SimpleNamespace(
        query=StubQuery(),
        user_id=303,
        chat_id=404,
        application_context=SimpleNamespace(bot=None),
        card_message_id=None,
        update=SimpleNamespace(),
    )

    class _CounterStub:
        def labels(self, **kwargs):  # type: ignore[override]
            return self

        def inc(self):  # type: ignore[override]
            return None

    monkeypatch.setattr(sum_module, "_acquire_run", lambda user_id: True)
    monkeypatch.setattr(sum_module, "_release_run", lambda user_id: None)
    monkeypatch.setattr(sum_module, "_load_payload", lambda user_id: None)
    monkeypatch.setattr(sum_module, "sum_start_total", _CounterStub())

    asyncio.run(sum_module.start_from_callback(callback))

    assert alert_calls and alert_calls[-1] == {
        "text": "Нет текста для суммирования.",
        "show_alert": True,
    }


def test_banana_sensitive_hidden_from_user(monkeypatch):
    messages: list[str] = []
    admin_alerts: list[str] = []

    class StubBot:
        async def send_message(self, chat_id, text, **kwargs):  # type: ignore[override]
            messages.append(text)

    async def fake_show_balance_notification(*args, **kwargs):  # pragma: no cover - test stub
        return None

    async def fake_show_banana_card(*args, **kwargs):  # pragma: no cover - test stub
        return None

    def fake_credit_balance(user_id, amount, *, reason, meta):  # type: ignore[override]
        return 95

    def fake_create_task(*args, **kwargs):
        raise bot_module.KieBananaError("banana fail: code=422 E005")

    monkeypatch.setattr(bot_module, "state", lambda ctx: {})
    monkeypatch.setattr(bot_module, "show_banana_card", fake_show_banana_card)
    monkeypatch.setattr(bot_module, "show_balance_notification", fake_show_balance_notification)
    monkeypatch.setattr(bot_module, "credit_balance", fake_credit_balance)
    monkeypatch.setattr(bot_module, "create_banana_task", fake_create_task)
    monkeypatch.setattr(bot_module, "wait_for_banana_result", lambda *args, **kwargs: [])
    async def fake_notify_admin(bot, text):  # type: ignore[override]
        admin_alerts.append(text)

    monkeypatch.setattr(bot_module, "notify_admin", fake_notify_admin)

    ctx = SimpleNamespace(bot=StubBot())

    asyncio.run(
        bot_module._banana_run_and_send(
            chat_id=111,
            ctx=ctx,
            prompt="demo",
            price=5,
            user_id=222,
        )
    )

    assert messages and "422" not in messages[-1]
    assert "Нельзя обработать" in messages[-1]
    assert admin_alerts and "code=422" in admin_alerts[-1]
