import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LEDGER_BACKEND", "memory")

from tests.suno_test_utils import FakeBot, bot_module  # noqa: E402


def _make_context(bot: FakeBot | None = None) -> SimpleNamespace:
    return SimpleNamespace(bot=bot or FakeBot(), user_data={}, args=[])


def _make_update(user_id: int, chat_id: int, *, text: str = "/start") -> SimpleNamespace:
    message = SimpleNamespace(
        text=text,
        caption=None,
        chat_id=chat_id,
    )
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
        message=message,
        effective_message=message,
        callback_query=None,
    )


def _assert_main_menu_payload(payload: dict, expected_balance: int) -> None:
    assert payload["text"] == bot_module._build_main_menu_text(expected_balance)
    assert payload.get("parse_mode") == bot_module.ParseMode.HTML
    markup = payload["reply_markup"]
    rows = markup.inline_keyboard
    for row in rows:
        for btn in row:
            assert btn.callback_data.startswith("hub:"), "callback prefix must be hub:"
            assert len(btn.callback_data.encode("utf-8")) <= 64, "callback data too long"
    assert [[btn.callback_data for btn in row] for row in rows] == [
        ["hub:video", "hub:image"],
        ["hub:music", "hub:balance"],
        ["hub:lang", "hub:faq"],
        ["hub:help"],
    ]


def test_menu_command(monkeypatch):
    ctx = _make_context()
    bot = ctx.bot

    async def fake_hub(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot_module, "show_emoji_hub_for_chat", fake_hub)

    update = _make_update(user_id=501, chat_id=777, text="/menu")
    asyncio.run(bot_module.on_menu(update, ctx))

    menu_messages = [
        payload
        for payload in bot.sent
        if isinstance(payload, dict)
        and payload.get("text", "").startswith("👋 Добро пожаловать!")
    ]

    assert menu_messages, "main menu message should be sent"
    _assert_main_menu_payload(menu_messages[-1], expected_balance=0)
    assert all("🎁" not in str(payload.get("text", "")) for payload in bot.sent)


def test_start_flow_without_bonus(monkeypatch):
    ctx = _make_context()
    bot = ctx.bot

    async def fake_hub(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot_module, "show_emoji_hub_for_chat", fake_hub)

    update = _make_update(user_id=9001, chat_id=42)
    asyncio.run(bot_module.on_start(update, ctx))

    bonus_messages = [
        payload
        for payload in bot.sent
        if isinstance(payload, dict)
        and "🎁" in str(payload.get("text", ""))
    ]
    assert not bonus_messages, "bonus message should not be sent"

    menu_messages = [
        payload
        for payload in bot.sent
        if isinstance(payload, dict)
        and payload.get("text", "").startswith("👋 Добро пожаловать!")
    ]
    assert menu_messages, "main menu message should be sent"
    _assert_main_menu_payload(menu_messages[-1], expected_balance=0)
