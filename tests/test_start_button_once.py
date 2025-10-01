import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tests.suno_test_utils import DummyMessage, FakeBot, bot_module


def _collect_start_buttons(bot: FakeBot) -> list[dict[str, object]]:
    buttons: list[dict[str, object]] = []
    for payload in bot.sent:
        if not isinstance(payload, dict):
            continue
        markup = payload.get("reply_markup")
        if not markup or not getattr(markup, "inline_keyboard", None):
            continue
        button = markup.inline_keyboard[0][0]
        if getattr(button, "text", "") == "▶️ Начать генерацию":
            buttons.append(payload)
    return buttons


def test_start_button_rendered_once(monkeypatch):
    chat_id = 1200
    user_id = 1300
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})

    asyncio.run(
        bot_module._music_begin_flow(
            chat_id,
            ctx,
            bot_module.state(ctx),
            flow="instrumental",
            user_id=user_id,
        )
    )

    state_dict = bot_module.state(ctx)

    waiting = state_dict.get("suno_waiting_state")
    assert waiting == bot_module.WAIT_SUNO_TITLE
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            DummyMessage("Skyline", chat_id),
            state_dict,
            waiting,
            user_id=user_id,
        )
    )

    waiting = state_dict.get("suno_waiting_state")
    assert waiting == bot_module.WAIT_SUNO_STYLE
    asyncio.run(
        bot_module._handle_suno_waiting_input(
            ctx,
            chat_id,
            DummyMessage("Dream pop", chat_id),
            state_dict,
            waiting,
            user_id=user_id,
        )
    )

    start_buttons = _collect_start_buttons(bot)
    assert len(start_buttons) == 1

    asyncio.run(bot_module.refresh_suno_card(ctx, chat_id, state_dict, price=bot_module.PRICE_SUNO))
    assert len(_collect_start_buttons(bot)) == 1

    start_msg_id = state_dict.get("suno_start_msg_id")
    assert isinstance(start_msg_id, int)

    monkeypatch.setattr(bot_module, "START_EMOJI_STICKER_ID", "sticker-one")

    async def fake_launch(*_args, **_kwargs):  # type: ignore[override]
        return None

    async def fake_notify(*_args, **_kwargs):  # type: ignore[override]
        return None

    monkeypatch.setattr(bot_module, "_launch_suno_generation", fake_launch)
    monkeypatch.setattr(bot_module, "_suno_notify", fake_notify)

    callback = SimpleNamespace(
        data="suno:start",
        message=SimpleNamespace(chat_id=chat_id, replies=[]),
        _answered=[],
    )

    async def answer(text: str | None = None, show_alert: bool = False):  # type: ignore[override]
        callback._answered.append((text, show_alert))

    callback.answer = answer

    update = SimpleNamespace(
        callback_query=callback,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )

    asyncio.run(bot_module.on_callback(update, ctx))

    disabled_buttons = [
        payload
        for payload in bot.edited
        if payload.get("message_id") == start_msg_id
        and getattr(payload.get("reply_markup"), "inline_keyboard", None)
    ]
    assert disabled_buttons
    disabled_button = disabled_buttons[-1]["reply_markup"].inline_keyboard[0][0]
    assert disabled_button.text == "⏳ Идёт генерация…"

    assert len(_collect_start_buttons(bot)) == 1
