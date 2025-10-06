import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.suno_state import load as load_suno_state, save as save_suno_state, set_style as set_suno_style, set_title as set_suno_title

from tests.suno_test_utils import FakeBot, bot_module


class FakeMessage:
    def __init__(self, chat_id: int) -> None:
        self.chat_id = chat_id
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_kwargs):  # type: ignore[override]
        self.replies.append(text)
        return SimpleNamespace(message_id=400)


class FakeCallback:
    def __init__(self, chat_id: int) -> None:
        self.data = "music:start"
        self.message = FakeMessage(chat_id)
        self._answered: list[tuple[str | None, bool]] = []

    async def answer(self, text: str | None = None, show_alert: bool = False):  # type: ignore[override]
        self._answered.append((text, show_alert))


def test_start_calls_existing_handler(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})
    state_dict = bot_module.state(ctx)

    chat_id = 999
    user_id = 555
    state_dict["mode"] = "suno"

    suno_state = load_suno_state(ctx)
    suno_state.mode = "instrumental"
    set_suno_title(suno_state, "Ready Song")
    set_suno_style(suno_state, "ambient focus")
    save_suno_state(ctx, suno_state)
    state_dict["suno_state"] = suno_state.to_dict()

    captured: dict[str, object] = {}

    async def fake_launch(chat_id_param, ctx_param, **kwargs):  # type: ignore[override]
        captured["launch"] = {
            "chat_id": chat_id_param,
            "user_id": kwargs.get("user_id"),
            "trigger": kwargs.get("trigger"),
        }

    async def fake_notify(*_args, **_kwargs):  # type: ignore[override]
        return None

    def fake_acquire(_user_id: int) -> bool:
        captured.setdefault("acquired", True)
        return True

    def fake_release(_user_id: int) -> None:
        captured["released"] = True

    monkeypatch.setattr(bot_module, "_launch_suno_generation", fake_launch)
    monkeypatch.setattr(bot_module, "_suno_notify", fake_notify)
    monkeypatch.setattr(bot_module, "_acquire_suno_lock", fake_acquire)
    monkeypatch.setattr(bot_module, "_release_suno_lock", fake_release)

    update = SimpleNamespace(
        callback_query=FakeCallback(chat_id),
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
    )

    asyncio.run(bot_module.on_callback(update, ctx))

    assert "launch" in captured
    launch_meta = captured["launch"]
    assert launch_meta["chat_id"] == chat_id
    assert launch_meta["user_id"] == user_id
    assert launch_meta["trigger"] == "start"
    assert captured.get("acquired") is True
    assert captured.get("released") is True
