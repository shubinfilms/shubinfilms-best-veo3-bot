import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module


def _build_update(data: str, *, chat_id: int = 555, message_id: int = 777):
    async def fake_answer():
        return None

    message = SimpleNamespace(chat_id=chat_id, message_id=message_id)
    query = SimpleNamespace(data=data, message=message, answer=fake_answer)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=None,
    )
    return update


def test_old_topup_callbacks_still_routed(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={})

    calls = []

    async def fake_edit(*_args, **_kwargs):
        calls.append(_kwargs.get("reply_markup"))
        return None

    monkeypatch.setattr(bot_module, "_safe_edit_message_text", fake_edit)

    update_stars = _build_update("topup:stars")
    update_card = _build_update("topup:yookassa")

    handled_stars = asyncio.run(bot_module.handle_topup_callback(update_stars, ctx, "topup:stars"))
    handled_card = asyncio.run(bot_module.handle_topup_callback(update_card, ctx, "topup:yookassa"))

    assert handled_stars is True
    assert handled_card is True
    assert len(calls) == 2
