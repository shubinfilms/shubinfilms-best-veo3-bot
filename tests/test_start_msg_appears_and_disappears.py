import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texts import SUNO_START_READY_MESSAGE
from ui_helpers import sync_suno_start_message
from utils.suno_state import SunoState

from tests.suno_test_utils import FakeBot


def test_start_msg_appears_and_disappears():
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot)
    suno_state = SunoState(mode="cover")
    state_dict = {"suno_state": suno_state.to_dict(), "msg_ids": {}}
    chat_id = 123

    # Initial call without readiness should not send anything.
    result_initial = asyncio.run(
        sync_suno_start_message(
            ctx,
            chat_id,
            state_dict,
            suno_state=suno_state,
            ready=False,
            generating=False,
            waiting_enqueue=False,
        )
    )
    assert result_initial is None
    assert not bot.sent

    # Ready state should trigger a new message.
    result_ready = asyncio.run(
        sync_suno_start_message(
            ctx,
            chat_id,
            state_dict,
            suno_state=suno_state,
            ready=True,
            generating=False,
            waiting_enqueue=False,
        )
    )
    assert isinstance(result_ready, int)
    assert bot.sent[-1]["text"] == SUNO_START_READY_MESSAGE
    assert state_dict["suno_start_msg_id"] == result_ready
    assert suno_state.start_msg_id == result_ready

    # Losing readiness should remove the message.
    result_hidden = asyncio.run(
        sync_suno_start_message(
            ctx,
            chat_id,
            state_dict,
            suno_state=suno_state,
            ready=False,
            generating=False,
            waiting_enqueue=False,
        )
    )
    assert result_hidden is None
    assert bot.deleted[-1]["message_id"] == result_ready
    assert state_dict.get("suno_start_msg_id") is None
    assert suno_state.start_msg_id is None
