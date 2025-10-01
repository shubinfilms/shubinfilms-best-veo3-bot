import asyncio
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texts import SUNO_START_READY_MESSAGE, t
from tests.suno_test_utils import DummyMessage, bot_module, setup_cover_context
from utils.suno_state import LYRICS_MAX_LENGTH


def _capture_prompt(bot, start_index: int) -> tuple[str, int]:
    prompt_text: str | None = None
    for item in bot.sent[start_index:]:
        text = item.get("text") if isinstance(item, dict) else None
        if isinstance(text, str) and (
            text.startswith("Шаг")
            or text.startswith("✅")
            or text.startswith("Все обязательные поля заполнены")
        ):
            prompt_text = text
    if prompt_text is None:
        raise AssertionError("No prompt message captured")
    return prompt_text, len(bot.sent)


@pytest.mark.parametrize(
    "mode,inputs,expected",
    [
        (
            "instrumental",
            ["Calm focus", "Night Drive"],
            [
                t("suno.prompt.step.style", index=1, total=2, current="—"),
                t("suno.prompt.step.title", index=2, total=2, current="—"),
                SUNO_START_READY_MESSAGE,
            ],
        ),
        (
            "lyrics",
            ["Dream pop", "City Lights", "First line\nSecond line"],
            [
                t("suno.prompt.step.style", index=1, total=3, current="—"),
                t("suno.prompt.step.title", index=2, total=3, current="—"),
                t("suno.prompt.step.lyrics", index=3, total=3, current="—", limit=LYRICS_MAX_LENGTH),
                SUNO_START_READY_MESSAGE,
            ],
        ),
        (
            "cover",
            ["https://example.com/audio.mp3", "Ambient chill", "Cover Title"],
            [
                t("suno.prompt.step.source", index=1, total=3, current="—"),
                t("suno.prompt.step.style", index=2, total=3, current="—"),
                t("suno.prompt.step.title", index=3, total=3, current="—"),
                SUNO_START_READY_MESSAGE,
            ],
        ),
    ],
)
def test_next_step_prompt_flow_per_mode(monkeypatch, mode, inputs, expected):
    async def fake_ensure(url_text: str) -> str:
        return url_text

    async def fake_upload(url_text: str, **_kwargs) -> str:
        return "kie-file"

    monkeypatch.setattr(bot_module, "ensure_cover_audio_url", fake_ensure)
    monkeypatch.setattr(bot_module, "upload_cover_url", fake_upload)

    chat_id = 800 + hash(mode) % 50
    ctx, state_dict, bot = setup_cover_context(chat_id=chat_id)
    bot.sent.clear()
    bot.edited.clear()

    user_id = 1000 + hash(mode) % 100
    asyncio.run(
        bot_module._music_begin_flow(
            chat_id,
            ctx,
            state_dict,
            flow=mode,
            user_id=user_id,
        )
    )

    prompts: list[str] = []
    last_index = 0
    prompt, last_index = _capture_prompt(bot, last_index)
    prompts.append(prompt)

    for value in inputs:
        waiting_field = state_dict.get("suno_waiting_state")
        assert isinstance(waiting_field, str)
        message = DummyMessage(value, chat_id)
        asyncio.run(
            bot_module._handle_suno_waiting_input(
                ctx,
                chat_id,
                message,
                state_dict,
                waiting_field,
                user_id=user_id,
            )
        )
        prompt, last_index = _capture_prompt(bot, last_index)
        prompts.append(prompt)

    assert prompts == expected
