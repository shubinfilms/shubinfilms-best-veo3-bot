import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest
from telegram.constants import ChatType

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from handlers.prompt_master_handler import (
    PM_ENGINE_KEY,
    PM_STATE_KEY,
    detect_language,
    prompt_master_callback,
    prompt_master_handle_text,
    prompt_master_open,
)
from keyboards import (
    CB_PM_PREFIX,
    prompt_master_keyboard,
    prompt_master_mode_keyboard,
)
from prompt_master import Engine, build_prompt


class DummyStatusMessage:
    def __init__(self) -> None:
        self.edits = []

    async def edit_text(self, text, **kwargs):  # pragma: no cover - simple recorder
        self.edits.append((text, kwargs))


class DummyMessage:
    def __init__(self, text: str, chat_type=ChatType.PRIVATE) -> None:
        self.text = text
        self.chat = SimpleNamespace(id=123, type=chat_type)
        self.reply_calls = []
        self.deleted = False
        self.status = DummyStatusMessage()

    async def reply_text(self, text, **kwargs):
        self.reply_calls.append((text, kwargs))
        return self.status

    async def delete(self):  # pragma: no cover - deletion recording
        self.deleted = True


def test_prompt_master_keyboard_layout_ru() -> None:
    keyboard = prompt_master_keyboard("ru")
    layout = [
        (button.text, button.callback_data)
        for row in keyboard.inline_keyboard
        for button in row
    ]
    expected = [
        ("🎬 Видеопромпт (VEO)", f"{CB_PM_PREFIX}veo"),
        ("🖼️ Изображение (Midjourney)", f"{CB_PM_PREFIX}mj"),
        ("🫥 Оживление фото", f"{CB_PM_PREFIX}animate"),
        ("✂️ Редактирование фото (Banana)", f"{CB_PM_PREFIX}banana"),
        ("🎵 Трек (Suno)", f"{CB_PM_PREFIX}suno"),
        ("⬅️ Назад", f"{CB_PM_PREFIX}back"),
    ]
    assert layout == expected


def test_prompt_master_open_replies_with_keyboard_html() -> None:
    calls = []

    async def fake_reply(text, **kwargs):
        calls.append((text, kwargs))

    message = SimpleNamespace(reply_text=fake_reply)
    user = SimpleNamespace(language_code="ru")
    update = SimpleNamespace(
        effective_message=message,
        message=message,
        callback_query=None,
        effective_user=user,
    )
    ctx = SimpleNamespace(user_data={})

    asyncio.run(prompt_master_open(update, ctx))

    assert calls
    _text, kwargs = calls[0]
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_keyboard("ru").inline_keyboard
    assert kwargs["parse_mode"] == "HTML"


def test_prompt_master_callback_sets_pm_state() -> None:
    edits = []

    async def fake_edit(text, **kwargs):
        edits.append((text, kwargs))

    async def fake_answer(*args, **kwargs):
        pass

    query = SimpleNamespace(
        data=f"{CB_PM_PREFIX}veo",
        answer=fake_answer,
        edit_message_text=fake_edit,
        message=SimpleNamespace(chat=SimpleNamespace(id=1)),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=42, language_code="ru"),
        effective_chat=SimpleNamespace(id=1),
    )
    ctx = SimpleNamespace(user_data={})

    asyncio.run(prompt_master_callback(update, ctx))

    assert ctx.user_data.get("mode") == "pm"
    assert ctx.user_data.get("pm_engine") == "veo"
    assert edits
    _text, kwargs = edits[0]
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_mode_keyboard("ru").inline_keyboard


def test_prompt_master_callback_back_returns_menu() -> None:
    edits = []

    async def fake_edit(text, **kwargs):
        edits.append((text, kwargs))

    async def fake_answer(*args, **kwargs):
        pass

    query = SimpleNamespace(
        data=f"{CB_PM_PREFIX}back",
        answer=fake_answer,
        edit_message_text=fake_edit,
        message=SimpleNamespace(chat=SimpleNamespace(id=1)),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(language_code="ru"),
    )
    ctx = SimpleNamespace(user_data={"mode": "pm", "pm_engine": "veo"})

    asyncio.run(prompt_master_callback(update, ctx))

    assert not ctx.user_data.get("pm_engine")
    assert edits
    _text, kwargs = edits[0]
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_keyboard("ru").inline_keyboard
    assert kwargs["parse_mode"] == "HTML"


@pytest.mark.parametrize(
    "engine,text_value",
    [
        (Engine.VEO_ANIMATE, "мягко оживить портрет"),
        (Engine.BANANA_EDIT, "убрать лишний фон"),
        (Engine.SUNO, "энергичный трек о путешествии"),
    ],
)
def test_prompt_master_handle_text_flow(engine: Engine, text_value: str) -> None:
    message = DummyMessage(text_value)
    update = SimpleNamespace(
        message=message,
        effective_chat=message.chat,
        effective_user=SimpleNamespace(id=7, language_code="ru"),
    )
    ctx = SimpleNamespace(
        user_data={PM_STATE_KEY: "pm", PM_ENGINE_KEY: engine.value},
    )

    asyncio.run(prompt_master_handle_text(update, ctx))

    assert message.reply_calls
    status_text, status_kwargs = message.reply_calls[0]
    assert status_text.startswith("⏳")
    assert status_kwargs["reply_markup"].inline_keyboard == prompt_master_mode_keyboard("ru").inline_keyboard

    edits = message.status.edits
    assert edits, "status message must be edited with final prompt"
    final_text, final_kwargs = edits[-1]
    assert "Готовый промпт" in final_text or "Ready prompt" in final_text
    markup_rows = final_kwargs["reply_markup"].inline_keyboard
    assert markup_rows[-1][0].callback_data == f"pm:copy:{engine.value}"
    assert markup_rows[-1][1].callback_data == f"pm:insert:{engine.value}"


def test_build_prompt_banana_contains_safety_phrase() -> None:
    prompt = asyncio.run(build_prompt(Engine.BANANA_EDIT, "remove blemishes", "en"))
    assert "face" in prompt.body_markdown.lower()
    assert prompt.insert_payload["engine"] == Engine.BANANA_EDIT.value


def test_build_prompt_animate_ru_mentions_safety() -> None:
    prompt = asyncio.run(build_prompt(Engine.VEO_ANIMATE, "мягко оживить портрет", "ru"))
    assert "Сохраняем черты лица" in prompt.body_markdown
    assert "Запрещено" in prompt.body_markdown


def test_build_prompt_veo_json_structure() -> None:
    prompt = asyncio.run(build_prompt(Engine.VEO_VIDEO, "cinematic sunrise", "en"))
    payload = json.loads(prompt.copy_text)
    assert {"scene", "camera", "motion", "lighting", "palette", "details"} <= set(payload.keys())


def test_build_prompt_mj_json_structure() -> None:
    prompt = asyncio.run(build_prompt(Engine.MJ, "futuristic portrait", "en"))
    payload = json.loads(prompt.copy_text)
    assert {"prompt", "camera", "lighting", "palette", "render"} <= set(payload.keys())


def test_detect_language_simple() -> None:
    assert detect_language("Привет") == "ru"
    assert detect_language("Hello") == "en"
