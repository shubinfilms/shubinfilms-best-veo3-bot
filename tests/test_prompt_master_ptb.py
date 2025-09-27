"""Tests for Prompt-Master handlers."""

import asyncio
import json
from types import SimpleNamespace

from handlers.prompt_master_handler import (
    PM_HINT,
    build_prompt_result,
    classify_prompt_engine,
    prompt_master_callback,
    prompt_master_open,
)
from keyboards import CB_PM_PREFIX, prompt_master_keyboard


def test_prompt_master_keyboard_layout() -> None:
    keyboard = prompt_master_keyboard()
    layout = [
        (button.text, button.callback_data)
        for row in keyboard.inline_keyboard
        for button in row
    ]
    expected = [
        ("🎬 Видеопромпт (VEO)", f"{CB_PM_PREFIX}video"),
        ("🖼️ Промпт генерации фото (MJ)", f"{CB_PM_PREFIX}mj_gen"),
        ("🫥 Оживление фото (VEO)", f"{CB_PM_PREFIX}photo_live"),
        ("✂️ Редактирование фото (Banana)", f"{CB_PM_PREFIX}banana_edit"),
        ("🎵 Текст песни (Suno)", f"{CB_PM_PREFIX}suno_lyrics"),
        ("↩️ Назад", f"{CB_PM_PREFIX}back"),
    ]
    assert layout == expected


def test_prompt_master_open_replies_with_keyboard() -> None:
    calls = []

    async def fake_reply(text, **kwargs):
        calls.append((text, kwargs))

    message = SimpleNamespace(reply_text=fake_reply)
    update = SimpleNamespace(effective_message=message, message=message, callback_query=None)
    ctx = SimpleNamespace()

    asyncio.run(prompt_master_open(update, ctx))

    assert calls
    text, kwargs = calls[0]
    assert text == PM_HINT
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_keyboard().inline_keyboard
    assert kwargs["parse_mode"] == "Markdown"


def test_prompt_master_callback_returns_placeholder() -> None:
    edits = []

    async def fake_edit(text, **kwargs):
        edits.append((text, kwargs))

    answers = []

    async def fake_answer():
        answers.append(True)

    query = SimpleNamespace(
        data=f"{CB_PM_PREFIX}video",
        answer=fake_answer,
        edit_message_text=fake_edit,
        message=SimpleNamespace(),
    )
    update = SimpleNamespace(callback_query=query, effective_user=None)
    ctx = SimpleNamespace()

    asyncio.run(prompt_master_callback(update, ctx))

    assert answers == [True]
    assert edits
    text, kwargs = edits[0]
    assert "Функция скоро будет доступна" in text
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_keyboard().inline_keyboard


def test_prompt_master_callback_back_returns_menu() -> None:
    edits = []

    async def fake_edit(text, **kwargs):
        edits.append((text, kwargs))

    answers = []

    async def fake_answer():
        answers.append(True)

    query = SimpleNamespace(
        data=f"{CB_PM_PREFIX}back",
        answer=fake_answer,
        edit_message_text=fake_edit,
        message=SimpleNamespace(),
    )
    update = SimpleNamespace(callback_query=query, effective_message=None, effective_user=None)
    ctx = SimpleNamespace()

    asyncio.run(prompt_master_callback(update, ctx))

    assert answers == [True]
    assert edits
    text, kwargs = edits[0]
    assert text == PM_HINT
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_keyboard().inline_keyboard


def test_classify_prompt_engine_variants() -> None:
    assert classify_prompt_engine("midjourney cinematic scene") == "mj"
    assert classify_prompt_engine("оживи фото дедушки") == "photo_live"
    assert classify_prompt_engine("banana edit portrait") == "banana"
    assert classify_prompt_engine("Suno make a song about hope") == "suno"
    assert classify_prompt_engine("снимем клип про рассвет") == "veo"


def test_build_prompt_result_veo_contains_duration() -> None:
    result, lang = build_prompt_result("Снимем видео VEO про рассвет")
    assert result.engine == "veo"
    assert lang == "ru"
    payload = json.loads(result.raw)
    assert "Длительность" in payload["action"]


def test_build_prompt_result_mj_json_structure() -> None:
    result, lang = build_prompt_result("Midjourney concept art of neon city")
    assert result.engine == "mj"
    assert lang == "en"
    payload = json.loads(result.raw)
    assert set(payload.keys()) == {"prompt", "style", "camera", "lighting"}
    assert "four" in payload["prompt"].lower()


def test_build_prompt_result_banana_has_safety_phrase() -> None:
    result, _ = build_prompt_result("Banana edit portrait with soft light")
    assert result.engine == "banana"
    assert not result.is_json
    assert "keep the real face unchanged" in result.raw
