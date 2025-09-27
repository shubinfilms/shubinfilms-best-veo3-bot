"""Tests for Prompt-Master MVP handlers."""

import asyncio
from types import SimpleNamespace

from handlers.prompt_master_handler import PM_HINT, prompt_master_callback, prompt_master_open
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
