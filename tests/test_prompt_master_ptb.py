import asyncio
import json
from types import SimpleNamespace

from handlers.prompt_master_handler import (
    build_prompt,
    detect_language,
    prompt_master_callback,
    prompt_master_open,
)
from keyboards import (
    CB_PM_PREFIX,
    prompt_master_keyboard,
    prompt_master_mode_keyboard,
)


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
    update = SimpleNamespace(effective_message=message, message=message, callback_query=None, effective_user=user)
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
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=42, language_code="ru"), effective_chat=SimpleNamespace(id=1))
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
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(language_code="ru"))
    ctx = SimpleNamespace(user_data={"mode": "pm", "pm_engine": "veo"})

    asyncio.run(prompt_master_callback(update, ctx))

    assert not ctx.user_data.get("pm_engine")
    assert edits
    _text, kwargs = edits[0]
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_keyboard("ru").inline_keyboard
    assert kwargs["parse_mode"] == "HTML"


def test_build_prompt_banana_contains_safety_phrase() -> None:
    prompt = build_prompt("banana", "remove blemishes", "en", {})
    assert not prompt.is_json
    assert "real face" in prompt.body.lower()


def test_build_prompt_animate_ru_mentions_safety() -> None:
    prompt = build_prompt("animate", "мягко оживить портрет", "ru", {})
    assert "Не менять внешность" in prompt.body


def test_build_prompt_veo_json_structure() -> None:
    prompt = build_prompt("veo", "cinematic sunrise", "en", {})
    assert prompt.is_json
    payload = json.loads(prompt.body)
    assert {"scene", "camera", "motion", "lighting", "palette", "details"} <= set(payload.keys())


def test_detect_language_simple() -> None:
    assert detect_language("Привет") == "ru"
    assert detect_language("Hello") == "en"
