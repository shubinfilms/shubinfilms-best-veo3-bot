"""FAQ handler tests for new inline menu."""

import asyncio
from types import SimpleNamespace

from handlers.faq_handler import configure_faq, faq_callback, faq_command
from keyboards import CB_FAQ_PREFIX, faq_keyboard
from texts import FAQ_INTRO, FAQ_SECTIONS


def setup_function():
    configure_faq(show_main_menu=None, on_root_view=None, on_section_view=None)


def test_faq_command_sends_intro_and_keyboard():
    calls = []

    async def fake_reply(text, **kwargs):
        calls.append((text, kwargs))

    message = SimpleNamespace(reply_text=fake_reply)
    update = SimpleNamespace(effective_message=message)
    ctx = SimpleNamespace()

    root_calls = []
    configure_faq(on_root_view=lambda: root_calls.append(True))

    asyncio.run(faq_command(update, ctx))

    assert calls
    text, kwargs = calls[0]
    assert text == FAQ_INTRO
    assert kwargs["reply_markup"].inline_keyboard == faq_keyboard().inline_keyboard
    assert kwargs["parse_mode"] == "Markdown"
    assert kwargs["disable_web_page_preview"] is True
    assert root_calls == [True]


def test_faq_callback_sends_section_text():
    edits = []

    async def fake_edit(text, **kwargs):
        edits.append((text, kwargs))

    answers = []

    async def fake_answer():
        answers.append(True)

    section_calls = []
    configure_faq(on_section_view=lambda key: section_calls.append(key))

    query = SimpleNamespace(
        data=f"{CB_FAQ_PREFIX}veo",
        edit_message_text=fake_edit,
        answer=fake_answer,
        message=SimpleNamespace(chat_id=1),
    )
    update = SimpleNamespace(callback_query=query, effective_user=None)
    ctx = SimpleNamespace(application=SimpleNamespace(bot=None))

    asyncio.run(faq_callback(update, ctx))

    assert answers == [True]
    assert edits
    text, kwargs = edits[0]
    assert text == FAQ_SECTIONS["veo"]
    assert kwargs["reply_markup"].inline_keyboard == faq_keyboard().inline_keyboard
    assert kwargs["parse_mode"] == "Markdown"
    assert kwargs["disable_web_page_preview"] is True
    assert section_calls == ["veo"]


def test_faq_callback_back_calls_main_menu():
    menu_calls = []

    async def fake_menu(update, ctx):
        menu_calls.append((update, ctx))

    configure_faq(show_main_menu=fake_menu)

    async def fake_answer():
        pass

    query = SimpleNamespace(
        data=f"{CB_FAQ_PREFIX}back",
        answer=fake_answer,
        edit_message_text=None,
        message=SimpleNamespace(chat_id=1),
    )
    update = SimpleNamespace(callback_query=query, effective_user=None)
    ctx = SimpleNamespace(application=SimpleNamespace(bot=None))

    asyncio.run(faq_callback(update, ctx))

    assert len(menu_calls) == 1
    assert menu_calls[0][0] is update
    assert menu_calls[0][1] is ctx


def test_faq_callback_unknown_section_returns_fallback():
    edits = []

    async def fake_edit(text, **kwargs):
        edits.append((text, kwargs))

    async def fake_answer():
        pass

    query = SimpleNamespace(
        data=f"{CB_FAQ_PREFIX}unknown",
        edit_message_text=fake_edit,
        answer=fake_answer,
        message=SimpleNamespace(chat_id=1),
    )
    update = SimpleNamespace(callback_query=query, effective_user=None)
    ctx = SimpleNamespace(application=SimpleNamespace(bot=None))

    asyncio.run(faq_callback(update, ctx))

    assert edits
    text, _ = edits[0]
    assert text == "Раздел не найден."
