import asyncio
from types import SimpleNamespace

from handlers.faq_handler import configure_faq, faq_callback, faq_command
from keyboards import CB_FAQ_PREFIX, faq_keyboard
from texts import FAQ_INTRO, FAQ_SECTIONS


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, dict]] = []

    async def send_message(self, chat_id: int, text: str, **kwargs):
        self.sent.append((chat_id, text, kwargs))
        return SimpleNamespace(message_id=1, chat_id=chat_id)


def setup_function():
    configure_faq(show_main_menu=None, on_root_view=None, on_section_view=None)


def test_faq_command_sends_intro_and_keyboard():
    bot = FakeBot()
    message = SimpleNamespace(chat=SimpleNamespace(id=101), chat_id=101)
    update = SimpleNamespace(effective_message=message)
    ctx = SimpleNamespace(bot=bot)

    root_calls = []
    configure_faq(on_root_view=lambda: root_calls.append(True))

    asyncio.run(faq_command(update, ctx))

    assert bot.sent
    chat_id, text, kwargs = bot.sent[0]
    assert chat_id == 101
    assert "<b>FAQ</b>" in text
    assert kwargs["reply_markup"].inline_keyboard == faq_keyboard().inline_keyboard
    assert kwargs["parse_mode"] == "HTML"
    assert root_calls == [True]


def test_faq_callback_sends_section_text():
    edits = []

    async def fake_edit(text, **kwargs):
        edits.append((text, kwargs))

    async def fake_answer():
        pass

    section_calls = []
    configure_faq(on_section_view=lambda key: section_calls.append(key))

    query = SimpleNamespace(
        data=f"{CB_FAQ_PREFIX}veo",
        edit_message_text=fake_edit,
        answer=fake_answer,
        message=SimpleNamespace(chat=SimpleNamespace(id=202)),
    )
    update = SimpleNamespace(callback_query=query, effective_user=None)
    ctx = SimpleNamespace()

    asyncio.run(faq_callback(update, ctx))

    assert edits
    text, kwargs = edits[0]
    assert "<b>Видео" in text
    assert kwargs["reply_markup"].inline_keyboard == faq_keyboard().inline_keyboard
    assert kwargs["parse_mode"] == "HTML"
    assert section_calls == ["veo"]


def test_faq_callback_back_calls_main_menu():
    bot = FakeBot()

    async def fake_menu(update, ctx):
        await bot.send_message(303, "menu called")

    configure_faq(show_main_menu=fake_menu)

    async def fake_answer():
        pass

    query = SimpleNamespace(
        data=f"{CB_FAQ_PREFIX}back",
        answer=fake_answer,
        edit_message_text=None,
        message=SimpleNamespace(chat=SimpleNamespace(id=303)),
    )
    update = SimpleNamespace(callback_query=query, effective_user=None)
    ctx = SimpleNamespace(bot=bot)

    asyncio.run(faq_callback(update, ctx))

    assert bot.sent
    assert bot.sent[0][0] == 303


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
        message=SimpleNamespace(chat=SimpleNamespace(id=404)),
    )
    update = SimpleNamespace(callback_query=query, effective_user=None)
    ctx = SimpleNamespace()

    asyncio.run(faq_callback(update, ctx))

    assert edits
    text, _ = edits[0]
    assert text == "Раздел не найден."
