import asyncio
from types import SimpleNamespace

from telegram.constants import ParseMode

from handlers.faq_handler import configure_faq, faq_callback, faq_command
from keyboards import CB_FAQ_PREFIX, faq_keyboard
from texts import FAQ_INTRO, FAQ_SECTIONS


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, dict]] = []
        self.edits: list[tuple[int, int, str, dict]] = []

    async def send_message(self, chat_id: int, text: str, **kwargs):
        self.sent.append((chat_id, text, kwargs))
        return SimpleNamespace(message_id=kwargs.get("message_id", 1), chat_id=chat_id)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str, **kwargs):
        self.edits.append((chat_id, message_id, text, kwargs))
        return SimpleNamespace(message_id=message_id, chat_id=chat_id)


def setup_function():
    configure_faq(show_main_menu=None, on_root_view=None, on_section_view=None)


def test_faq_command_sends_intro_and_keyboard():
    bot = FakeBot()
    chat = SimpleNamespace(id=101)
    message = SimpleNamespace(chat=chat, chat_id=101)
    update = SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=1),
        effective_chat=chat,
    )
    ctx = SimpleNamespace(bot=bot)

    root_calls = []
    configure_faq(on_root_view=lambda: root_calls.append(True))

    asyncio.run(faq_command(update, ctx))

    assert bot.sent
    chat_id, text, kwargs = bot.sent[0]
    assert chat_id == 101
    assert "<b>FAQ</b>" in text
    assert kwargs["reply_markup"].inline_keyboard == faq_keyboard().inline_keyboard
    assert kwargs["parse_mode"] in (ParseMode.HTML, "HTML")
    assert root_calls == [True]


def test_faq_callback_sends_section_text():
    bot = FakeBot()

    async def fake_answer():
        pass

    section_calls = []
    configure_faq(on_section_view=lambda key: section_calls.append(key))

    chat = SimpleNamespace(id=202)
    query = SimpleNamespace(
        data=f"{CB_FAQ_PREFIX}veo",
        answer=fake_answer,
        message=SimpleNamespace(chat=chat, message_id=55),
        bot=bot,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=2),
        effective_chat=chat,
    )
    ctx = SimpleNamespace(bot=bot)

    asyncio.run(faq_callback(update, ctx))

    assert bot.edits
    chat_id, msg_id, text, kwargs = bot.edits[0]
    assert chat_id == 202 and msg_id == 55
    assert "<b>Видео" in text
    assert kwargs["reply_markup"].inline_keyboard == faq_keyboard().inline_keyboard
    assert kwargs["parse_mode"] in (ParseMode.HTML, "HTML")
    assert section_calls == ["veo"]


def test_faq_callback_back_calls_main_menu():
    bot = FakeBot()

    async def fake_menu(update, ctx):
        await bot.send_message(303, "menu called")

    configure_faq(show_main_menu=fake_menu)

    async def fake_answer():
        pass

    chat = SimpleNamespace(id=303)
    query = SimpleNamespace(
        data=f"{CB_FAQ_PREFIX}back",
        answer=fake_answer,
        edit_message_text=None,
        message=SimpleNamespace(chat=chat, message_id=10),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=3),
        effective_chat=chat,
    )
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

    bot = FakeBot()

    async def fake_answer():
        pass

    chat = SimpleNamespace(id=404)
    query = SimpleNamespace(
        data=f"{CB_FAQ_PREFIX}unknown",
        answer=fake_answer,
        message=SimpleNamespace(chat=chat, message_id=77),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=4),
        effective_chat=chat,
    )
    ctx = SimpleNamespace(bot=bot)

    asyncio.run(faq_callback(update, ctx))

    assert bot.edits
    _, _, text, _ = bot.edits[0]
    assert text == "Раздел не найден."
