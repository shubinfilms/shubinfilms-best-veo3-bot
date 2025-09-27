import asyncio
import json
from types import SimpleNamespace

import pytest

from telegram.constants import ChatType

from handlers.prompt_master_handler import (
    PM_STATE_KEY,
    clear_pm_prompts,
    detect_language,
    get_pm_prompt,
    prompt_master_callback,
    prompt_master_open,
    prompt_master_text_handler,
)
from keyboards import CB_PM_PREFIX, prompt_master_keyboard, prompt_master_mode_keyboard
from prompt_master import Engine, build_banana_prompt, build_mj_prompt, build_prompt


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, dict]] = []
        self.edited: list[tuple[int, int, str, dict]] = []
        self._message_id = 100

    async def send_message(self, chat_id: int, text: str, **kwargs):  # pragma: no cover - simple recorder
        self._message_id += 1
        self.sent.append((chat_id, text, kwargs))
        return SimpleNamespace(message_id=self._message_id, chat_id=chat_id)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str, **kwargs):
        self.edited.append((chat_id, message_id, text, kwargs))
        return SimpleNamespace()


class FakeQuery:
    def __init__(self, data: str, bot: FakeBot):
        self.data = data
        self.bot = bot
        self.message = SimpleNamespace(chat=SimpleNamespace(id=321))
        self._answers: list[tuple[str, bool]] = []

    async def answer(self, text: str = "", show_alert: bool = False):  # pragma: no cover - simple recorder
        self._answers.append((text, show_alert))

    async def edit_message_text(self, text: str, **kwargs):
        await self.bot.edit_message_text(self.message.chat.id, 999, text, **kwargs)


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


def test_prompt_master_open_uses_safe_send_html() -> None:
    bot = FakeBot()
    update = SimpleNamespace(
        effective_message=SimpleNamespace(chat=SimpleNamespace(id=555), chat_id=555),
        effective_user=SimpleNamespace(language_code="ru"),
    )
    ctx = SimpleNamespace(bot=bot, user_data={})
    asyncio.run(prompt_master_open(update, ctx))
    assert bot.sent
    _, text, kwargs = bot.sent[-1]
    assert "<b>Prompt-Master</b>" in text
    assert kwargs["parse_mode"] == "HTML"
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_keyboard("ru").inline_keyboard


def test_prompt_master_callback_selects_engine_and_creates_card() -> None:
    bot = FakeBot()
    query = FakeQuery(f"{CB_PM_PREFIX}veo", bot)
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=77, language_code="ru"),
        effective_chat=SimpleNamespace(id=321),
    )
    ctx = SimpleNamespace(bot=bot, user_data={})
    asyncio.run(prompt_master_callback(update, ctx))
    state = ctx.user_data.get(PM_STATE_KEY)
    assert state["engine"] == "veo"
    assert bot.sent, "card must be rendered"
    chat_id, _text, kwargs = bot.sent[-1]
    assert chat_id == 321
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_mode_keyboard("ru").inline_keyboard
    assert state["card_msg_id"]


def test_prompt_master_text_handler_generates_prompt_and_updates_status() -> None:
    bot = FakeBot()
    state = {"engine": "mj", "card_msg_id": None, "autodelete": False}
    ctx = SimpleNamespace(bot=bot, user_data={PM_STATE_KEY: state})
    message = SimpleNamespace(text="futuristic portrait", chat=SimpleNamespace(id=333, type=ChatType.PRIVATE))
    update = SimpleNamespace(message=message, effective_chat=message.chat)
    asyncio.run(prompt_master_text_handler(update, ctx))
    # First send is card, second send is status
    assert len(bot.sent) >= 2
    card_chat, card_text, card_kwargs = bot.sent[0]
    assert card_chat == 333
    assert "<pre><code>" in card_text
    status_chat, status_text, status_kwargs = bot.sent[1]
    assert status_text.startswith("✍️")
    assert status_kwargs["reply_markup"].inline_keyboard == prompt_master_mode_keyboard("en").inline_keyboard
    # Final edit should include result keyboard
    assert bot.edited, "status message must be edited"
    _chat, _mid, final_text, final_kwargs = bot.edited[-1]
    assert "Ready prompt" in final_text
    buttons = final_kwargs["reply_markup"].inline_keyboard[-1]
    assert buttons[0].callback_data == "pm:copy:mj"
    assert get_pm_prompt(333, "mj") is not None


def test_prompt_master_insert_uses_cached_payload() -> None:
    bot = FakeBot()
    payload = build_banana_prompt("почисти фон", "ru")
    ctx = SimpleNamespace(bot=bot, user_data={PM_STATE_KEY: {"engine": "banana", "card_msg_id": None}})
    chat_id = 444
    clear_pm_prompts(chat_id)
    # store payload to simulate previous run
    from handlers.prompt_master_handler import _store_prompt  # type: ignore

    _store_prompt(chat_id, "banana", payload)
    query = FakeQuery(f"pm:insert:banana", bot)
    query.message.chat.id = chat_id
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(language_code="ru"),
    )
    asyncio.run(prompt_master_callback(update, ctx))
    assert ctx.user_data[PM_STATE_KEY]["prompt"] == payload.card_text
    assert bot.sent, "card should be rendered on insert"


@pytest.mark.parametrize(
    "builder,text,lang,needle",
    [
        (build_banana_prompt, "remove blemishes", "en", "face"),
        (build_banana_prompt, "сделай аккуратно", "ru", "Сохраняем черты лица"),
        (build_mj_prompt, "cinematic hero", "en", "prompt"),
    ],
)
def test_prompt_builders_return_html(builder, text, lang, needle) -> None:
    payload = builder(text, lang)
    assert needle.lower() in payload.body_html.lower()


def test_build_prompt_veo_json_structure() -> None:
    prompt = asyncio.run(build_prompt(Engine.VEO_VIDEO, "cinematic sunrise", "en"))
    payload = json.loads(prompt.copy_text)
    assert {"scene", "camera", "motion", "lighting", "palette", "details"} <= set(payload.keys())


def test_detect_language_simple() -> None:
    assert detect_language("Привет") == "ru"
    assert detect_language("Hello") == "en"
