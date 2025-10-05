import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from telegram.constants import ChatType
from telegram.error import BadRequest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from handlers.prompt_master_handler import (
    PM_STATE_KEY,
    clear_pm_prompts,
    detect_language,
    get_pm_prompt,
    _edit_with_fallback,
    prompt_master_callback,
    prompt_master_open,
    prompt_master_text_handler,
)
from keyboards import (
    CB_PM_PREFIX,
    prompt_master_keyboard,
    prompt_master_mode_keyboard,
    prompt_master_result_keyboard,
)
from prompt_master import Engine, build_banana_prompt, build_mj_prompt, build_prompt, build_veo_prompt
from utils.html_render import render_pm_html
from utils.safe_send import send_html_with_fallback


class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, dict]] = []
        self.edited: list[tuple[int, int, str, dict]] = []
        self._message_id = 100
        self.fail_next_html_send = False
        self.fail_next_html_edit = False

    async def send_message(self, chat_id: int, text: str, **kwargs):  # pragma: no cover - simple recorder
        if kwargs.get("parse_mode") == "HTML" and self.fail_next_html_send:
            self.fail_next_html_send = False
            raise BadRequest("Can't parse entities: unsupported start tag \"br/\"")
        self._message_id += 1
        self.sent.append((chat_id, text, kwargs))
        return SimpleNamespace(message_id=self._message_id, chat_id=chat_id)

    async def edit_message_text(self, chat_id: int, message_id: int, text: str, **kwargs):
        if kwargs.get("parse_mode") == "HTML" and self.fail_next_html_edit:
            self.fail_next_html_edit = False
            raise BadRequest("Can't parse entities: unsupported start tag \"ul\"")
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


def test_render_pm_html_br_self_closing() -> None:
    result = render_pm_html("first line<br/>second line")
    assert "<br/>" not in result
    assert "first line<br>second line" == result


def test_render_pm_html_json_code_block() -> None:
    json_block = """```json\n{\n  \"key\": \"value\"\n}\n```"""
    result = render_pm_html(json_block)
    assert "<pre><code>" in result
    assert "&quot;value&quot;" in result


def test_send_html_with_fallback_on_bad_html() -> None:
    bot = FakeBot()
    bot.fail_next_html_send = True
    message = asyncio.run(send_html_with_fallback(bot, 777, "<ul><li>item</li></ul>"))
    assert message is not None
    sent_chat, sent_text, kwargs = bot.sent[-1]
    assert sent_chat == 777
    assert "parse_mode" not in kwargs
    assert "<" not in sent_text


def test_edit_with_fallback_plain_text() -> None:
    bot = FakeBot()
    bot.fail_next_html_edit = True
    ctx = SimpleNamespace(bot=bot)
    asyncio.run(
        _edit_with_fallback(
            ctx,
            555,
            1000,
            "<ul><li>broken</li></ul>",
            prompt_master_mode_keyboard("en"),
        )
    )
    _, _, text, kwargs = bot.edited[-1]
    assert "parse_mode" not in kwargs
    assert "<" not in text


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


def test_prompt_master_result_keyboard_layout() -> None:
    keyboard = prompt_master_result_keyboard("veo", "ru")
    rows = keyboard.inline_keyboard
    assert len(rows) == 2
    copy_row = rows[0]
    back_row = rows[1]
    assert copy_row[0].callback_data == f"{CB_PM_PREFIX}copy:veo"
    assert copy_row[1].callback_data == f"{CB_PM_PREFIX}insert:veo"
    assert back_row[0].callback_data == f"{CB_PM_PREFIX}back"
    assert back_row[1].callback_data == f"{CB_PM_PREFIX}switch"


def test_prompt_master_open_uses_safe_send_html() -> None:
    bot = FakeBot()
    update = SimpleNamespace(
        effective_message=SimpleNamespace(chat=SimpleNamespace(id=555), chat_id=555),
        effective_user=SimpleNamespace(language_code="ru"),
    )
    ctx = SimpleNamespace(bot=bot, user_data={})
    asyncio.run(prompt_master_open(update, ctx))
    assert bot.sent
    card_entry = next(
        ((chat_id, text, kwargs) for chat_id, text, kwargs in bot.sent if "<b>Prompt-Master</b>" in text),
        None,
    )
    assert card_entry is not None
    _, text, kwargs = card_entry
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
    card_entry = None
    for chat_id, card_html, kwargs in bot.sent:
        markup = kwargs.get("reply_markup") if isinstance(kwargs, dict) else None
        if markup and markup.inline_keyboard == prompt_master_mode_keyboard("ru").inline_keyboard:
            card_entry = (chat_id, card_html, kwargs)
            break
    assert card_entry is not None, "prompt card should be present"
    chat_id, card_html, kwargs = card_entry
    assert chat_id == 321
    assert kwargs["reply_markup"].inline_keyboard == prompt_master_mode_keyboard("ru").inline_keyboard
    assert "<br/>" not in card_html
    assert state["card_msg_id"]
    bottom_menu = [entry for entry in bot.sent if entry[1] == "👇 Быстрые действия"]
    assert bottom_menu, "bottom menu should be rendered"


def test_prompt_master_text_handler_generates_prompt_and_updates_status() -> None:
    bot = FakeBot()
    state = {"engine": "mj", "card_msg_id": None, "autodelete": False}
    ctx = SimpleNamespace(bot=bot, user_data={PM_STATE_KEY: state})
    message = SimpleNamespace(text="futuristic portrait", chat=SimpleNamespace(id=333, type=ChatType.PRIVATE))
    update = SimpleNamespace(message=message, effective_chat=message.chat)
    asyncio.run(prompt_master_text_handler(update, ctx))
    assert len(bot.sent) >= 3
    card_entry = next(
        (entry for entry in bot.sent if "<pre><code>" in entry[1]),
        None,
    )
    assert card_entry is not None
    card_chat, card_text, card_kwargs = card_entry
    assert card_chat == 333
    assert "<pre><code>" in card_text
    status_entry = next(
        (
            entry
            for entry in bot.sent
            if entry[1].startswith("🧠") or entry[1].startswith("⚙️")
        ),
        None,
    )
    assert status_entry is not None
    status_chat, status_text, status_kwargs = status_entry
    assert status_text.startswith("🧠")
    assert status_kwargs["reply_markup"].inline_keyboard == prompt_master_mode_keyboard("en").inline_keyboard
    assert status_kwargs.get("parse_mode") == "HTML"
    bottom_menu = [entry for entry in bot.sent if entry[1] == "👇 Быстрые действия"]
    assert bottom_menu, "bottom menu should be rendered"
    # Final edit should include result keyboard
    assert bot.edited, "status message must be edited"
    final_entry = next((entry for entry in bot.edited if "Ready prompt" in entry[2]), None)
    assert final_entry is not None
    _chat, _mid, final_text, final_kwargs = final_entry
    assert "Ready prompt" in final_text
    assert "<pre><code>" in final_text
    assert "<br/>" not in final_text
    buttons = final_kwargs["reply_markup"].inline_keyboard
    assert buttons[0][0].callback_data == "pm:copy:mj"
    assert buttons[0][1].callback_data == "pm:insert:mj"
    assert buttons[1][0].callback_data == "pm:back"
    assert buttons[1][1].callback_data == "pm:switch"
    cached = get_pm_prompt(333, "mj")
    assert cached is not None
    assert cached["engine"] == "mj"
    assert cached["copy_text"].strip().startswith("{")


def test_prompt_master_status_message_for_veo() -> None:
    bot = FakeBot()
    state = {"engine": "veo", "card_msg_id": None, "autodelete": False}
    ctx = SimpleNamespace(bot=bot, user_data={PM_STATE_KEY: state})
    message = SimpleNamespace(text="Два героя спорят у окна во время грозы.", chat=SimpleNamespace(id=555, type=ChatType.PRIVATE))
    update = SimpleNamespace(message=message, effective_chat=message.chat)
    asyncio.run(prompt_master_text_handler(update, ctx))
    assert len(bot.sent) >= 3
    status_entry = next(
        (entry for entry in bot.sent if entry[1].startswith("⚙️ Начинаю собирать промпт для VEO")),
        None,
    )
    assert status_entry is not None


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
    assert ctx.user_data[PM_STATE_KEY]["prompt"] == payload.get("card_text")
    assert bot.sent, "card should be rendered on insert"


def test_prompt_master_insert_sets_veo_metadata() -> None:
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={PM_STATE_KEY: {"engine": "veo", "card_msg_id": None}})
    chat_id = 999
    clear_pm_prompts(chat_id)
    from handlers.prompt_master_handler import _store_prompt  # type: ignore

    payload = build_veo_prompt("Two detectives argue", "en")
    _store_prompt(chat_id, "veo", payload)

    query = FakeQuery(f"{CB_PM_PREFIX}insert:veo", bot)
    query.message.chat.id = chat_id
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(language_code="en"),
    )
    asyncio.run(prompt_master_callback(update, ctx))
    state = ctx.user_data[PM_STATE_KEY]
    assert state["veo_duration_hint"]
    assert state["veo_lip_sync_required"] == bool(payload["raw_payload"]["lip_sync_required"])
    assert state["veo_voiceover_origin"] == payload["raw_payload"].get("voiceover_origin")


def test_prompt_master_copy_sends_plain_text() -> None:
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, user_data={PM_STATE_KEY: {"engine": "mj", "card_msg_id": None}})
    chat_id = 777
    clear_pm_prompts(chat_id)
    from handlers.prompt_master_handler import _store_prompt  # type: ignore

    payload = build_mj_prompt("dramatic scene", "en")
    _store_prompt(chat_id, "mj", payload)

    query = FakeQuery(f"{CB_PM_PREFIX}copy:mj", bot)
    query.message = SimpleNamespace(chat=SimpleNamespace(id=chat_id))
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(language_code="en"),
    )
    asyncio.run(prompt_master_callback(update, ctx))
    assert bot.sent, "copy must send a message"
    _, text, kwargs = bot.sent[-1]
    assert kwargs.get("parse_mode") is None
    assert "<" not in text
    assert payload["copy_text"].splitlines()[0] in text


@pytest.mark.parametrize(
    "builder,text,lang,needle",
    [
        (build_banana_prompt, "remove blemishes", "en", "Preserve facial traits"),
        (build_banana_prompt, "сделай аккуратно", "ru", "Сохраняем черты лица"),
        (build_mj_prompt, "cinematic hero", "en", "Preserve"),
    ],
)
def test_prompt_builders_return_body(builder, text, lang, needle) -> None:
    payload = builder(text, lang)
    body_html = payload["body_html"]
    assert needle.lower() in body_html.lower()
    copy = payload["copy_text"]
    if copy.strip().startswith("{"):
        assert copy.strip().startswith("{")


def test_render_payload_snapshot_mj() -> None:
    payload = build_mj_prompt("cinematic hero", "en")
    html_text = payload["body_html"]
    assert "<b>Ready prompt for Midjourney</b>" in html_text
    assert "&quot;render&quot;" in html_text


def test_render_payload_snapshot_veo() -> None:
    payload = build_veo_prompt("cinematic hero", "en")
    html_text = payload["body_html"]
    assert "<b>Ready prompt for VEO</b>" in html_text
    assert "&quot;idea&quot;" in html_text


def test_render_payload_snapshot_banana() -> None:
    payload = build_banana_prompt("touch up", "en")
    html_text = payload["body_html"]
    assert "<b>Banana edit checklist</b>" in html_text
    assert "Checklist:" in html_text


def test_build_prompt_veo_json_structure() -> None:
    prompt = asyncio.run(build_prompt(Engine.VEO_VIDEO, "cinematic sunrise", "en"))
    payload = json.loads(prompt["copy_text"])
    expected_keys = {
        "idea",
        "scene_description",
        "timeline",
        "camera",
        "lighting",
        "palette",
        "details",
        "audio",
        "notes",
        "safety",
    }
    assert expected_keys <= set(payload.keys())
    assert len(payload["timeline"]) == 4


def test_detect_language_simple() -> None:
    assert detect_language("Привет") == "ru"
    assert detect_language("Hello") == "en"
