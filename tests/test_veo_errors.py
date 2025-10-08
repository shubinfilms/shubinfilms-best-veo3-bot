from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from handlers import veo_fast as veo_fast_module


class DummyBot:
    def __init__(self) -> None:
        self.sent_messages: list[SimpleNamespace] = []

    async def send_message(
        self, *, chat_id: int, text: str, reply_markup=None
    ) -> SimpleNamespace:  # type: ignore[override]
        self.sent_messages.append(
            SimpleNamespace(chat_id=chat_id, text=text, reply_markup=reply_markup)
        )
        return SimpleNamespace(message_id=len(self.sent_messages))


def _make_context(bot: DummyBot) -> SimpleNamespace:
    return SimpleNamespace(bot=bot, chat_data={}, user_data={"state": {}})


def _get_retry_id(message: SimpleNamespace) -> str:
    markup = message.reply_markup
    assert markup is not None
    button = markup.inline_keyboard[0][0]
    assert button.text == "🔁 Повторить"
    return button.callback_data


def test_policy_error_shows_friendly_text_and_retry(caplog: pytest.LogCaptureFixture) -> None:
    bot = DummyBot()
    ctx = _make_context(bot)
    called: list[str] = []

    async def retry_handler() -> None:
        called.append("retry")

    error = veo_fast_module.VEOFastHTTPError(
        429,
        payload={"reason": "policy_violation"},
        req_id="req-1",
    )

    with caplog.at_level(logging.INFO, logger="user-errors"):
        asyncio.run(
            veo_fast_module.handle_veo_fast_error(
                ctx,
                chat_id=555,
                user_id=777,
                error=error,
                retry_handler=retry_handler,
            )
        )

    assert len(bot.sent_messages) == 1
    message = bot.sent_messages[0]
    assert message.chat_id == 555
    assert (
        message.text
        == "Не получилось выполнить запрос. Похоже, в тексте есть запрещённый или "
        "негативный контент. Попробуйте переформулировать короче и нейтральнее."
    )
    retry_id = _get_retry_id(message)
    assert retry_id in ctx.chat_data["veo_fast_retry_callbacks"]

    asyncio.run(veo_fast_module.trigger_retry_callback(ctx, retry_id))
    assert called == ["retry"]
    assert retry_id not in ctx.chat_data["veo_fast_retry_callbacks"]

    records = [rec for rec in caplog.records if rec.message == "ERR_USER_SENT"]
    assert records, "log record not found"
    log_record = records[0]
    assert log_record.kind == "content_policy"
    assert log_record.chat_id == 555
    assert log_record.user_id == 777
    assert log_record.req_id == "req-1"


def test_timeout_shows_retry_and_retries_job() -> None:
    bot = DummyBot()
    ctx = _make_context(bot)
    retry_calls: list[str] = []

    async def retry_handler() -> None:
        retry_calls.append("again")

    asyncio.run(
        veo_fast_module.handle_veo_fast_error(
            ctx,
            chat_id=42,
            user_id=13,
            error=veo_fast_module.VEOFastTimeout("timeout"),
            retry_handler=retry_handler,
        )
    )

    assert len(bot.sent_messages) == 1
    message = bot.sent_messages[0]
    assert (
        message.text
        == "Сервис сейчас отвечает дольше обычного. Нажмите «Повторить» или попробуйте позже."
    )
    retry_id = _get_retry_id(message)
    asyncio.run(veo_fast_module.trigger_retry_callback(ctx, retry_id))
    assert retry_calls == ["again"]


def test_backend_5xx_shows_backend_fail() -> None:
    bot = DummyBot()
    ctx = _make_context(bot)

    asyncio.run(
        veo_fast_module.handle_veo_fast_error(
            ctx,
            chat_id=9,
            user_id=8,
            error=veo_fast_module.VEOFastHTTPError(503, payload={"error": "oops"}),
            retry_handler=lambda: None,
        )
    )

    assert len(bot.sent_messages) == 1
    message = bot.sent_messages[0]
    assert (
        message.text
        == "Не удалось получить результат. Проверьте текст запроса и попробуйте снова."
    )
    assert message.reply_markup is not None


def test_invalid_input_prompts_user() -> None:
    bot = DummyBot()
    ctx = _make_context(bot)

    asyncio.run(
        veo_fast_module.handle_veo_fast_error(
            ctx,
            chat_id=77,
            user_id=88,
            error=veo_fast_module.VEOFastInvalidInput("missing"),
            retry_handler=lambda: None,
        )
    )

    assert len(bot.sent_messages) == 1
    message = bot.sent_messages[0]
    assert (
        message.text
        == "Нужна фотография или корректный промт. Пришлите изображение и повторите."
    )
    assert message.reply_markup is None
