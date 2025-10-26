from __future__ import annotations

import html
from typing import Iterable, Sequence

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

_SUM_HINT = "Напишите текст — я его суммирую."
_PREVIEW_LIMIT = 600
_RESULT_LIMIT = 2000


def _inline(rows: Sequence[Sequence[InlineKeyboardButton]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([list(row) for row in rows])


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _base_payload(text: str, buttons: Iterable[Iterable[InlineKeyboardButton]]) -> dict:
    return {
        "text": text,
        "reply_markup": _inline([list(row) for row in buttons]),
        "parse_mode": ParseMode.HTML,
        "disable_web_page_preview": True,
    }


def render_sum_card() -> dict:
    text = "<b>🧾 Режим суммирования</b>\n\n" + html.escape(_SUM_HINT)
    buttons = [
        [InlineKeyboardButton("✍️ Ввести текст", callback_data="btn:sum|view=prompt")],
        [InlineKeyboardButton("▶️ Начать генерацию", callback_data="sum:start")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="sum:back")],
    ]
    return _base_payload(text, buttons)


def render_sum_prompt_card() -> dict:
    text = "<b>✍️ Введите текст для суммирования</b>\n\n" + html.escape(_SUM_HINT)
    buttons = [
        [InlineKeyboardButton("▶️ Готово, начать", callback_data="sum:start")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="sum:back")],
    ]
    return _base_payload(text, buttons)


def render_sum_ready_to_start(user_text: str) -> dict:
    preview = _truncate(user_text.strip(), _PREVIEW_LIMIT)
    text = (
        "<b>▶️ Готово к запуску</b>\n\n"
        f"<b>Текст:</b> {html.escape(preview)}\n\n"
        f"{html.escape(_SUM_HINT)}"
    )
    buttons = [
        [InlineKeyboardButton("▶️ Начать генерацию", callback_data="sum:start")],
        [InlineKeyboardButton("✍️ Изменить текст", callback_data="btn:sum|view=prompt")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="sum:back")],
    ]
    return _base_payload(text, buttons)


def render_sum_progress(*, step: str, disabled: bool = False) -> dict:
    label = "⏳ Генерация…" if disabled else "▶️ Начать генерацию"
    buttons = [
        [InlineKeyboardButton(label, callback_data="sum:start")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="sum:back")],
    ]
    status_text = html.escape(step or "Инициализация…")
    text = f"<b>⏳ Выполняю суммирование</b>\n\n{status_text}"
    return _base_payload(text, buttons)


def render_sum_error(message: str) -> dict:
    clean = html.escape(message or "Произошла ошибка.")
    text = f"<b>⚠️ Не удалось получить результат</b>\n\n{clean}"
    buttons = [
        [InlineKeyboardButton("🔁 Повторить", callback_data="sum:start")],
        [InlineKeyboardButton("✍️ Изменить текст", callback_data="btn:sum|view=prompt")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="sum:back")],
    ]
    return _base_payload(text, buttons)


def render_sum_result(result_text: str) -> dict:
    preview = _truncate(result_text.strip(), _RESULT_LIMIT)
    text = f"<b>✅ Результат готов</b>\n\n{html.escape(preview)}"
    buttons = [
        [InlineKeyboardButton("🔁 Ещё раз", callback_data="sum:start")],
        [InlineKeyboardButton("✍️ Новый текст", callback_data="btn:sum|view=prompt")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="sum:back")],
    ]
    return _base_payload(text, buttons)


def render_sum_queue_notice(task_id: str) -> dict:
    text = (
        "<b>⏳ Задача в очереди</b>\n\n"
        "Сервис ещё обрабатывает запрос. Я пришлю результат, как только он появится.\n\n"
        f"<i>ID задачи:</i> {html.escape(task_id)}"
    )
    buttons = [
        [InlineKeyboardButton("⬅️ Назад", callback_data="sum:back")],
    ]
    return _base_payload(text, buttons)
