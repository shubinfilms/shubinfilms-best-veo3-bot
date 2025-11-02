"""UI helpers for rendering the Banana card."""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from utils.banana_state import BananaState


def banana_card_text(balance: int, state: BananaState) -> str:
    """Return the textual representation of the Banana card."""

    photos = f"📸 Фото: {len(state.images)}/4"
    prompt_status = "есть" if state.prompt else "нет"
    lines = [
        "🍌 Карточка Banana",
        f"💎 Баланс: {balance}",
        f"{photos} • Промпт: {prompt_status}",
    ]
    return "\n".join(lines)


def banana_card_kb(state: BananaState) -> InlineKeyboardMarkup:
    """Return the inline keyboard for the Banana card."""

    rows = []
    if state.ready:
        rows.append([
            InlineKeyboardButton("🚀 Начать генерацию", callback_data="banana:start"),
        ])
    rows.append([
        InlineKeyboardButton("✨ Готовые шаблоны", callback_data="banana:templates"),
    ])
    rows.append(
        [
            InlineKeyboardButton("⚙️ Движок", callback_data="img_engine:banana"),
            InlineKeyboardButton("⬅️ Назад", callback_data="back_main"),
        ]
    )
    return InlineKeyboardMarkup(rows)


__all__ = ["banana_card_text", "banana_card_kb"]

