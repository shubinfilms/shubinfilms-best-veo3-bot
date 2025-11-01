"""Helpers to render Banana card UI."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def banana_card_text(balance: int, state) -> str:
    """Return the Banana card caption for ``state``."""

    photos = f"📸 Фото: {len(state.images)}/4"
    prompt_state = "есть" if state.prompt else "нет"
    return f"🍌 Карточка Banana\n💎 Баланс: {balance}\n{photos} • Промпт: {prompt_state}"


def banana_card_kb(state) -> InlineKeyboardMarkup:
    """Return inline keyboard for Banana card based on readiness."""

    rows: list[list[InlineKeyboardButton]] = []
    if state.ready:
        rows.append(
            [InlineKeyboardButton("🚀 Начать генерацию", callback_data="banana:start")]
        )
    rows.append([InlineKeyboardButton("✨ Готовые шаблоны", callback_data="banana:templates")])
    rows.append(
        [
            InlineKeyboardButton("⚙️ Движок", callback_data="img_engine:banana"),
            InlineKeyboardButton("⬅️ Назад", callback_data="back_main"),
        ]
    )
    return InlineKeyboardMarkup(rows)


__all__ = ["banana_card_text", "banana_card_kb"]
