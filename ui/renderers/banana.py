"""Helpers to render Banana card UI."""

from telegram import InlineKeyboardMarkup

from keyboards import banana_card_kb as build_banana_card_kb


def banana_card_text(balance: int, state) -> str:
    """Return the Banana card caption for ``state``."""

    count = len(getattr(state, "images", getattr(state, "photos", [])))
    prompt_value = (getattr(state, "prompt", None) or "").strip()
    prompt_state = "есть" if prompt_value else "нет"
    lines = [
        "🍌 Карточка Banana",
        f"📷 Фото: {count}/4 • ✏️ Промпт: {prompt_state}",
    ]
    if count == 0 and not prompt_value:
        lines.append("Пришлите 1–4 фото (JPEG/PNG до 15 МБ) или текст-промпт.")
    return "\n".join(lines)


def banana_card_kb(state, *, generating: bool = False) -> InlineKeyboardMarkup:
    """Return inline keyboard for Banana card."""

    return build_banana_card_kb(state, generating=generating)


__all__ = ["banana_card_text", "banana_card_kb"]
