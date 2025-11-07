"""Helpers to render Banana card UI."""

from telegram import InlineKeyboardMarkup

from keyboards import banana_card_kb as build_banana_card_kb


def banana_card_text(balance: int, state) -> str:
    """Return the Banana card caption for ``state``."""

    photos = f"📷 Фото: {len(state.images)}/4"
    prompt_state = "есть" if state.prompt else "нет"
    return f"🍌 Карточка Banana\n{photos} • ✏️ Промпт: {prompt_state}"


def banana_card_kb(state, *, generating: bool = False) -> InlineKeyboardMarkup:
    """Return inline keyboard for Banana card."""

    return build_banana_card_kb(state, generating=generating)


__all__ = ["banana_card_text", "banana_card_kb"]
