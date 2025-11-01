"""Notification helpers for user/admin messaging."""
from __future__ import annotations

import os

ADMIN_ID = int(os.getenv("ADMIN_ID", "0") or 0)


async def user_error(bot, chat_id: int, text: str) -> None:
    """Send a short error notice to the user."""

    await bot.send_message(chat_id, f"❌ {text}")


async def admin_warn(bot, text: str) -> None:
    """Send a warning to the configured admin user, if any."""

    if not ADMIN_ID:
        return
    try:
        await bot.send_message(ADMIN_ID, f"⚠️ {text[:3500]}")
    except Exception:
        pass
