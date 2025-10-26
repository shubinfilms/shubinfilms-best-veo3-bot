import os

from telegram import Message
from telegram.constants import ParseMode


ADMIN_IDS = {
    int(x)
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x
}


async def notify_admin(bot, text: str) -> None:
    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(
                admin_id,
                f"⚠️ {text}",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


async def user_safe_error(msg: Message, user_text: str) -> None:
    try:
        await msg.reply_text(user_text, disable_web_page_preview=True)
    except Exception:
        pass
