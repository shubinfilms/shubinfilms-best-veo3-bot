import os, logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN = os.getenv("ADMIN_CHAT_ID")

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Canary bot is alive")

async def ping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def post_init(app: Application):
    if ADMIN:
        try:
            await app.bot.send_message(chat_id=int(ADMIN), text="✅ Canary started")
        except Exception as e:
            print("Admin notify failed:", e)

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("No TELEGRAM_TOKEN in ENV")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.post_init = post_init
    app.run_polling(drop_pending_updates=True, allowed_updates=None)
