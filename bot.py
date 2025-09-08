# bot.py
# Python-Telegram-Bot v20.7
import os
import io
import json
import time
import asyncio
import logging
from typing import Optional, Tuple, List, Dict

import requests
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputMediaVideo, InputMediaPhoto
)
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    MessageHandler, CallbackQueryHandler, filters
)

# ---------- ENV ----------
TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
KIE_API_KEY = os.getenv("KIE_API_KEY", "").strip()
KIE_BASE = os.getenv("KIE_BASE", "https://api.kie.ai").rstrip("/")

if not TG_TOKEN or not KIE_API_KEY:
    raise SystemExit("Set TELEGRAM_TOKEN and KIE_API_KEY in environment")

# ---------- LOG ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("best-veo3-bot")

# ---------- STATE ----------
# очень простой in-memory стейт на пользователя
USERS: Dict[int, Dict] = {}

def get_state(uid: int) -> Dict:
    st = USERS.setdefault(uid, {
        "mode": None,          # "veo_text" | "veo_photo" | "mj_face"
        "prompt": "",
        "photo_file_id": None,
        "ratio": "16:9",       # "1:1" | "16:9" | "9:16" | "3:4"
        "speed": "fast",       # "fast" | "quality" (для VEO) / "fast|relaxed|turbo" (для MJ)
        "mj_speed": "fast",
        "mj_version": "7",
        "stylization": 50,
        "weirdness": 0,
        "variety": 5,
        "_busy": False,        # анти-даблклик
    })
    return st

# ---------- Keyboards ----------
def kb_main() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton("🎬 Сгенерировать по тексту (VEO)", callback_data="veo_text")],
        [InlineKeyboardButton("🖼️ Сгенерировать по фото (VEO)", callback_data="veo_photo")],
        [InlineKeyboardButton("👤 Фото с вашим лицом (MJ)", callback_data="mj_face")],
        [InlineKeyboardButton("💳 Пополнить баланс", url="https://kie.ai/pricing")],
    ]
    return InlineKeyboardMarkup(rows)

def kb_params_common(st: Dict, for_mj: bool) -> InlineKeyboardMarkup:
    r1 = [
        InlineKeyboardButton("1:1" , callback_data="ratio:1:1"),
        InlineKeyboardButton("16:9", callback_data="ratio:16:9"),
        InlineKeyboardButton("9:16", callback_data="ratio:9:16"),
        InlineKeyboardButton("3:4" , callback_data="ratio:3:4"),
    ]
    if for_mj:
        r2 = [
            InlineKeyboardButton(("🐢 relaxed" + (" ✅" if st["mj_speed"]=="relaxed" else "")), callback_data="mjspeed:relaxed"),
            InlineKeyboardButton(("⚡ fast"    + (" ✅" if st["mj_speed"]=="fast"    else "")), callback_data="mjspeed:fast"),
            InlineKeyboardButton(("🚀 turbo"   + (" ✅" if st["mj_speed"]=="turbo"   else "")), callback_data="mjspeed:turbo"),
        ]
        r3 = [InlineKeyboardButton("🧠 Изм. промпт", callback_data="prompt_edit"),
              InlineKeyboardButton("📸 Добавить/Удалить селфи", callback_data="photo_toggle")]
        r4 = [InlineKeyboardButton("🧩 Сгенерировать фото", callback_data="run_mj"),
              InlineKeyboardButton("⬅️ Назад", callback_data="back")]
        r5 = [InlineKeyboardButton("💳 Пополнить баланс", url="https://kie.ai/pricing")]
        return InlineKeyboardMarkup([r3, r1, r2, r4, r5])
    else:
        r2 = [
            InlineKeyboardButton(("⚡ fast"    + (" ✅" if st["speed"]=="fast"    else "")), callback_data="speed:fast"),
            InlineKeyboardButton(("💎 quality" + (" ✅" if st["speed"]=="quality" else "")), callback_data="speed:quality"),
        ]
        r3 = [InlineKeyboardButton("🧠 Изм. промпт", callback_data="prompt_edit"),
              InlineKeyboardButton("📸 Добавить/Удалить фото", callback_data="photo_toggle")]
        r4 = [InlineKeyboardButton("🎬 Сгенерировать видео", callback_data="run_veo"),
              InlineKeyboardButton("⬅️ Назад", callback_data="back")]
        r5 = [InlineKeyboardButton("💳 Пополнить баланс", url="https://kie.ai/pricing")]
        return InlineKeyboardMarkup([r3, r1, r2, r4, r5])

# ---------- Helpers: Telegram file URL + KIE upload ----------
def tg_file_direct_url(bot_token: str, file_id: str) -> str:
    from telegram import Bot
    bot = Bot(bot_token)
    f = bot.get_file(file_id)
    return f"https://api.telegram.org/file/bot{bot_token}/{f.file_path}"

def kie_try_upload_endpoints(data_bytes: bytes, filename="image.jpg", mime="image/jpeg") -> str:
    """
    Пробуем несколько известных путей аплоада KIE.
    Возвращаем data.fileUrl.
    """
    endpoints = [
        f"{KIE_BASE}/api/v1/file/upload",
        f"{KIE_BASE}/common-api/file/upload",
        f"{KIE_BASE}/api/v1/common/file/upload",
    ]
    last_error = None
    for url in endpoints:
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {KIE_API_KEY}"},
                files={"file": (filename, io.BytesIO(data_bytes), mime)},
                timeout=60,
            )
            j = r.json()
            if r.ok and j.get("code") == 200 and "data" in j and j["data"].get("fileUrl"):
                return j["data"]["fileUrl"]
            last_error = f"{url} -> {j}"
        except Exception as e:
            last_error = f"{url} -> {e}"
    raise RuntimeError(f"KIE upload failed: {last_error}")

def upload_tg_photo_to_kie(bot_token: str, file_id: str) -> str:
    # 1) скачать байты из TG
    direct = tg_file_direct_url(bot_token, file_id)
    r = requests.get(direct, timeout=30)
    r.raise_for_status()
    # 2) загрузить в KIE
    return kie_try_upload_endpoints(r.content)

# ---------- KIE: Midjourney ----------
def clamp(v, lo, hi):
    try:
        v = int(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))

def mj_generate(task_type: str, prompt: str, aspect_ratio: str,
                speed: str, version: str,
                file_url: Optional[str] = None,
                stylization: int = 50, weirdness: int = 0, variety: int = 5) -> dict:
    payload = {
        "taskType": task_type,                   # 'mj_txt2img' | 'mj_img2img' | 'mj_video'
        "prompt": prompt[:2000],
        "aspectRatio": aspect_ratio,
        "speed": speed,                          # relaxed|fast|turbo
        "version": version,                      # '7'|'6.1'|'6'|'5.2'|'5.1'|'niji6'
        "stylization": clamp(stylization, 0, 1000),
        "weirdness": clamp(weirdness, 0, 3000),
        "variety": clamp(variety, 0, 100),
        "enableTranslation": False,
    }
    if file_url:
        payload["fileUrl"] = file_url

    r = requests.post(
        f"{KIE_BASE}/api/v1/mj/generate",
        headers={"Authorization": f"Bearer {KIE_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    try:
        return r.json()
    except Exception:
        return {"code": 500, "msg": f"http error {r.status_code}"}

def kie_poll(kind: str, task_id: str) -> dict:
    r = requests.get(
        f"{KIE_BASE}/api/v1/{kind}/record-info",
        params={"taskId": task_id},
        headers={"Authorization": f"Bearer {KIE_API_KEY}"},
        timeout=30,
    )
    try:
        j = r.json()
    except Exception:
        return {"final": True, "ok": False, "error": f"http {r.status_code}"}

    if j.get("code") != 200:
        return {"final": True, "ok": False, "error": f"{j.get('code')} {j.get('msg')}"}

    data = j.get("data") or {}
    flag = data.get("successFlag")
    if flag == 0:
        return {"final": False, "ok": True, "data": data}
    if flag == 1:
        return {"final": True, "ok": True, "data": data}
    if flag in (2, 3):
        return {"final": True, "ok": False, "error": data.get("errorMessage") or f"flag={flag}", "data": data}
    return {"final": False, "ok": True, "data": data}

def extract_result_urls(data: dict) -> List[str]:
    out = []
    res = (data or {}).get("resultInfoJson") or {}
    for key in ("result_urls", "resultUrls"):
        v = res.get(key)
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict) and x.get("resultUrl"):
                    out.append(x["resultUrl"])
    return out

# ---------- KIE: Veo 3 ----------
def veo_generate_text(prompt: str, aspect_ratio: str, speed: str) -> dict:
    payload = {
        "taskType": "veo_txt2vid",
        "prompt": prompt[:2000],
        "aspectRatio": aspect_ratio,            # "16:9"|"9:16"|"1:1"|"3:4"
        "speed": "fast" if speed == "fast" else "quality",
    }
    r = requests.post(
        f"{KIE_BASE}/api/v1/veo/generate",
        headers={"Authorization": f"Bearer {KIE_API_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=60
    )
    try:
        return r.json()
    except Exception:
        return {"code": 500, "msg": f"http error {r.status_code}"}

def veo_generate_with_ref(prompt: str, aspect_ratio: str, speed: str, file_url: str) -> dict:
    payload = {
        "taskType": "veo_img_ref",              # если у вашего провайдера иначе — подставьте нужное значение
        "prompt": prompt[:2000],
        "aspectRatio": aspect_ratio,
        "speed": "fast" if speed == "fast" else "quality",
        "fileUrl": file_url,
    }
    r = requests.post(
        f"{KIE_BASE}/api/v1/veo/generate",
        headers={"Authorization": f"Bearer {KIE_API_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=60
    )
    try:
        return r.json()
    except Exception:
        return {"code": 500, "msg": f"http error {r.status_code}"}

# ---------- Bot flows ----------
WELCOME = (
    "🎥 *Veo 3 — супер-генерация видео*\n"
    "Опиши идею — получишь готовый клип. Поддерживаются 16:9 и 9:16, режимы Fast/Quality, фото-референс.\n\n"
    "🖼️ *Midjourney* — фотогенерация, включая портреты по селфи.\n\n"
    "Выберите режим ниже 👇"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    st = get_state(update.effective_user.id)
    st.update({"mode": None, "prompt": "", "photo_file_id": None, "_busy": False})
    await update.effective_message.reply_text(WELCOME, reply_markup=kb_main(), parse_mode="Markdown")

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ping простого эндпоинта аккаунта (если есть), иначе просто сообщаем версии и пингуем mj record-info с фейк id
    import telegram
    try:
        r = requests.get(f"{KIE_BASE}/common-api/get-account-credits",
                         headers={"Authorization": f"Bearer {KIE_API_KEY}"}, timeout=15)
        ok = r.ok
        msg = r.text[:150]
    except Exception as e:
        ok = False
        msg = str(e)[:150]
    await update.effective_message.reply_text(
        f"PTB: {getattr(telegram, '__version__', 'unknown')}\n"
        f"KIE ping: {'OK' if ok else 'FAIL'}\n{msg}"
    )

async def on_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = update.effective_user.id
    st = get_state(uid)

    data = query.data

    if data == "back":
        st["mode"] = None
        await query.edit_message_text(WELCOME, reply_markup=kb_main(), parse_mode="Markdown")
        return

    if data in ("veo_text", "veo_photo", "mj_face"):
        st["mode"] = data
        header = {
            "veo_text": "🎬 VEO — подготовка к рендеру (по тексту)",
            "veo_photo":"🖼️ VEO — подготовка к рендеру (по фото-референсу)",
            "mj_face":  "👤 MJ — подготовка к рендеру (селфи ➜ фото)",
        }[data]
        need = "Пришлите *промпт*." if data == "veo_text" else "Нужны *селфи* и *промпт*."
        await query.edit_message_text(
            f"{header}\n{need}\n\n"
            f"📝 Промпт: {st['prompt'] or '—'}\n"
            f"📷 Фото: {'есть' if st['photo_file_id'] else 'нет'}\n"
            f"⚙️ Aspect: {st['ratio']}\n",
            reply_markup=kb_params_common(st, for_mj=(data=="mj_face")),
            parse_mode="Markdown"
        )
        return

    if data.startswith("ratio:"):
        st["ratio"] = data.split(":",1)[1]
        await query.edit_message_reply_markup(kb_params_common(st, for_mj=(st["mode"]=="mj_face")))
        return

    if data.startswith("speed:"):
        st["speed"] = data.split(":",1)[1]
        await query.edit_message_reply_markup(kb_params_common(st, for_mj=False))
        return

    if data.startswith("mjspeed:"):
        st["mj_speed"] = data.split(":",1)[1]
        await query.edit_message_reply_markup(kb_params_common(st, for_mj=True))
        return

    if data == "prompt_edit":
        await query.edit_message_text("Пришлите текст промпта одной следующей репликой.")
        st["_await_prompt"] = True
        return

    if data == "photo_toggle":
        await query.edit_message_text("Пришлите фото (jpg/png). Повторное фото перезапишет прежнее.")
        st["_await_photo"] = True
        return

    # --- RUN VEO ---
    if data == "run_veo":
        if st["_busy"]:
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return
        if not st["prompt"]:
            await query.message.reply_text("Нужен промпт.")
            return
        if st["mode"] == "veo_photo" and not st["photo_file_id"]:
            await query.message.reply_text("Нужно фото-референс.")
            return

        st["_busy"] = True
        try:
            if st["mode"] == "veo_text":
                resp = veo_generate_text(st["prompt"], st["ratio"], st["speed"])
            else:
                # upload to KIE
                file_url = upload_tg_photo_to_kie(context.bot.token, st["photo_file_id"])
                resp = veo_generate_with_ref(st["prompt"], st["ratio"], st["speed"], file_url)

            if resp.get("code") != 200:
                await query.message.reply_text(f"❌ Не удалось создать VEO-задачу: {resp.get('msg')}")
                st["_busy"] = False
                return

            task_id = (resp.get("data") or {}).get("taskId")
            await query.message.reply_text(f"🚀 Задача отправлена (VEO). taskId={task_id}\n⏳ Идёт рендеринг…")

            # poll
            for _ in range(60):
                pol = kie_poll("veo", task_id)
                if pol["final"]:
                    if pol["ok"]:
                        urls = extract_result_urls(pol["data"])
                        if urls:
                            u0 = urls[0]
                            if u0.lower().endswith((".mp4",".mov",".webm")):
                                await query.message.reply_video(u0, caption="✅ Готово!")
                            else:
                                await query.message.reply_photo(u0, caption="✅ Готово!")
                        else:
                            await query.message.reply_text("✅ Готово (но без ссылок в ответе KIE).")
                    else:
                        await query.message.reply_text(f"❌ Ошибка KIE (VEO): {pol.get('error','')}")
                    break
                await asyncio.sleep(10)
        finally:
            st["_busy"] = False
        return

    # --- RUN MJ ---
    if data == "run_mj":
        if st["_busy"]:
            await query.message.reply_text("⏳ Генерация уже идёт. Дождитесь завершения.")
            return
        if not st["prompt"] or not st["photo_file_id"]:
            await query.message.reply_text("Нужны селфи и промпт.")
            return
        st["_busy"] = True
        try:
            # upload photo to KIE and use img2img
            file_url = upload_tg_photo_to_kie(context.bot.token, st["photo_file_id"])
            resp = mj_generate(
                task_type="mj_img2img",
                prompt=st["prompt"],
                aspect_ratio=st["ratio"],
                speed=st["mj_speed"],
                version=st["mj_version"],
                file_url=file_url,
                stylization=st["stylization"],
                weirdness=st["weirdness"],
                variety=st["variety"],
            )
            if resp.get("code") != 200:
                await query.message.reply_text(f"❌ Не удалось создать MJ-задачу: {resp.get('msg')}")
                st["_busy"] = False
                return

            task_id = (resp.get("data") or {}).get("taskId")
            await query.message.reply_text(f"🧩 MJ задача отправлена. taskId={task_id}\n⏳ Идёт рендеринг…")

            for _ in range(60):
                pol = kie_poll("mj", task_id)
                if pol["final"]:
                    if pol["ok"]:
                        urls = extract_result_urls(pol["data"])
                        if urls:
                            medias = []
                            # MJ обычно 4 изображения — отправим как альбом (до 4)
                            for i,u in enumerate(urls[:4]):
                                medias.append(InputMediaPhoto(u, caption="✅ Готово!" if i==0 else None))
                            await query.message.reply_media_group(medias)
                        else:
                            await query.message.reply_text("✅ Готово (но без ссылок в ответе KIE).")
                    else:
                        # частая реальная причина ранее: Image fetch failed -> теперь решено аплоадом
                        await query.message.reply_text(f"❌ Ошибка MJ: {pol.get('error','')}")
                    break
                await asyncio.sleep(10)
        finally:
            st["_busy"] = False
        return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = get_state(uid)
    text = (update.message.text or "").strip()

    if st.get("_await_prompt"):
        st["prompt"] = text
        st["_await_prompt"] = False
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        return

    # если пользователь пишет промпт «в лоб» в активном режиме — примем:
    if st["mode"] in ("veo_text","veo_photo","mj_face"):
        st["prompt"] = text
        await update.message.reply_text("🧠 Готово! Промпт добавлен в карточку.")
        return

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = get_state(uid)
    ph = update.message.photo[-1]
    st["photo_file_id"] = ph.file_id
    await update.message.reply_text("🖼️ Фото принято как референс.")

# ---------- App ----------
def main():
    app = ApplicationBuilder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CallbackQueryHandler(on_cb))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_text))
    log.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main(
