from __future__ import annotations

import hashlib
import json
from typing import Optional

from telegram import InlineKeyboardMarkup
from telegram.error import BadRequest


_last_hash: dict[str, str] = {}


def _payload_hash(text: str, reply_markup: Optional[InlineKeyboardMarkup]) -> str:
    markup_payload = reply_markup.to_dict() if reply_markup else None
    raw = json.dumps({"t": text or "", "rm": markup_payload}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def safe_edit(
    bot,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
):
    """Edit a Telegram message only when payload changes."""

    key = f"{chat_id}:{message_id}"
    payload_hash = _payload_hash(text, reply_markup)
    if _last_hash.get(key) == payload_hash:
        return
    try:
        return bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except BadRequest as exc:
        if "not modified" in str(exc).lower():
            _last_hash[key] = payload_hash
            return
        raise
    else:
        _last_hash[key] = payload_hash
