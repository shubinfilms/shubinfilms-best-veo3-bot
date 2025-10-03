"""Webhook processing for YooKassa payments."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

from ledger import LedgerStorage
from texts import common_text

from .yookassa_storage import (
    acquire_payment_lock,
    load_pending_payment,
    release_payment_lock,
    update_pending_status,
)

log = logging.getLogger("payments.yookassa.callback")

_TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
_TELEGRAM_BASE_URL = (
    f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}" if _TELEGRAM_TOKEN else None
)
_TELEGRAM_SESSION = requests.Session()

_SUCCESS_STICKER_ID = "5471952986970267163"

_ledger_instance: Optional[LedgerStorage] = None


def _ledger() -> LedgerStorage:
    global _ledger_instance
    if _ledger_instance is None:
        backend = (os.getenv("LEDGER_BACKEND") or "postgres").lower()
        dsn = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
        _ledger_instance = LedgerStorage(dsn, backend=backend)
    return _ledger_instance


def _telegram_request(method: str, payload: Dict[str, Any]) -> None:
    if not _TELEGRAM_BASE_URL:
        log.warning("topup.telegram.missing_token")
        return
    url = f"{_TELEGRAM_BASE_URL}/{method}"
    try:
        response = _TELEGRAM_SESSION.post(url, json=payload, timeout=20)
    except requests.RequestException as exc:  # pragma: no cover - network failure
        log.warning("topup.telegram.network", extra={"meta": {"method": method, "err": str(exc)}})
        return
    if not response.ok:
        log.warning(
            "topup.telegram.failed",
            extra={"meta": {"method": method, "status": response.status_code, "body": response.text[:200]}},
        )


def _notify_success(user_id: int, new_balance: int) -> None:
    _telegram_request("sendSticker", {"chat_id": user_id, "sticker": _SUCCESS_STICKER_ID})
    text = common_text("balance.success", new_balance=new_balance)
    _telegram_request("sendMessage", {"chat_id": user_id, "text": text})


def process_callback(payload: Dict[str, Any]) -> Dict[str, Any]:
    event = payload.get("event")
    object_data = payload.get("object") or {}
    payment_id = str(object_data.get("id")) if object_data.get("id") else None

    meta = {"event": event, "payment_id": payment_id}
    log.info("topup.yookassa.callback", extra={"meta": meta})

    if event != "payment.succeeded" or not payment_id:
        return {"status": "ignored"}

    if not acquire_payment_lock(payment_id):
        log.info("topup.yookassa.locked", extra={"meta": meta})
        return {"status": "locked"}

    try:
        pending = load_pending_payment(payment_id)
        if not pending:
            log.warning("topup.yookassa.unknown_payment", extra={"meta": meta})
            return {"status": "missing"}

        if pending.status == "SUCCESS":
            log.info("topup.yookassa.duplicate", extra={"meta": meta})
            return {"status": "duplicate"}

        op_id = f"yk:{payment_id}"
        ledger = _ledger()
        result = ledger.credit(
            pending.user_id,
            pending.tokens_to_add,
            "yookassa_topup",
            op_id,
            {"pack_id": pending.pack_id, "amount": pending.amount, "currency": pending.currency},
        )
        update_pending_status(payment_id, "SUCCESS")
        balance = int(result.balance)

        log.info(
            "topup.success",
            extra={"meta": {"payment_id": payment_id, "user_id": pending.user_id, "balance": balance}},
        )

        _notify_success(pending.user_id, balance)
        return {"status": "success", "balance": balance}
    except Exception as exc:
        log.exception(
            "topup.failed", extra={"meta": {"payment_id": payment_id, "error": str(exc)}}
        )
        update_pending_status(payment_id, "FAILED")
        return {"status": "error", "error": str(exc)}
    finally:
        release_payment_lock(payment_id)


__all__ = ["process_callback"]
