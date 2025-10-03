"""Integration helpers for YooKassa payments."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, List, Optional

import requests

from settings import (
    HTTP_TIMEOUT_CONNECT,
    HTTP_TIMEOUT_READ,
    YOOKASSA_CURRENCY,
    YOOKASSA_RETURN_URL,
    YOOKASSA_SECRET_KEY,
    YOOKASSA_SHOP_ID,
)
from texts import common_text

from .yookassa_storage import PendingPayment, save_pending_payment

log = logging.getLogger("payments.yookassa")

_API_BASE = os.getenv("YOOKASSA_API_BASE", "https://api.yookassa.ru").rstrip("/")
_PAYMENTS_PATH = "/v3/payments"
_RETRY_DELAYS = (0.0, 1.0, 3.0)


@dataclass(frozen=True)
class YookassaPack:
    pack_id: str
    label_key: str
    amount: Decimal
    tokens: int

    @property
    def button_label(self) -> str:
        return common_text(self.label_key)


@dataclass
class YookassaPayment:
    payment_id: str
    confirmation_url: str
    amount: str
    currency: str
    user_id: int
    pack_id: str
    tokens_to_add: int
    idempotence_key: str
    created_at: float


class YookassaError(RuntimeError):
    """Base exception for YooKassa operations."""


YOOKASSA_PACKS: Dict[str, YookassaPack] = {
    "pack_1": YookassaPack("pack_1", "topup.yookassa.pack_1", Decimal("199.00"), 120),
    "pack_2": YookassaPack("pack_2", "topup.yookassa.pack_2", Decimal("399.00"), 360),
    "pack_3": YookassaPack("pack_3", "topup.yookassa.pack_3", Decimal("899.00"), 900),
}

# Maintain deterministic order for menu rendering.
YOOKASSA_PACKS_ORDER: List[YookassaPack] = [YOOKASSA_PACKS[key] for key in sorted(YOOKASSA_PACKS)]


def list_packs() -> Iterable[YookassaPack]:
    return list(YOOKASSA_PACKS_ORDER)


def get_pack(pack_id: str) -> Optional[YookassaPack]:
    return YOOKASSA_PACKS.get(pack_id)


def _authorization_header(shop_id: str, secret_key: str) -> str:
    token = base64.b64encode(f"{shop_id}:{secret_key}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _make_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _make_session()


def _request_timeout() -> tuple[float, float]:
    connect = max(0.5, float(HTTP_TIMEOUT_CONNECT))
    read = max(1.0, float(HTTP_TIMEOUT_READ))
    return connect, read


def _format_amount(amount: Decimal) -> str:
    normalized = amount.quantize(Decimal("0.01"))
    return f"{normalized:.2f}"


def _validate_configuration() -> None:
    if not (YOOKASSA_SHOP_ID and YOOKASSA_SECRET_KEY and YOOKASSA_RETURN_URL):
        raise YookassaError("YooKassa configuration is incomplete")


def create_payment(user_id: int, pack_id: str) -> YookassaPayment:
    """Create a YooKassa payment for ``user_id`` and ``pack_id``."""

    _validate_configuration()

    pack = get_pack(pack_id)
    if not pack:
        raise YookassaError(f"Unknown YooKassa pack: {pack_id}")

    idempotence_key = f"yk:{user_id}:{pack_id}:{uuid.uuid4()}"
    payload = {
        "amount": {"value": _format_amount(pack.amount), "currency": YOOKASSA_CURRENCY or "RUB"},
        "capture": True,
        "confirmation": {"type": "redirect", "return_url": YOOKASSA_RETURN_URL},
        "description": f"Best AI Bot: {pack_id} for user {user_id}",
        "metadata": {"user_id": user_id, "pack_id": pack_id, "tokens": pack.tokens},
    }

    headers = {
        "Content-Type": "application/json",
        "Idempotence-Key": idempotence_key,
        "Authorization": _authorization_header(YOOKASSA_SHOP_ID, YOOKASSA_SECRET_KEY),
    }

    connect, read = _request_timeout()
    url = f"{_API_BASE}{_PAYMENTS_PATH}"

    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        if delay:
            time.sleep(delay)
        try:
            response = _SESSION.post(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                timeout=(connect, read),
            )
        except requests.RequestException as exc:
            log.warning(
                "topup.yookassa.create.network",
                extra={"meta": {"user_id": user_id, "pack_id": pack_id, "attempt": attempt, "err": str(exc)}},
            )
            if attempt == len(_RETRY_DELAYS):
                raise YookassaError("Не удалось создать платёж в YooKassa") from exc
            continue

        if response.status_code >= 500 and attempt < len(_RETRY_DELAYS):
            log.warning(
                "topup.yookassa.create.retry",
                extra={"meta": {"user_id": user_id, "pack_id": pack_id, "status": response.status_code}},
            )
            continue

        if response.status_code >= 400:
            log.error(
                "topup.yookassa.create.failed",
                extra={"meta": {"user_id": user_id, "pack_id": pack_id, "status": response.status_code, "body": response.text[:400]}},
            )
            raise YookassaError(f"YooKassa responded with status {response.status_code}")

        try:
            payload_json = response.json()
        except ValueError as exc:  # pragma: no cover - invalid JSON
            log.error(
                "topup.yookassa.create.invalid_json",
                extra={"meta": {"user_id": user_id, "pack_id": pack_id, "status": response.status_code}},
            )
            raise YookassaError("YooKassa returned invalid JSON") from exc

        payment_id = payload_json.get("id")
        confirmation = payload_json.get("confirmation") or {}
        confirmation_url = confirmation.get("confirmation_url") or confirmation.get("url")

        if not payment_id or not confirmation_url:
            log.error(
                "topup.yookassa.create.malformed",
                extra={"meta": {"user_id": user_id, "pack_id": pack_id, "body": payload_json}},
            )
            raise YookassaError("YooKassa response missing payment details")

        payment = YookassaPayment(
            payment_id=str(payment_id),
            confirmation_url=str(confirmation_url),
            amount=_format_amount(pack.amount),
            currency=YOOKASSA_CURRENCY or "RUB",
            user_id=user_id,
            pack_id=pack.pack_id,
            tokens_to_add=pack.tokens,
            idempotence_key=idempotence_key,
            created_at=time.time(),
        )

        log.info(
            "topup.yookassa.create", extra={"meta": {"user_id": user_id, "pack_id": pack.pack_id, "payment_id": payment.payment_id}}
        )

        save_pending_payment(
            PendingPayment(
                payment_id=payment.payment_id,
                user_id=user_id,
                pack_id=pack.pack_id,
                amount=payment.amount,
                currency=payment.currency,
                tokens_to_add=pack.tokens,
                idempotence_key=idempotence_key,
            )
        )

        return payment

    raise YookassaError("Не удалось создать платёж в YooKassa")


def pack_button_label(pack_id: str) -> str:
    pack = get_pack(pack_id)
    if not pack:
        return pack_id
    return pack.button_label


__all__ = [
    "YookassaPack",
    "YookassaPayment",
    "YOOKASSA_PACKS",
    "YOOKASSA_PACKS_ORDER",
    "create_payment",
    "get_pack",
    "list_packs",
    "pack_button_label",
    "YookassaError",
]
