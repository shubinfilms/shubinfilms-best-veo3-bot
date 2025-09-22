# -*- coding: utf-8 -*-
"""Prompt-Master text constants."""

PROMPT_MASTER_HEADER = "🧠 Prompt-Master 2.0"
PROMPT_MASTER_BODY = (
    "Опиши идею (1–3 предложения). Если нужна озвучка — укажи язык и характер голоса.\n"
    "Можно добавить тех. детали (например: “85mm prime, shallow DOF, real-time”)."
)
PROMPT_MASTER_HINT = f"{PROMPT_MASTER_HEADER}\n{PROMPT_MASTER_BODY}"

# Поддержка старого имени, если где-то ещё используется.
PROMPT_MASTER_INVITE_TEXT = PROMPT_MASTER_HINT

__all__ = [
    "PROMPT_MASTER_HEADER",
    "PROMPT_MASTER_BODY",
    "PROMPT_MASTER_HINT",
    "PROMPT_MASTER_INVITE_TEXT",
]
