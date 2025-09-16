# -*- coding: utf-8 -*-
import importlib
import sys


def _reload_prompt_master(monkeypatch, mode=None):
    if mode is None:
        monkeypatch.delenv("PM_QUOTE_MODE", raising=False)
    else:
        monkeypatch.setenv("PM_QUOTE_MODE", mode)

    if "prompt_master" in sys.modules:
        del sys.modules["prompt_master"]

    return importlib.import_module("prompt_master")


def _all_lines_quoted(text: str) -> bool:
    return all(line.startswith(">") for line in text.splitlines() if line)


def test_prompt_master_generator_mode_blockquote(monkeypatch):
    pm = _reload_prompt_master(monkeypatch, None)
    pm._client = None  # ensure fallback mode

    text = pm.generate_prompt_master("Тёплый осенний вечер в городе, огни витрин, живые эмоции")

    assert text
    assert pm.PM_QUOTE_MODE == "generator"
    assert _all_lines_quoted(text)
    assert "Карточка Prompt-Master" not in text


def test_prompt_master_bot_mode_plain_then_quote(monkeypatch):
    pm = _reload_prompt_master(monkeypatch, "bot")
    pm._client = None  # ensure fallback mode

    text = pm.generate_prompt_master("Контрастное освещение, портрет крупным планом")

    assert text
    assert pm.PM_QUOTE_MODE == "bot"
    assert not text.startswith(">")
    assert "Карточка Prompt-Master" not in text

    quoted = pm.ensure_quote_block(text)
    assert _all_lines_quoted(quoted)

    _reload_prompt_master(monkeypatch, None)  # restore default for other tests
