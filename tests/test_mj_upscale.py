import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def bot_module(monkeypatch):
    monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")
    monkeypatch.setenv("SUNO_API_BASE", "https://example.com")
    monkeypatch.setenv("SUNO_API_TOKEN", "dummy-token")
    monkeypatch.setenv("LEDGER_BACKEND", "memory")
    monkeypatch.setenv("DATABASE_URL", "postgres://test")
    module = importlib.import_module("bot")
    module = importlib.reload(module)
    module.mj_log.disabled = True
    return module


def _make_update(data: str, chat_id: int = 10, user_id: int = 20, markup=None):
    async def answer():
        return None

    async def edit_message_reply_markup(*, reply_markup=None):
        edit_calls.append(reply_markup)

    query = SimpleNamespace(
        data=data,
        message=SimpleNamespace(chat=SimpleNamespace(id=chat_id), reply_markup=markup, message_id=0),
        answer=answer,
        edit_message_reply_markup=edit_message_reply_markup,
        from_user=SimpleNamespace(id=user_id, language_code="ru"),
    )
    return SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id, language_code="ru"),
    )


def test_handle_mj_upscale_menu_shows_select(monkeypatch, bot_module):
    grid = {"task_id": "grid", "result_urls": ["a", "b", "c"]}
    monkeypatch.setattr(bot_module, "_load_mj_grid_snapshot", lambda grid_id: grid)
    monkeypatch.setattr(
        bot_module,
        "get_mj_gallery",
        lambda chat_id, message_id: [
            {"source_url": "a", "file_name": "midjourney_01.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 1}
            for _ in range(3)
        ],
    )

    global edit_calls
    edit_calls = []

    update = _make_update(
        data="mj.upscale.menu:grid",
        markup=bot_module.mj_upscale_root_keyboard("grid"),
    )

    asyncio.run(bot_module.handle_mj_upscale_menu(update, SimpleNamespace()))

    assert edit_calls
    keyboard = edit_calls[0]
    assert len(keyboard.inline_keyboard) == len(grid["result_urls"]) + 1
    assert keyboard.inline_keyboard[-1][0].text == "⬅️ Назад"


def test_handle_mj_upscale_menu_back_to_root(monkeypatch, bot_module):
    grid = {"task_id": "grid", "result_urls": ["a", "b"]}
    monkeypatch.setattr(bot_module, "_load_mj_grid_snapshot", lambda grid_id: grid)
    monkeypatch.setattr(
        bot_module,
        "get_mj_gallery",
        lambda chat_id, message_id: [
            {"source_url": "a", "file_name": "midjourney_01.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 1},
            {"source_url": "b", "file_name": "midjourney_02.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 2},
        ],
    )

    global edit_calls
    edit_calls = []

    select_markup = bot_module.mj_upscale_select_keyboard("grid", count=2)
    update = _make_update(data="mj.upscale.menu:grid", markup=select_markup)

    asyncio.run(bot_module.handle_mj_upscale_menu(update, SimpleNamespace()))

    assert edit_calls
    keyboard = edit_calls[0]
    assert len(keyboard.inline_keyboard) == 3
    assert keyboard.inline_keyboard[0][0].text == "✨ Улучшить качество"


def test_handle_mj_upscale_choice_launches_task(monkeypatch, bot_module):
    grid = {"task_id": "grid", "result_urls": ["a", "b", "c"]}
    monkeypatch.setattr(bot_module, "_load_mj_grid_snapshot", lambda grid_id: grid)
    monkeypatch.setattr(bot_module, "acquire_ttl_lock", lambda name, ttl: True)
    release_calls: list[str] = []
    monkeypatch.setattr(bot_module, "release_ttl_lock", lambda name: release_calls.append(name))
    monkeypatch.setattr(
        bot_module,
        "get_mj_gallery",
        lambda chat_id, message_id: [
            {"source_url": "a", "file_name": "midjourney_01.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 1},
            {"source_url": "b", "file_name": "midjourney_02.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 2},
            {"source_url": "c", "file_name": "midjourney_03.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 3},
        ],
    )

    launch_calls: list[dict[str, object]] = []

    async def fake_launch(chat_id, ctx, *, user_id, grid, image_index, locale, source, **kwargs):
        launch_calls.append(
            {
                "chat_id": chat_id,
                "user_id": user_id,
                "grid": grid,
                "index": image_index,
                "locale": locale,
                "source": source,
            }
        )
        return True

    monkeypatch.setattr(bot_module, "_launch_mj_upscale", fake_launch)
    monkeypatch.setattr(bot_module, "state", lambda ctx: {})

    update = _make_update(data="mj.upscale:grid:2")

    asyncio.run(bot_module.handle_mj_upscale_choice(update, SimpleNamespace()))

    assert launch_calls
    call = launch_calls[0]
    assert call["index"] == 1
    assert call["grid"] == grid
    assert release_calls == ["lock:mj:upscale:grid:2"]


def test_handle_mj_upscale_choice_respects_lock(monkeypatch, bot_module):
    monkeypatch.setattr(bot_module, "acquire_ttl_lock", lambda *args, **kwargs: False)
    monkeypatch.setattr(bot_module, "_launch_mj_upscale", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))
    update = _make_update(data="mj.upscale:grid:1")
    asyncio.run(bot_module.handle_mj_upscale_choice(update, SimpleNamespace()))


def test_handle_mj_upscale_choice_grid_missing(monkeypatch, bot_module):
    monkeypatch.setattr(bot_module, "acquire_ttl_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr(bot_module, "release_ttl_lock", lambda *args: None)
    monkeypatch.setattr(bot_module, "_load_mj_grid_snapshot", lambda *_: None)
    monkeypatch.setattr(
        bot_module,
        "get_mj_gallery",
        lambda chat_id, message_id: [
            {"source_url": "a", "file_name": "midjourney_01.jpeg", "bytes_len": 2048, "mime": "image/jpeg", "sent_message_id": 1}
        ],
    )

    sent_messages: list[tuple[int, str]] = []

    ctx = SimpleNamespace(bot=SimpleNamespace(send_message=None))

    async def fake_send_message(chat_id, text):
        sent_messages.append((chat_id, text))

    monkeypatch.setattr(ctx.bot, "send_message", fake_send_message)

    update = _make_update(data="mj.upscale:grid:1")

    asyncio.run(bot_module.handle_mj_upscale_choice(update, ctx))

    assert sent_messages
