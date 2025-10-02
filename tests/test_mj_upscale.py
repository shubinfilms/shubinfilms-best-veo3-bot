import asyncio
from types import SimpleNamespace

import pytest

from tests.suno_test_utils import FakeBot, bot_module


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def ctx():
    return SimpleNamespace(bot=FakeBot(), user_data={})


@pytest.fixture
def state(ctx):
    return bot_module.state(ctx)


def test_mj_upscale_from_grid_ok(monkeypatch, ctx, state):
    chat_id = 101
    user_id = 505
    state["mode"] = "mj_upscale"
    state["mj_locale"] = "ru"
    state["mj_last_grid"] = {"task_id": "grid123", "result_urls": ["url1", "url2", "url3", "url4"]}

    submissions = {}

    async def fake_show_balance(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot_module, "show_balance_notification", fake_show_balance)

    def fake_debit(uid, price, **_):
        assert price == bot_module.PRICE_MJ_UPSCALE
        return True, 90

    monkeypatch.setattr(bot_module, "debit_try", fake_debit)
    monkeypatch.setattr(bot_module, "credit_balance", lambda *args, **kwargs: None)

    calls_status = []

    def fake_status(task_id):
        calls_status.append(task_id)
        return True, 1, {"resultUrl": "https://img/upscaled.png"}

    monkeypatch.setattr(bot_module, "mj_status", fake_status)

    def fake_generate_upscale(task_id, index, **_):
        submissions[(task_id, index)] = True
        return True, "upscale-1", "ok"

    monkeypatch.setattr(bot_module, "mj_generate_upscale", fake_generate_upscale)

    monkeypatch.setattr(bot_module, "_download_mj_image_bytes", lambda url, idx: (b"image-bytes", "name.png"))

    sent_docs = {}

    async def fake_send_image(bot, chat_id_arg, data, filename, **_):
        sent_docs[chat_id_arg] = (data, filename)

    monkeypatch.setattr(bot_module, "send_image_as_document", fake_send_image)
    monkeypatch.setattr(bot_module, "acquire_mj_upscale_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr(bot_module, "release_mj_upscale_lock", lambda *args, **kwargs: None)

    message = SimpleNamespace(chat_id=chat_id, replies=[])

    async def reply_text(text, **_):
        message.replies.append(text)

    message.reply_text = reply_text

    async def answer(*_args, **_kwargs):
        return None

    query = SimpleNamespace(
        data="mj_upscale:select:1",
        message=message,
        answer=answer,
    )

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id, language_code="ru"),
        callback_query=query,
    )

    _run(bot_module.on_callback(update, ctx))

    assert submissions == {("grid123", 1): True}
    assert sent_docs[chat_id][1] == "mj_upscaled_grid123_1.png"
    assert calls_status[-1] == "upscale-1"


def test_mj_upscale_from_file_ok(monkeypatch, ctx, state):
    chat_id = 303
    user_id = 606
    state["mode"] = "mj_upscale"

    async def fake_show_balance(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot_module, "show_balance_notification", fake_show_balance)

    def fake_debit(uid, price, **_):
        assert price == bot_module.PRICE_MJ_UPSCALE
        return True, 80

    monkeypatch.setattr(bot_module, "debit_try", fake_debit)
    refund_calls = []
    monkeypatch.setattr(bot_module, "credit_balance", lambda *args, **kwargs: refund_calls.append(args))

    status_calls = []

    def fake_status(task_id):
        status_calls.append(task_id)
        if task_id == "grid-task":
            return True, 1, {
                "resultUrls": [
                    "https://img.example/a.png",
                    "https://img.example/b.png",
                    "https://img.example/c.png",
                    "https://img.example/d.png",
                ]
            }
        return True, 1, {"resultUrl": "https://img/upscaled.png"}

    monkeypatch.setattr(bot_module, "mj_status", fake_status)
    monkeypatch.setattr(bot_module, "mj_generate_img2img", lambda *_args, **_kwargs: (True, "grid-task", "ok"))
    monkeypatch.setattr(bot_module, "mj_generate_upscale", lambda *_args, **_kwargs: (True, "upscale-task", "ok"))
    monkeypatch.setattr(bot_module, "_download_mj_image_bytes", lambda url, idx: (b"image", "name.png"))
    monkeypatch.setattr(bot_module, "acquire_mj_upscale_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr(bot_module, "release_mj_upscale_lock", lambda *args, **kwargs: None)

    sent_docs = {}

    async def fake_send_image(bot, chat_id_arg, data, filename, **_):
        sent_docs[chat_id_arg] = filename

    monkeypatch.setattr(bot_module, "send_image_as_document", fake_send_image)

    class FakeFile:
        file_path = "photos/file.jpg"

    async def fake_get_file(_file_id):
        return FakeFile()

    ctx.bot.get_file = fake_get_file  # type: ignore[assignment]

    document = SimpleNamespace(file_id="doc1", mime_type="image/png", width=512, height=512)

    async def reply_text(*args, **kwargs):
        return None

    message = SimpleNamespace(document=document, reply_text=reply_text)
    update = SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id, language_code="ru"),
    )

    _run(bot_module.on_document(update, ctx))

    assert sent_docs[chat_id] == "mj_upscaled_grid-task_0.png"
    assert not refund_calls


def test_mj_upscale_bad_status(monkeypatch, ctx, state):
    chat_id = 404
    user_id = 808
    state["mode"] = "mj_upscale"
    state["mj_last_grid"] = {"task_id": "grid-bad", "result_urls": ["a", "b"]}

    async def fake_show_balance(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot_module, "show_balance_notification", fake_show_balance)
    monkeypatch.setattr(bot_module, "debit_try", lambda *_args, **_kwargs: (True, 70))

    refunds = []

    def fake_credit(uid, price, **kwargs):
        refunds.append((uid, price))
        return 100

    monkeypatch.setattr(bot_module, "credit_balance", fake_credit)
    monkeypatch.setattr(bot_module, "acquire_mj_upscale_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr(bot_module, "release_mj_upscale_lock", lambda *args, **kwargs: None)

    def fake_generate_upscale(*_args, **_kwargs):
        return True, "upscale-bad", "ok"

    monkeypatch.setattr(bot_module, "mj_generate_upscale", fake_generate_upscale)

    def fake_status(task_id):
        return True, 3, {"errorMessage": "failure"}

    monkeypatch.setattr(bot_module, "mj_status", fake_status)
    monkeypatch.setattr(bot_module, "_download_mj_image_bytes", lambda *_args, **_kwargs: (b"", ""))
    monkeypatch.setattr(bot_module, "send_image_as_document", lambda *args, **kwargs: None)

    async def bad_answer(*_args, **_kwargs):
        return None

    async def bad_reply(*_args, **_kwargs):
        return None

    query = SimpleNamespace(
        data="mj_upscale:select:0",
        message=SimpleNamespace(chat_id=chat_id, reply_text=bad_reply),
        answer=bad_answer,
    )

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id, language_code="ru"),
        callback_query=query,
    )

    _run(bot_module.on_callback(update, ctx))

    assert refunds == [(user_id, bot_module.PRICE_MJ_UPSCALE)]


def test_mj_upscale_no_context(monkeypatch, ctx, state):
    chat_id = 505
    user_id = 909
    state["mode"] = "mj_upscale"
    state["mj_last_grid"] = None

    sent_messages = []

    async def fake_send_message(*args, **kwargs):
        text = kwargs.get("text")
        if text is None and args:
            if len(args) >= 2:
                text = args[1]
            else:
                text = args[0]
        sent_messages.append(text or "")

    ctx.bot.send_message = fake_send_message  # type: ignore[assignment]

    async def answer_up(*_args, **_kwargs):
        return None

    async def reply_up(*_args, **_kwargs):
        return None

    query = SimpleNamespace(
        data="mj_upscale:select:0",
        message=SimpleNamespace(chat_id=chat_id, reply_text=reply_up),
        answer=answer_up,
    )

    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id, language_code="ru"),
        callback_query=query,
    )

    _run(bot_module.on_callback(update, ctx))

    assert any("Пришлите фото" in text for text in sent_messages)
