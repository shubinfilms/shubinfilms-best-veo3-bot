import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.suno_test_utils import FakeBot, bot_module
import handlers.knowledge_base as kb_module  # noqa: E402


def test_menu_callbacks_route_ok(monkeypatch):
    ctx = SimpleNamespace(chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))

    calls: list[tuple[str, tuple, dict]] = []

    async def fake_profile(update, context, *, suppress_nav, edit, force_new=False):
        calls.append(("profile", (update, context), {"suppress_nav": suppress_nav}))
        return 101

    async def fake_kb(context, chat_id, *, suppress_nav, fallback_message_id=None):
        calls.append(("kb", (context, chat_id), {"suppress_nav": suppress_nav, "fallback": fallback_message_id}))
        return 102

    async def fake_photo(context, chat_id, *, suppress_nav, fallback_message_id=None):
        calls.append(("photo", (context, chat_id), {"suppress_nav": suppress_nav, "fallback": fallback_message_id}))
        return 103

    async def fake_music(context, chat_id, *, suppress_nav, fallback_message_id=None):
        calls.append(("music", (context, chat_id), {"suppress_nav": suppress_nav, "fallback": fallback_message_id}))
        return 104

    async def fake_video(
        context,
        chat_id,
        *,
        veo_fast_cost,
        veo_photo_cost,
        sora2_cost,
        suppress_nav,
        fallback_message_id=None,
    ):
        calls.append(
            (
                "video",
                (context, chat_id),
                {
                    "suppress_nav": suppress_nav,
                    "fallback": fallback_message_id,
                    "fast": veo_fast_cost,
                    "photo": veo_photo_cost,
                    "sora2": sora2_cost,
                },
            )
        )
        return 105

    async def fake_dialog(context, chat_id, *, suppress_nav, fallback_message_id=None):
        calls.append(("dialog", (context, chat_id), {"suppress_nav": suppress_nav, "fallback": fallback_message_id}))
        return 106

    monkeypatch.setattr(bot_module, "open_profile_card", fake_profile)
    monkeypatch.setattr(bot_module, "knowledge_base_open_root", fake_kb)
    monkeypatch.setattr(bot_module, "photo_open_menu", fake_photo)
    monkeypatch.setattr(bot_module, "music_open_menu", fake_music)
    monkeypatch.setattr(bot_module, "video_open_menu", fake_video)
    monkeypatch.setattr(bot_module, "dialog_open_menu", fake_dialog)

    async def fake_answer():
        return None

    tests = [
        ("profile", bot_module._PROFILE_MSG_ID_KEY, 101),
        ("kb", bot_module._KB_MSG_ID_KEY, 102),
        ("photo", bot_module._PHOTO_MSG_ID_KEY, 103),
        ("music", bot_module._MUSIC_MSG_ID_KEY, 104),
        ("video", bot_module._VIDEO_MSG_ID_KEY, 105),
        ("dialog", bot_module._DIALOG_MSG_ID_KEY, 106),
    ]

    async def scenario():
        for item, key, mid in tests:
            calls.clear()
            ctx.chat_data.clear()
            query = SimpleNamespace(
                data=f"menu:{item}",
                message=SimpleNamespace(
                    chat=SimpleNamespace(id=500 + len(item)),
                    chat_id=500 + len(item),
                ),
                from_user=SimpleNamespace(id=900 + len(item)),
                answer=fake_answer,
            )
            update = SimpleNamespace(
                callback_query=query,
                effective_chat=query.message.chat,
                effective_user=query.from_user,
            )

            await bot_module.handle_main_menu_callback(update, ctx)

            assert calls and calls[0][0] == item
            assert calls[0][2]["suppress_nav"] is True
            assert ctx.chat_data.get(key) == mid
            assert ctx.chat_data.get("nav_in_progress") is False

    asyncio.run(scenario())


def test_menu_open_no_duplicates(monkeypatch):
    ctx = SimpleNamespace(chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))

    fallback_values: list[object] = []

    async def fake_open_root(context, chat_id, *, suppress_nav=False, fallback_message_id=None):
        fallback_values.append(fallback_message_id)
        result = fallback_message_id if fallback_message_id is not None else 777
        if isinstance(context.chat_data, dict):
            context.chat_data[bot_module._KB_MSG_ID_KEY] = result
        return result

    monkeypatch.setattr(kb_module, "open_root", fake_open_root)

    async def fake_answer():
        return None

    query = SimpleNamespace(
        data="kb_open",
        message=SimpleNamespace(chat=SimpleNamespace(id=42), chat_id=42, message_id=555),
        from_user=SimpleNamespace(id=321),
        answer=fake_answer,
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=query.message.chat,
        effective_user=query.from_user,
    )

    async def scenario():
        await kb_module.kb_open_handler(update, ctx)
        assert fallback_values == [555]
        assert ctx.chat_data.get(bot_module._KB_MSG_ID_KEY) == 555

        await kb_module.kb_open_handler(update, ctx)
        assert fallback_values == [555, 555]
        assert ctx.chat_data.get(bot_module._KB_MSG_ID_KEY) == 555

    asyncio.run(scenario())


def test_menu_suppresses_dialog_notice(monkeypatch):
    bot = FakeBot()
    ctx = SimpleNamespace(bot=bot, chat_data={}, user_data={}, application=SimpleNamespace(bot_data={}))

    async def fake_photo(context, chat_id, *, suppress_nav, fallback_message_id=None):
        assert context.chat_data.get("nav_in_progress") is True
        return 888

    async def fake_answer():
        return None

    async def fake_ensure(update):
        return None

    monkeypatch.setattr(bot_module, "photo_open_menu", fake_photo)
    monkeypatch.setattr(bot_module, "ensure_user_record", fake_ensure)

    query = SimpleNamespace(
        data="menu:photo",
        message=SimpleNamespace(chat=SimpleNamespace(id=77), chat_id=77),
        from_user=SimpleNamespace(id=55),
        answer=fake_answer,
    )
    update = SimpleNamespace(callback_query=query, effective_chat=query.message.chat, effective_user=query.from_user)

    async def scenario():
        await bot_module.handle_main_menu_callback(update, ctx)

        assert ctx.chat_data.get(bot_module._PHOTO_MSG_ID_KEY) == 888
        assert ctx.chat_data.get("nav_in_progress") is False

        text_message = SimpleNamespace(text="привет", chat_id=77, chat=SimpleNamespace(id=77))
        text_update = SimpleNamespace(message=text_message, effective_message=text_message, effective_user=None)

        await bot_module.on_text(text_update, ctx)

        assert not bot.sent

    asyncio.run(scenario())
