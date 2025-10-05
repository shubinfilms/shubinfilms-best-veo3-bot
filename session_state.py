from dataclasses import dataclass
from enum import Enum

from telegram.ext import ContextTypes


class ChatMode(str, Enum):
    OFF = "off"
    REGULAR = "regular"


@dataclass
class Session:
    chat_mode: ChatMode = ChatMode.OFF


_SESSION_KEY = "__session__"


def get_session(ctx: ContextTypes.DEFAULT_TYPE) -> Session:
    """Return the per-user session object stored in the Telegram context."""

    user_data = getattr(ctx, "user_data", None)
    if not isinstance(user_data, dict):
        session = Session()
        try:
            setattr(ctx, "user_data", {_SESSION_KEY: session})
        except Exception:
            return session
        return session

    session_obj = user_data.get(_SESSION_KEY)
    if isinstance(session_obj, Session):
        return session_obj

    session = Session()
    user_data[_SESSION_KEY] = session
    return session


def enable_regular_chat(ctx: ContextTypes.DEFAULT_TYPE) -> Session:
    session = get_session(ctx)
    session.chat_mode = ChatMode.REGULAR
    return session


def disable_chat(ctx: ContextTypes.DEFAULT_TYPE) -> Session:
    session = get_session(ctx)
    session.chat_mode = ChatMode.OFF
    return session


def is_chat_enabled(ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    return get_session(ctx).chat_mode == ChatMode.REGULAR
