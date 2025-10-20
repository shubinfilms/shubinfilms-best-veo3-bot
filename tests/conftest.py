from __future__ import annotations

from typing import Generator

import os
import sys
import types
import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("LEDGER_BACKEND", "memory")
os.environ.setdefault("TELEGRAM_TOKEN", "dummy-token")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("KIE_API_KEY", "dummy-key")
os.environ.setdefault("SUNO_CALLBACK_URL", "https://example.com/callback")
os.environ.setdefault("LOG_LEVEL", "ERROR")
os.environ.setdefault("LOG_JSON", "false")

if "sqlalchemy" not in sys.modules:
    sqlalchemy_stub = types.ModuleType("sqlalchemy")

    def _noop(*args, **kwargs):
        return None

    sqlalchemy_stub.create_engine = _noop  # type: ignore[attr-defined]
    sqlalchemy_stub.text = lambda value: value  # type: ignore[attr-defined]
    sys.modules["sqlalchemy"] = sqlalchemy_stub

    sqlalchemy_engine_stub = types.ModuleType("sqlalchemy.engine")

    from urllib.parse import parse_qsl, quote, urlparse, urlencode, urlunparse

    class _DummyURL:
        def __init__(self, raw: str):
            parsed = urlparse(str(raw))
            self.drivername = parsed.scheme or ""
            self.username = parsed.username
            self.password = parsed.password
            self.host = parsed.hostname or ""
            self.port = parsed.port
            self.database = parsed.path[1:] if parsed.path.startswith("/") else parsed.path
            self.query = dict(parse_qsl(parsed.query))
            self._raw = self._compose(hide_password=False)

        def _compose(self, hide_password: bool) -> str:
            netloc = ""
            if self.username:
                user = quote(self.username, safe="")
                netloc += user
                if self.password is not None:
                    password = "***" if hide_password else quote(self.password, safe="")
                    netloc += f":{password}"
                netloc += "@"
            netloc += self.host or ""
            if self.port is not None:
                netloc += f":{self.port}"
            path = f"/{self.database}" if self.database else ""
            query = urlencode(self.query)
            return urlunparse((self.drivername, netloc, path, "", query, ""))

        def set(self, **kwargs):
            if "drivername" in kwargs:
                self.drivername = kwargs["drivername"]
            if "username" in kwargs:
                self.username = kwargs["username"]
            if "password" in kwargs:
                self.password = kwargs["password"]
            if "host" in kwargs:
                self.host = kwargs["host"] or ""
            if "port" in kwargs:
                self.port = kwargs["port"]
            if "database" in kwargs:
                self.database = kwargs["database"] or ""
            if "query" in kwargs and kwargs["query"] is not None:
                self.query = dict(kwargs["query"])
            self._raw = self._compose(hide_password=False)
            return self

        def __str__(self) -> str:
            return self._raw

        def render_as_string(self, hide_password: bool = True) -> str:
            return self._compose(hide_password=hide_password)

    sqlalchemy_engine_stub.Engine = object  # type: ignore[attr-defined]
    sqlalchemy_engine_stub.URL = _DummyURL  # type: ignore[attr-defined]
    sqlalchemy_engine_stub.make_url = lambda value: _DummyURL(str(value))  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.engine"] = sqlalchemy_engine_stub

    sqlalchemy_exc_stub = types.ModuleType("sqlalchemy.exc")

    class _SqlAlchemyStubError(Exception):
        pass

    sqlalchemy_exc_stub.OperationalError = _SqlAlchemyStubError  # type: ignore[attr-defined]
    sqlalchemy_exc_stub.SQLAlchemyError = _SqlAlchemyStubError  # type: ignore[attr-defined]
    sys.modules["sqlalchemy.exc"] = sqlalchemy_exc_stub

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_postgres_module = importlib.import_module("db.postgres")


class _DummyCursor:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, *args, **kwargs):
        return None

    async def fetchone(self):
        return (1,)


class _DummyConnection:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _DummyCursor()


class _DummyPool:
    def connection(self):
        return _DummyConnection()

    async def close(self):
        return None


async def _fake_create_pg_pool(dsn: str | None = None):
    return _DummyPool()


def _fake_configure_engine(dsn: str | None = None):
    return None


setattr(_postgres_module, "create_pg_pool", _fake_create_pg_pool)
setattr(_postgres_module, "configure_engine", _fake_configure_engine)


@pytest.fixture
def user_id() -> Generator[int, None, None]:
    yield 12345
