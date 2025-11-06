import asyncio
import importlib
import sys


def test_get_redis_returns_memory_store_when_url_missing(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    sys.modules.pop("utils.redis_client", None)
    module = importlib.import_module("utils.redis_client")
    importlib.reload(module)

    store = module.get_redis()

    assert isinstance(store, module.InMemoryStore)

    async def _scenario() -> None:
        ok = await store.set("example", "value", ex=5)
        assert ok is True
        assert await store.exists("example") is True
        assert await store.get("example") == "value"
        removed = await store.delete("example")
        assert removed == 1
        assert await store.get("example") is None

    asyncio.run(_scenario())
