import pytest

from utils import input_state
from utils.input_state import WaitInputState, WaitKind, clear_wait_state


class DummyRedis:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def set(self, key: str, payload: str, ex: int) -> None:
        self.calls.append((key, int(ex)))

    def delete(self, key: str) -> None:  # pragma: no cover - no-op
        return


@pytest.mark.parametrize("ttl_seconds, expected", [(45, 45), (1, 1), (0, 1)])
def test_set_wait_state_uses_ttl_seconds(monkeypatch, ttl_seconds, expected) -> None:
    dummy = DummyRedis()
    monkeypatch.setattr(input_state, "_redis", dummy, raising=False)
    monkeypatch.setattr(input_state, "_memory_store", {}, raising=False)
    wait_state = WaitInputState(kind=WaitKind.VEO_PROMPT, card_msg_id=1, chat_id=1, meta={})

    input_state.set_wait_state(101, wait_state, ttl_seconds=ttl_seconds)

    assert dummy.calls, "redis.set should be called"
    key, ttl = dummy.calls[0]
    assert key.endswith(":wait-input:101")
    assert ttl == expected
    clear_wait_state(101)
