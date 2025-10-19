from __future__ import annotations

from typing import Generator

import pytest


@pytest.fixture
def user_id() -> Generator[int, None, None]:
    yield 12345
