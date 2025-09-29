import os
import sys

import pytest
import requests_mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suno.service import SunoError, suno_generate


def test_generate_happy():
    with requests_mock.Mocker() as mocker:
        mocker.post(
            "https://api.kie.ai/api/v1/generate",
            json={"code": 200, "data": {"taskId": "t-123"}},
        )
        task_id = suno_generate(
            prompt="lofi beat",
            customMode=True,
            instrumental=True,
            model="V5",
        )
        assert task_id == "t-123"


def test_ip_whitelist_error():
    with requests_mock.Mocker() as mocker:
        mocker.post(
            "https://api.kie.ai/api/v1/generate",
            json={"code": 401, "msg": "Illegal IP, please set up a whitelist"},
            status_code=200,
        )
        with pytest.raises(SunoError) as exc:
            suno_generate(prompt="x", customMode=True, instrumental=False, model="V5")
        assert "Illegal IP" in str(exc.value)
