from __future__ import annotations
from typing import Any, Dict, List
import os
import json
import datetime as dt

from flask import Flask, request, jsonify, abort

try:
    from suno.store import SunoStore  # типовой наш слой хранения
except Exception:
    SunoStore = None  # веб будет работать даже если store недоступен

app = Flask(__name__)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": dt.datetime.utcnow().isoformat()}, 200


@app.post("/suno-callback")
def suno_callback():
    if not request.is_json:
        abort(400, "JSON body required")

    payload = request.get_json(force=True)
    code: int = payload.get("code", 0)
    msg: str = payload.get("msg", "")
    data: Dict[str, Any] = payload.get("data", {}) or {}
    cb_type: str = data.get("callbackType", "")
    task_id: str = data.get("task_id", "") or data.get("taskId", "")
    tracks: List[Dict[str, Any]] = data.get("data", []) or []

    app.logger.info(
        "[SUNO] callback code=%s type=%s task_id=%s msg=%s",
        code,
        cb_type,
        task_id,
        msg,
    )

    try:
        os.makedirs("/tmp/suno", exist_ok=True)
        stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        with open(f"/tmp/suno/{stamp}_{task_id or 'no-task'}.json", "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:  # pragma: no cover - best effort logging
        app.logger.warning("Failed to write temp file: %s", e)

    if SunoStore:
        try:
            store = SunoStore()
            store.save_callback(
                task_id=task_id,
                callback_type=cb_type,
                code=code,
                message=msg,
                payload=payload,
                tracks=tracks,
            )
        except Exception as e:  # pragma: no cover - best effort logging
            app.logger.error("Store save failed: %s", e)

    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
