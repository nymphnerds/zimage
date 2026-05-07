from __future__ import annotations

from copy import deepcopy
from threading import Lock


_LOCK = Lock()
_STATE = {
    "status": "idle",
    "stage": "idle",
    "detail": "Idle",
    "model_id": None,
    "progress_current": None,
    "progress_total": None,
    "progress_percent": None,
    "last_output_path": None,
}


def update(**values):
    with _LOCK:
        _STATE.update(values)


def reset():
    update(
        status="idle",
        stage="idle",
        detail="Idle",
        model_id=_STATE.get("model_id"),
        progress_current=None,
        progress_total=None,
        progress_percent=None,
    )


def snapshot():
    with _LOCK:
        return deepcopy(_STATE)
