"""Crash-safe persistence helpers.

All state is written with the classic atomic-rename protocol:

1. write to a temporary file in the *same* directory,
2. ``flush()`` + ``fsync()`` the file,
3. ``os.replace`` onto the final name,
4. ``fsync`` the parent directory so the rename itself is durable.

A reader therefore only ever observes the previous complete file or the new
complete file, never a half-written one. This is what makes "save failure keeps
the previous complete round" hold even across a power loss / container kill.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(directory)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, separators=(",", ":"),
                                       sort_keys=True))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def exists(path: str) -> bool:
    return os.path.exists(path)


def if_exists_read_json(path: str) -> Any | None:
    if not os.path.exists(path):
        return None
    return read_json(path)
