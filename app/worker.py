"""Training worker: one independent CPU process holding one data shard.

Each worker is deployed as its own container with its own persistent volume
mounted at ``WORKER_DATA`` (default ``/data``). Durable files:

* ``shard.json``  - the exclusive sample shard assigned to this worker
* ``state.json``  - highest persisted ``{round, weights, momentum}``

Fencing
-------
Every round-protocol message carries ``task_id`` + ``gen`` (training-group
generation). A worker only serves round traffic for the generation installed by
``/prepare``; messages from an older/unknown generation are rejected with 409,
so a late message from a dead training group can never advance a new group.

Crash semantics
---------------
``/commit`` applies one momentum step and *atomically* persists the new state
before acknowledging. A kill at any instant leaves either the previous or the
new complete state on disk. After restart the worker reloads that state but
stays generation-neutral until the coordinator calls ``/prepare``, which rolls
it to the last *confirmed* round (possibly discarding an unconfirmed commit).
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import lrmath
from storage import atomic_write_json, if_exists_read_json

WORKER_ID = os.environ.get("WORKER_ID", "worker")
DATA_DIR = os.environ.get("WORKER_DATA", "/data")
SHARD_PATH = os.path.join(DATA_DIR, "shard.json")
STATE_PATH = os.path.join(DATA_DIR, "state.json")

app = FastAPI(title=f"lr-worker-{WORKER_ID}")


class WorkerState:
    def __init__(self) -> None:
        self.task_id: Optional[str] = None
        self.gen: Optional[int] = None  # active fencing generation
        self.round: int = 0
        self.w: np.ndarray = lrmath.zero_buffers()
        self.v: np.ndarray = lrmath.zero_buffers()
        self.x_aug: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.delay_seconds: float = 0.0
        self.started_at: float = time.time()
        self._load_durable()

    # ------------------------------------------------------------------ duraload
    def _load_durable(self) -> None:
        shard = if_exists_read_json(SHARD_PATH)
        if shard:
            self.task_id = shard["task_id"]
            self._install_shard(shard["rows"])
        state = if_exists_read_json(STATE_PATH)
        if state:
            # A persisted state always belongs to the shard's task.
            self.task_id = state["task_id"]
            self.round = int(state["round"])
            self.w = np.asarray(state["w"], dtype=np.float64)
            self.v = np.asarray(state["v"], dtype=np.float64)
        # gen intentionally stays None: a restarted worker must be re-prepared.

    def _install_shard(self, rows: list) -> None:
        xs = np.asarray([r["x"] for r in rows], dtype=np.float64)
        self.x_aug = lrmath.augment(xs)
        self.y = np.asarray([r["y"] for r in rows], dtype=np.float64)

    # ------------------------------------------------------------------ guards
    def check_fence(self, task_id: str, gen: int) -> None:
        if self.task_id is None or self.x_aug is None:
            raise HTTPException(status_code=410,
                                detail="worker holds no task shard")
        if task_id != self.task_id:
            raise HTTPException(status_code=410,
                                detail="task id mismatch (stale task)")
        if self.gen is None or gen != self.gen:
            raise HTTPException(
                status_code=409,
                detail=f"generation fence: active={self.gen} got={gen}")

    async def inject_delay(self) -> None:
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

    # ------------------------------------------------------------------ actions
    def load_task(self, task_id: str, worker_id: str, rows: list) -> None:
        if worker_id != WORKER_ID:
            raise HTTPException(status_code=400,
                                detail=f"this worker is {WORKER_ID}")
        # New task (or re-creation): wipe all previous durable state.
        self.task_id = task_id
        self._install_shard(rows)
        self.round = 0
        self.w = lrmath.zero_buffers()
        self.v = lrmath.zero_buffers()
        self.gen = None
        atomic_write_json(SHARD_PATH, {
            "task_id": task_id, "worker_id": WORKER_ID,
            "rows": rows, "n": len(rows),
        })
        atomic_write_json(STATE_PATH, {
            "task_id": task_id, "gen": None, "round": 0,
            "w": lrmath.to_list(self.w), "v": lrmath.to_list(self.v),
        })

    def prepare(self, task_id: str, gen: int, round_: int,
                w: list, v: list) -> Dict[str, Any]:
        if self.task_id != task_id:
            raise HTTPException(status_code=410, detail="task id mismatch")
        w_arr = np.asarray(w, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        # Install the new fencing generation *before* touching state.
        self.gen = int(gen)
        # Align (roll back or catch up) to the coordinator's confirmed round.
        changed = (round_ != self.round
                   or not np.array_equal(w_arr, self.w)
                   or not np.array_equal(v_arr, self.v))
        if changed:
            self.round = int(round_)
            self.w = w_arr
            self.v = v_arr
        # Always persist: a new generation must be durable even when the
        # confirmed round and buffers themselves did not change.
        atomic_write_json(STATE_PATH, {
            "task_id": task_id, "gen": int(gen),
            "round": self.round,
            "w": lrmath.to_list(self.w),
            "v": lrmath.to_list(self.v),
        })
        return {"worker_id": WORKER_ID, "gen": self.gen,
                "round": self.round, "durable_round": self.round}

    def gradient(self, task_id: str, gen: int,
                 round_: int) -> Dict[str, Any]:
        self.check_fence(task_id, gen)
        if round_ != self.round + 1:
            raise HTTPException(
                status_code=409,
                detail=f"round out of order: at {self.round}, got {round_}")
        # Return per-sample residuals rather than a pre-summed shard gradient:
        # the coordinator rebuilds the full-data average gradient in canonical
        # row order, matching the single-process reference bit-for-bit.
        residuals, shard_loss = lrmath.shard_residuals(
            self.x_aug, self.y, self.w)
        return {
            "worker_id": WORKER_ID, "task_id": task_id, "gen": gen,
            "round": round_, "n": int(self.x_aug.shape[0]),
            "residuals": lrmath.to_list(residuals),
            "shard_loss": shard_loss,
            "from_round": self.round,
        }

    def commit(self, task_id: str, gen: int, round_: int,
               grad: list) -> Dict[str, Any]:
        self.check_fence(task_id, gen)
        # Idempotent at-least-once commit: a retried commit for an already
        # persisted round returns the stored state instead of applying twice.
        if round_ == self.round:
            return {"worker_id": WORKER_ID, "gen": gen, "round": self.round,
                    "w": lrmath.to_list(self.w),
                    "v": lrmath.to_list(self.v), "dedup": True}
        if round_ != self.round + 1:
            raise HTTPException(
                status_code=409,
                detail=f"round out of order: at {self.round}, got {round_}")
        new_w, new_v = lrmath.momentum_update(self.w, self.v, grad)
        # Persist BEFORE acknowledging: the coordinator only confirms a round
        # after both workers have durably stored exactly this round.
        atomic_write_json(STATE_PATH, {
            "task_id": task_id, "gen": int(gen),
            "round": int(round_),
            "w": lrmath.to_list(new_w),
            "v": lrmath.to_list(new_v),
        })
        self.round = int(round_)
        self.w = new_w
        self.v = new_v
        return {"worker_id": WORKER_ID, "gen": gen, "round": self.round,
                "w": lrmath.to_list(self.w),
                "v": lrmath.to_list(self.v), "dedup": False}


STATE = WorkerState()


# ---------------------------------------------------------------------- schemas
class LoadIn(BaseModel):
    task_id: str
    worker_id: str
    rows: list = Field(..., min_length=1)


class PrepareIn(BaseModel):
    task_id: str
    gen: int
    round: int
    w: list
    v: list


class GradIn(BaseModel):
    task_id: str
    gen: int
    round: int


class CommitIn(BaseModel):
    task_id: str
    gen: int
    round: int
    grad: list


class DelayIn(BaseModel):
    seconds: float = Field(..., ge=0, le=120)


# --------------------------------------------------------------------- routes
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "worker_id": WORKER_ID, "alive": True,
        "task_id": STATE.task_id, "gen": STATE.gen,
        "round": STATE.round, "durable_round": STATE.round,
        "delay_seconds": STATE.delay_seconds,
        "uptime": round(time.time() - STATE.started_at, 3),
        "pid": os.getpid(),
    }


@app.post("/load")
async def load(body: LoadIn) -> Dict[str, Any]:
    STATE.load_task(body.task_id, body.worker_id, body.rows)
    return {"worker_id": WORKER_ID, "task_id": STATE.task_id,
            "n": STATE.x_aug.shape[0], "round": STATE.round}


@app.post("/prepare")
async def prepare(body: PrepareIn) -> Dict[str, Any]:
    await STATE.inject_delay()
    return STATE.prepare(body.task_id, body.gen, body.round, body.w, body.v)


@app.post("/gradient")
async def gradient(body: GradIn) -> Dict[str, Any]:
    await STATE.inject_delay()
    return STATE.gradient(body.task_id, body.gen, body.round)


@app.post("/commit")
async def commit(body: CommitIn) -> Dict[str, Any]:
    await STATE.inject_delay()
    return STATE.commit(body.task_id, body.gen, body.round, body.grad)


@app.post("/delay")
async def set_delay(body: DelayIn) -> Dict[str, Any]:
    STATE.delay_seconds = float(body.seconds)
    return {"worker_id": WORKER_ID, "delay_seconds": STATE.delay_seconds}


@app.post("/kill")
async def kill() -> Dict[str, Any]:
    # Strong kill: SIGKILL on self, no cleanup handlers, like `kill -9`.
    # The container entrypoint supervisor respawns the process; the group
    # itself stays paused because the coordinator fences the fresh process.
    os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError("unreachable")  # pragma: no cover
