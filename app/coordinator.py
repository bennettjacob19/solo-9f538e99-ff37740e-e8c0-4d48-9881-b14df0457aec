"""Coordinator: FastAPI web app that orchestrates the two training workers.

Responsibilities
----------------
* Validate the uploaded CSV and deterministically split it into two
  mutually-exclusive shards, one per independent worker process.
* Drive round-by-round synchronous training: ask both workers for their shard
  gradients, combine them into the *full-data average gradient*, ask both
  workers to apply the identical momentum step and durably persist it, and
  only then atomically confirm the round locally.
* Persist all progress with atomic writes so a container restart resumes from
  the last confirmed round (redo of an unconfirmed round is allowed; skipping
  rounds or double-applying a confirmed update is impossible).
* Fence every round message with ``task_id`` + ``gen`` so late messages from a
  previous training group can never advance a new group.
* Keep a downloadable weight artifact that only ever reflects confirmed
  rounds; a failed save leaves the previous complete artifact in place.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import lrmath
from storage import (atomic_write_json, exists,
                     if_exists_read_json, read_json)

DATA_DIR = os.environ.get("COORD_DATA", "/data")
JOB_PATH = os.path.join(DATA_DIR, "job.json")
STATUS_PATH = os.path.join(DATA_DIR, "status.json")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
LATEST_WEIGHTS = os.path.join(WEIGHTS_DIR, "weights_latest.json")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

WORKERS = {
    "worker1": os.environ.get("WORKER1_URL", "http://worker1:8000"),
    "worker2": os.environ.get("WORKER2_URL", "http://worker2:8000"),
}
HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_ROWS, MIN_ROWS = 200, 2
N_FEATURES = 2


class Ctx:
    """In-memory runtime state, always rebuildable from durable files."""

    def __init__(self) -> None:
        self.job: Optional[Dict[str, Any]] = None
        self.status: Optional[Dict[str, Any]] = None
        self.task: Optional[asyncio.Task] = None
        self.client: Optional[httpx.AsyncClient] = None
        self.worker_live: Dict[str, bool] = {k: False for k in WORKERS}


CTX = Ctx()


# ============================================================ small utilities
def now_ms() -> int:
    return int(time.time() * 1000)


def httpc() -> httpx.AsyncClient:
    assert CTX.client is not None
    return CTX.client


def save_status(status: Dict[str, Any]) -> Dict[str, Any]:
    atomic_write_json(STATUS_PATH, status)
    CTX.status = status
    return status


def require_job() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if CTX.job is None or CTX.status is None:
        raise HTTPException(404, "no training task yet; upload a CSV first")
    return CTX.job, CTX.status


def split_indices(n: int) -> Tuple[List[int], List[int]]:
    """Deterministic, mutually-exclusive contiguous shard assignment."""
    cut = (n + 1) // 2  # worker1 gets the extra row when n is odd
    return list(range(0, cut)), list(range(cut, n))


# ============================================================ CSV validation
def parse_csv(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, f"CSV must be UTF-8 encoded: {exc}")
    reader = list(csv.reader(io.StringIO(text)))
    reader = [r for r in reader if any(c.strip() for c in r)]  # drop blanks
    if len(reader) < MIN_ROWS:
        raise HTTPException(400, "CSV needs 2-200 data rows (plus an optional "
                                 "header row)")

    header, data_rows = reader[0], reader[1:]
    # Detect an optional header row: if the first row has a non-numeric cell,
    # treat it as the header; otherwise every row is data (no header).
    def _row_is_numeric(row: List[str]) -> bool:
        if len(row) != N_FEATURES + 1:
            return False
        try:
            float(row[0]), float(row[1])
        except ValueError:
            return False
        return row[2].strip() in ("0", "1")

    first_is_header = not _row_is_numeric(header)
    rows = data_rows if first_is_header else reader

    if not (MIN_ROWS <= len(rows) <= MAX_ROWS):
        raise HTTPException(400, f"row count must be in [{MIN_ROWS},{MAX_ROWS}],"
                                 f" got {len(rows)}")
    xs: List[List[float]] = []
    ys: List[float] = []
    for i, row in enumerate(rows):
        if len(row) != N_FEATURES + 1:
            raise HTTPException(400, f"row {i+1}: expected 3 columns "
                                     f"(x1,x2,label), got {len(row)}")
        try:
            x1, x2 = float(row[0]), float(row[1])
        except ValueError:
            raise HTTPException(400, f"row {i+1}: non-numeric feature value")
        label = row[2].strip()
        if label not in ("0", "1"):
            raise HTTPException(400, f"row {i+1}: label must be 0 or 1")
        if not (np.isfinite(x1) and np.isfinite(x2)):
            raise HTTPException(400, f"row {i+1}: feature must be finite")
        xs.append([x1, x2])
        ys.append(float(label))
    if len(set(ys)) < 2:
        raise HTTPException(400, "labels must contain both classes 0 and 1")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


# ============================================================ weight artifacts
def write_weights_artifact(round_: int, w: List[float], v: List[float],
                           loss: float) -> None:
    """Publish weights for a confirmed round.

    Writes a new immutable file and atomically swaps the 'latest' pointer.
    If anything fails, the previous complete artifact stays untouched because
    the swap is the very last step and itself atomic.
    """
    payload = {
        "round": round_, "confirmed_round": round_, "loss": loss,
        "weights": {"bias": w[0], "w1": w[1], "w2": w[2]},
        "momentum": {"v_bias": v[0], "v1": v[1], "v2": v[2]},
        "saved_at": now_ms(),
    }
    named = os.path.join(WEIGHTS_DIR, f"weights_round_{round_:04d}.json")
    atomic_write_json(named, payload)
    atomic_write_json(LATEST_WEIGHTS, payload)


# ============================================================ worker messaging
async def worker_call(worker_id: str, path: str, payload: Dict[str, Any],
                      tolerate: bool = False) -> Dict[str, Any]:
    url = WORKERS[worker_id] + path
    try:
        r = await httpc().post(url, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            raise HTTPException(r.status_code,
                                f"{worker_id} {path} -> {r.status_code}: "
                                f"{r.text[:200]}")
        CTX.worker_live[worker_id] = True
        return r.json()
    except (httpx.HTTPError, HTTPException) as exc:
        CTX.worker_live[worker_id] = False
        if tolerate:
            return {"_error": str(exc)}
        raise


async def worker_health(worker_id: str) -> Dict[str, Any]:
    try:
        r = await httpc().get(WORKERS[worker_id] + "/health",
                              timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        CTX.worker_live[worker_id] = True
        return r.json()
    except Exception as exc:  # noqa: BLE001 - surfaced as status field
        CTX.worker_live[worker_id] = False
        return {"worker_id": worker_id, "alive": False, "error": str(exc)}


# ============================================================ training loop
async def run_training() -> None:
    """Round machine. A failure at any point pauses the whole group; progress
    never advances past the last durably confirmed round."""
    job, status = require_job()
    st = dict(status)
    st["status"] = "running"
    st["last_event"] = "training loop started"
    st["running_round"] = None
    save_status(st)

    x = np.asarray(job["x"], dtype=np.float64)
    y = np.asarray(job["y"], dtype=np.float64)
    x_aug = lrmath.augment(x)
    total_rounds = int(job["rounds"])
    try:
        while st["confirmed_round"] < total_rounds:
            rnd = st["confirmed_round"] + 1
            st["status"] = "running"
            st["running_round"] = rnd
            st["last_event"] = f"round {rnd}: requesting shard gradients"
            save_status(st)

            gen = int(st["gen"])
            payload = {"task_id": job["task_id"], "gen": gen, "round": rnd}
            g1, g2 = await asyncio.gather(
                worker_call("worker1", "/gradient", payload),
                worker_call("worker2", "/gradient", payload),
            )
            # Defensive: both gradients must be computed at the same
            # confirmed round within the same fenced generation.
            for g in (g1, g2):
                if g.get("gen") != gen or g.get("from_round") != rnd - 1:
                    raise RuntimeError(f"fence/round mismatch in gradient: {g}")

            n1, n2 = int(g1["n"]), int(g2["n"])
            if n1 + n2 != len(y):
                raise RuntimeError("shard sizes do not cover the full dataset")
            # Concatenate residuals in canonical dataset row order (worker1's
            # shard precedes worker2's), then take ONE full-data gradient:
            # identical summation order as uninterrupted single-process
            # training, therefore bit-identical even after fault recovery.
            residuals = np.asarray(
                list(g1["residuals"]) + list(g2["residuals"]),
                dtype=np.float64)
            if residuals.shape[0] != len(y):
                raise RuntimeError("residual count mismatch")
            full_grad = lrmath.full_average_gradient(x_aug, residuals)
            st["last_event"] = (f"round {rnd}: rebuilt full-data average "
                                f"gradient over {n1+n2} samples")
            save_status(st)

            commit_payload = {"task_id": job["task_id"], "gen": gen,
                              "round": rnd,
                              "grad": lrmath.to_list(full_grad)}
            c1, c2 = await asyncio.gather(
                worker_call("worker1", "/commit", commit_payload),
                worker_call("worker2", "/commit", commit_payload),
            )
            for c in (c1, c2):
                if c.get("gen") != gen or c.get("round") != rnd:
                    raise RuntimeError(f"fence/round mismatch in commit: {c}")
            # Both workers persisted identical w,v for this round. Verify the
            # coordinator's own math matches what the workers applied.
            w_exp, v_exp = lrmath.momentum_update(
                st["w"], st["v"], full_grad)
            if not (np.allclose(c1["w"], c2["w"], atol=0.0, rtol=0.0)
                    and np.allclose(c1["v"], c2["v"], atol=0.0, rtol=0.0)
                    and np.array_equal(np.asarray(c1["w"]), w_exp)):
                raise RuntimeError("worker state divergence detected")

            loss = lrmath.bce_loss(x_aug, y, w_exp)
            st["confirmed_round"] = rnd
            st["w"] = lrmath.to_list(w_exp)
            st["v"] = lrmath.to_list(v_exp)
            st["loss_history"].append({"round": rnd, "loss": loss,
                                       "at": now_ms()})
            st["running_round"] = None
            st["last_event"] = f"round {rnd}: confirmed (both sides persisted)"
            save_status(st)                     # durable confirmation first
            write_weights_artifact(rnd, st["w"], st["v"], loss)  # then export

        st["status"] = "completed"
        st["running_round"] = None
        st["last_event"] = f"all {total_rounds} rounds confirmed"
        save_status(st)
    except Exception as exc:  # noqa: BLE001 - any worker failure pauses group
        st["status"] = "paused"
        st["running_round"] = None
        st["last_error"] = f"{type(exc).__name__}: {exc}"
        st["last_event"] = ("group paused: round "
                            f"{st['confirmed_round'] + 1} not confirmed; "
                            "restart the dead worker and press Resume")
        save_status(st)


def ensure_not_busy() -> None:
    if CTX.status and CTX.status["status"] == "running":
        raise HTTPException(409, "a round is currently in progress")


# ============================================================ API: schemas
class DelayIn(BaseModel):
    seconds: float = Field(..., ge=0, le=120)


# ============================================================ application
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    CTX.client = httpx.AsyncClient()
    CTX.job = if_exists_read_json(JOB_PATH)
    CTX.status = if_exists_read_json(STATUS_PATH)
    if CTX.status is not None and CTX.status["status"] == "running":
        # Coordinator died mid-round: nothing after the last confirmed round
        # may count. Mark paused; the next resume re-prepares both workers.
        CTX.status["status"] = "paused"
        CTX.status["running_round"] = None
        CTX.status["last_event"] = (
            "coordinator restarted while a round was in flight; "
            "unconfirmed round will be redone")
        save_status(CTX.status)
    yield
    if CTX.task:
        CTX.task.cancel()
    await CTX.client.aclose()


app = FastAPI(title="Fault-tolerant logistic regression trainer",
              lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
async def status() -> JSONResponse:
    health = await asyncio.gather(*(worker_health(w) for w in WORKERS))
    out: Dict[str, Any] = {"workers": {h["worker_id"]: h for h in health}}
    if CTX.status:
        out.update(CTX.status)
    out["worker_live"] = CTX.worker_live
    out["weights_available"] = exists(LATEST_WEIGHTS)
    return JSONResponse(out)


@app.post("/api/job")
async def create_job(file: UploadFile = File(...),
                     rounds: int = 10) -> JSONResponse:
    if not (1 <= rounds <= 100):
        raise HTTPException(400, "rounds must be in [1,100]")
    if CTX.status and CTX.status["status"] not in ("completed",):
        raise HTTPException(409, "a task already exists; reset it first")
    raw = await file.read()
    x, y = parse_csv(raw)
    n = len(y)
    idx1, idx2 = split_indices(n)

    task_id = uuid.uuid4().hex
    rows_all = [{"x": [float(x[i, 0]), float(x[i, 1])], "y": float(y[i])}
                for i in range(n)]
    shards = {
        "worker1": [{"x": r["x"], "y": r["y"]}
                    for r in (rows_all[i] for i in idx1)],
        "worker2": [{"x": r["x"], "y": r["y"]}
                    for r in (rows_all[i] for i in idx2)],
    }
    job = {
        "task_id": task_id, "rounds": int(rounds), "n": n,
        "x": [[float(a), float(b)] for a, b in x],
        "y": [float(v) for v in y],
        "shards": {"worker1": idx1, "worker2": idx2},
        "created_at": now_ms(),
        "hyper": {"lr": lrmath.LEARNING_RATE,
                  "momentum": lrmath.MOMENTUM},
    }
    # Durable first, before touching any worker.
    atomic_write_json(JOB_PATH, job)
    CTX.job = job  # keep in-memory context in sync with the durable file
    zero = lrmath.to_list(lrmath.zero_buffers())
    status_doc = {
        "task_id": task_id, "gen": 0,
        "status": "idle", "confirmed_round": 0,
        "total_rounds": int(rounds),
        "w": zero, "v": zero,
        "loss_history": [{"round": 0,
                          "loss": lrmath.bce_loss(lrmath.augment(x), y, zero),
                          "at": now_ms()}],
        "running_round": None, "last_event": "task created",
        "last_error": None, "created_at": now_ms(),
    }
    save_status(status_doc)
    write_weights_artifact(0, zero, zero, status_doc["loss_history"][0]["loss"])

    # Push the mutually-exclusive shards to the two independent processes.
    results = await asyncio.gather(*(
        worker_call(wid, "/load",
                    {"task_id": task_id, "worker_id": wid,
                     "rows": shards[wid]}, tolerate=True)
        for wid in WORKERS))
    for wid, res in zip(WORKERS, results):
        if "_error" in res:
            st = dict(CTX.status)
            st["status"] = "paused"
            st["last_error"] = f"shard load to {wid} failed: {res['_error']}"
            st["last_event"] = "task created but a worker is unreachable"
            save_status(st)
            return JSONResponse(st, status_code=202)
    st = dict(CTX.status)
    st["last_event"] = "shards loaded on both workers; press Start"
    save_status(st)
    return JSONResponse(st)


async def _prepare_workers(gen: int) -> None:
    """Install a fresh fencing generation and align both workers to the
    coordinator's last confirmed round (durable rollback/catch-up)."""
    job, st = require_job()
    payload = {"task_id": job["task_id"], "gen": int(gen),
               "round": int(st["confirmed_round"]),
               "w": st["w"], "v": st["v"]}
    p1, p2 = await asyncio.gather(
        worker_call("worker1", "/prepare", payload),
        worker_call("worker2", "/prepare", payload),
    )
    for p in (p1, p2):
        if p.get("gen") != gen or p.get("round") != st["confirmed_round"]:
            raise RuntimeError(f"prepare fence/round mismatch: {p}")


@app.post("/api/start")
async def start() -> JSONResponse:
    job, st = require_job()
    ensure_not_busy()
    if st["status"] not in ("idle", "paused"):
        raise HTTPException(409, f"cannot start from status {st['status']}")
    new_gen = int(st["gen"]) + 1  # every (re)start opens a new training group
    st = dict(st)
    st["gen"] = new_gen
    st["status"] = "preparing"
    st["last_error"] = None
    st["last_event"] = f"opening training generation {new_gen}"
    save_status(st)
    await _prepare_workers(new_gen)
    CTX.task = asyncio.create_task(run_training())
    return JSONResponse(save_status({**st, "last_event":
                                     f"generation {new_gen} prepared; running"}))


@app.post("/api/resume")
async def resume() -> JSONResponse:
    # Resume == guarded restart: same entry point, new generation, roll both
    # workers to the confirmed round, then continue. No round is skipped.
    return await start()


@app.post("/api/pause")
async def pause() -> JSONResponse:
    job, st = require_job()
    if st["status"] != "running":
        raise HTTPException(409, "not running")
    if CTX.task:
        CTX.task.cancel()
        CTX.task = None
    st = dict(st)
    st["status"] = "paused"
    st["running_round"] = None
    st["last_event"] = "paused between rounds; resume redoes nothing confirmed"
    save_status(st)
    return JSONResponse(st)


@app.post("/api/reset")
async def reset() -> JSONResponse:
    ensure_not_busy()
    for p in (JOB_PATH, STATUS_PATH):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    CTX.job = None
    CTX.status = None
    return JSONResponse({"ok": True, "event": "task cleared"})


@app.post("/api/workers/{worker_id}/delay")
async def set_worker_delay(worker_id: str, body: DelayIn) -> JSONResponse:
    if worker_id not in WORKERS:
        raise HTTPException(404, "unknown worker")
    res = await worker_call(worker_id, "/delay",
                            {"seconds": body.seconds}, tolerate=True)
    return JSONResponse({"worker_id": worker_id, "response": res})


@app.post("/api/workers/delay")
async def set_both_delay(body: DelayIn) -> JSONResponse:
    results = await asyncio.gather(*(
        worker_call(w, "/delay", {"seconds": body.seconds}, tolerate=True)
        for w in WORKERS))
    return JSONResponse(dict(zip(WORKERS, results)))


@app.post("/api/workers/{worker_id}/kill")
async def kill_worker(worker_id: str) -> JSONResponse:
    if worker_id not in WORKERS:
        raise HTTPException(404, "unknown worker")
    # Fire-and-forget SIGKILL. The in-flight round (if any) fails and the group
    # pauses; the entrypoint supervisor respawns the process fresh.
    asyncio.create_task(
        httpc().post(WORKERS[worker_id] + "/kill", timeout=2.0))
    CTX.worker_live[worker_id] = False
    return JSONResponse({"worker_id": worker_id,
                         "event": "SIGKILL sent; group will pause"})


@app.get("/api/weights")
async def download_weights() -> FileResponse:
    if not exists(LATEST_WEIGHTS):
        raise HTTPException(404, "no confirmed weights yet")
    doc = read_json(LATEST_WEIGHTS)
    # The artifact only ever contains a fully confirmed round.
    if CTX.status and doc["confirmed_round"] > CTX.status["confirmed_round"]:
        raise HTTPException(409, "weight artifact ahead of confirmed state")
    return FileResponse(
        LATEST_WEIGHTS, media_type="application/json",
        filename=f"weights_round_{doc['confirmed_round']:04d}.json")


@app.get("/api/reference")
async def reference() -> JSONResponse:
    """Single-process full-data reference trajectory (for verification)."""
    job, _ = require_job()
    x = np.asarray(job["x"], dtype=np.float64)
    y = np.asarray(job["y"], dtype=np.float64)
    ws, vs, losses = lrmath.reference_train(x, y, int(job["rounds"]))
    return JSONResponse({
        "rounds": int(job["rounds"]),
        "weights": [lrmath.to_list(w) for w in ws],
        "momentum": [lrmath.to_list(v) for v in vs],
        "loss": losses,
    })
