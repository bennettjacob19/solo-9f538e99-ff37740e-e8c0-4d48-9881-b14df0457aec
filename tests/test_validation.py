"""End-to-end tests for the independent validation set, sample-count loss
merge, early stopping and crash recovery *during* validation.

Reuses the real three-process cluster (one coordinator + two independent
worker processes with separate data directories) from ``test_integration``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import httpx
import numpy as np

from app import lrmath as lr
from test_integration import (_dataset_csv, _wait_status, cluster)  # noqa: F401


# --------------------------------------------------------------------- helpers
def _val_csv(tmp_path: Path, n: int = 24, seed: int = 77) -> Path:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    z = 1.2 * x[:, 0] - 0.6 * x[:, 1] + 0.1
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    p = tmp_path / "val.csv"
    lines = ["x1,x2,label"]
    for (a, b), c in zip(x, y):
        lines.append(f"{a:.10f},{b:.10f},{c}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _read_csv(path: Path):
    rows = list(csv.reader(open(path, encoding="utf-8")))[1:]
    xs = np.asarray([[float(r[0]), float(r[1])] for r in rows])
    ys = np.asarray([float(r[2]) for r in rows])
    return xs, ys


def _create_job(base, train_csv, val_csv=None, rounds=20,
                patience=3, min_delta=0.0):
    with open(train_csv, "rb") as f:
        files = {"file": ("data.csv", f, "text/csv")}
        if val_csv is not None:
            vf = open(val_csv, "rb")
            files["val_file"] = ("val.csv", vf, "text/csv")
        try:
            r = httpx.post(
                base + (f"/api/job?rounds={rounds}&patience={patience}"
                        f"&min_delta={min_delta}"),
                files=files, timeout=20)
        finally:
            if val_csv is not None:
                vf.close()
    r.raise_for_status()
    return r


def _start_and_wait(base, predicate, timeout=90):
    httpx.post(base + "/api/start", timeout=10).raise_for_status()
    return _wait_status(base, predicate, timeout=timeout)


# --------------------------------------------------------------------- tests
def test_validation_loss_merge_matches_full_set_and_training_stays_exact(
        cluster, tmp_path):
    base = cluster["coord"]
    val_path = _val_csv(tmp_path)
    vx, vy = _read_csv(val_path)
    vx_aug = lr.augment(vx)
    _create_job(base, _dataset_csv(tmp_path, n=40), val_path,
                rounds=12, patience=10, min_delta=0.0)
    st = _start_and_wait(base, lambda d: d["status"] == "completed")

    # One evaluation per confirmed round, no gaps and no double counting.
    val_rounds = [h["round"] for h in st["val_history"]]
    assert val_rounds == list(range(1, 13))
    assert st["val_evaluated_round"] == 12

    # Workers hold mutually-exclusive validation shards covering the whole set.
    h1 = httpx.get(
        f"http://127.0.0.1:{cluster['w1'].port}/health").json()
    h2 = httpx.get(
        f"http://127.0.0.1:{cluster['w2'].port}/health").json()
    assert h1["val_n"] + h2["val_n"] == len(vy)
    assert h1["val_n"] > 0 and h2["val_n"] > 0

    # The sample-count-merged validation loss equals the full validation-set
    # mean BCE evaluated on the bit-identical reference weight trajectory.
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for h in st["val_history"]:
        expected = lr.bce_loss(vx_aug, vy, ref["weights"][h["round"]])
        assert np.isclose(h["val_loss"], expected, rtol=1e-12, atol=1e-12)

    # Validation never feeds gradients: training trajectory stays exact.
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]

    # Best tracking invariants.
    losses = [h["val_loss"] for h in st["val_history"]]
    assert st["best_val_loss"] == min(losses)
    assert st["best_val_round"] == val_rounds[int(np.argmin(losses))]
    assert st["no_improve_rounds"] == 12 - st["best_val_round"]


def test_early_stopping_and_best_round_download(cluster, tmp_path):
    base = cluster["coord"]
    # A minimum improvement larger than any real per-round move means only
    # the first evaluated round can ever be "best"; with patience=2 training
    # must stop right after round 3.
    _create_job(base, _dataset_csv(tmp_path, n=40), _val_csv(tmp_path),
                rounds=30, patience=2, min_delta=1e6)
    st = _start_and_wait(base, lambda d: d["status"] == "early_stopped")

    assert st["confirmed_round"] == 3
    assert st["val_evaluated_round"] == 3
    assert st["best_val_round"] == 1
    assert st["no_improve_rounds"] == 2
    assert [h["round"] for h in st["val_history"]] == [1, 2, 3]

    # Best-round download: full weights AND momentum of round 1 only.
    doc = httpx.get(base + "/api/best_weights", timeout=5).json()
    assert doc["best_round"] == 1 and doc["kind"] == "best_validation"
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    assert doc["weights"]["w1"] == ref["weights"][1][1]
    assert doc["momentum"]["v1"] == ref["momentum"][1][1]
    cdir = cluster["paths"]["coord"] / "weights"
    assert (cdir / "weights_best_round_0001.json").exists()
    assert (cdir / "weights_best.json").exists()

    # The original confirmed-weights download is preserved and still points at
    # the last *confirmed* round (3), never at an unevaluated round.
    latest = httpx.get(base + "/api/weights", timeout=5).json()
    assert latest["confirmed_round"] == 3

    # Terminal state cannot be restarted without a reset.
    r = httpx.post(base + "/api/start", timeout=5)
    assert r.status_code == 409


def test_patience_min_delta_parameter_validation(cluster, tmp_path):
    base = cluster["coord"]
    train, val = _dataset_csv(tmp_path, n=20), _val_csv(tmp_path, n=10)
    for bad_p in (0, 11):
        with open(train, "rb") as f, open(val, "rb") as vf:
            r = httpx.post(
                base + f"/api/job?rounds=5&patience={bad_p}&min_delta=0",
                files={"file": ("d.csv", f, "text/csv"),
                       "val_file": ("v.csv", vf, "text/csv")})
        assert r.status_code == 400
    with open(train, "rb") as f, open(val, "rb") as vf:
        r = httpx.post(base + "/api/job?rounds=5&patience=2&min_delta=-0.01",
                       files={"file": ("d.csv", f, "text/csv"),
                              "val_file": ("v.csv", vf, "text/csv")})
    assert r.status_code == 400
    # Boundary values are accepted.
    r = _create_job(base, train, val, rounds=3, patience=1, min_delta=0.0)
    assert r.status_code in (200, 202)


def test_worker_kill_during_validation_resumes_without_double_counting(
        cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay",
               json={"seconds": 0.8}, timeout=5)
    _create_job(base, _dataset_csv(tmp_path, n=40), _val_csv(tmp_path),
                rounds=18, patience=10, min_delta=0.0)
    httpx.post(base + "/api/start", timeout=10).raise_for_status()
    _wait_status(base, lambda d: d["confirmed_round"] >= 3)

    # Wait for the validation phase to be visibly in flight, then hard-kill a
    # worker so the evaluation fails mid-way. Retry across rounds if needed.
    killed_round = None
    deadline = time.time() + 40
    while time.time() < deadline:
        d = httpx.get(base + "/api/status", timeout=5).json()
        if d["status"] == "paused":
            break
        stage = d.get("stage") or ""
        if stage.startswith("validating"):
            killed_round = d["confirmed_round"]
            cluster["w2"].hard_kill()
            break
        time.sleep(0.1)
    assert killed_round is not None, "never observed a validation phase"

    st = _wait_status(base, lambda d: d["status"] == "paused")
    # The round stayed confirmed but unevaluated: it must not be marked best.
    assert st["confirmed_round"] == killed_round
    assert (st["val_evaluated_round"] or 0) < killed_round
    assert st["best_val_round"] is None or st["best_val_round"] < killed_round

    # Recover: the pending validation is finished before any new round.
    cluster["w2"].start()
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=120)

    val_rounds = [h["round"] for h in st["val_history"]]
    # Exactly one evaluation per round: the interrupted round was not counted
    # twice, and none of the 18 rounds is missing.
    assert val_rounds == list(range(1, 19))
    assert st["val_evaluated_round"] == 18

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    vx, vy = _read_csv(_val_csv(tmp_path))
    vx_aug = lr.augment(vx)
    for h in st["val_history"]:
        expected = lr.bce_loss(vx_aug, vy, ref["weights"][h["round"]])
        assert np.isclose(h["val_loss"], expected, rtol=1e-12, atol=1e-12)


def test_coordinator_restart_during_validation_finishes_pending_round(
        cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay",
               json={"seconds": 0.8}, timeout=5)
    _create_job(base, _dataset_csv(tmp_path, n=40), _val_csv(tmp_path),
                rounds=10, patience=10)
    httpx.post(base + "/api/start", timeout=10).raise_for_status()
    _wait_status(base, lambda d: d["confirmed_round"] >= 2)

    killed_round = None
    deadline = time.time() + 40
    while time.time() < deadline:
        d = httpx.get(base + "/api/status", timeout=5).json()
        if (d.get("stage") or "").startswith("validating"):
            killed_round = d["confirmed_round"]
            cluster["coord_svc"].hard_kill()
            break
        time.sleep(0.1)
    assert killed_round is not None

    time.sleep(0.5)
    cluster["coord_svc"].start()
    st = httpx.get(base + "/api/status", timeout=10).json()
    assert st["status"] == "paused"
    assert st["confirmed_round"] == killed_round
    assert (st["val_evaluated_round"] or 0) < killed_round

    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=120)
    assert [h["round"] for h in st["val_history"]] == list(range(1, 11))
    assert st["val_evaluated_round"] == 10


def test_validation_csv_rules_and_no_validation_mode(cluster, tmp_path):
    base = cluster["coord"]
    train = _dataset_csv(tmp_path, n=20)

    def post(val_name, content):
        p = tmp_path / val_name
        p.write_text(content)
        with open(train, "rb") as f, open(p, "rb") as vf:
            return httpx.post(
                base + "/api/job?rounds=4&patience=2&min_delta=0",
                files={"file": ("d.csv", f, "text/csv"),
                       "val_file": ("v.csv", vf, "text/csv")})

    assert post("v1.csv", "x1,x2,label\n1,2,0\n").status_code == 400  # 1 row
    big = "\n".join(["x1,x2,label"] + [f"{i},1,{i % 2}"
                                       for i in range(201)])
    assert post("v2.csv", big).status_code == 400                    # 201 rows
    assert post("v3.csv", "x1,x2,label\n1,2,9\n3,4,0\n").status_code == 400
    assert post("v4.csv", "1,2,0\n3,4,0\n").status_code == 400       # one class
    ok = post("v5.csv", "1,2,0\n3,4,1\n")                            # 2 rows
    assert ok.status_code in (200, 202)
    httpx.post(base + "/api/reset", timeout=5)

    # No validation file: pure training mode, no best weights endpoint.
    _create_job(base, train, None, rounds=4, patience=2)
    st = _start_and_wait(base, lambda d: d["status"] == "completed")
    assert st["validation_enabled"] is False
    assert st["val_history"] == []
    r = httpx.get(base + "/api/best_weights", timeout=5)
    assert r.status_code == 404


def test_resume_reinstalls_missing_validation_shard(cluster, tmp_path):
    """If a worker loses its validation shard (e.g. crash during job setup),
    resume re-installs both shards from the durable job before fencing."""
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay",
               json={"seconds": 0.3}, timeout=5)
    _create_job(base, _dataset_csv(tmp_path, n=30), _val_csv(tmp_path, n=12),
                rounds=8, patience=10)
    httpx.post(base + "/api/start", timeout=10).raise_for_status()
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)

    # Simulate a worker whose validation shard is missing on disk and in
    # memory, then hard-kill it so the group pauses.
    w1 = cluster["w1"]
    (cluster["paths"]["w1"] / "val_shard.json").unlink()
    w1.hard_kill()
    _wait_status(base, lambda d: d["status"] == "paused")
    w1.start()
    httpx.post(base + "/api/resume", timeout=20).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=120)

    # The healed worker holds its exclusive validation shard again and every
    # round was evaluated exactly once end to end.
    h1 = httpx.get(
        f"http://127.0.0.1:{w1.port}/health", timeout=5).json()
    assert h1["val_n"] > 0
    assert (cluster["paths"]["w1"] / "val_shard.json").exists()
    assert [h["round"] for h in st["val_history"]] == list(range(1, 9))


def test_worker_validate_endpoint_guards(cluster, tmp_path):
    base = cluster["coord"]
    _create_job(base, _dataset_csv(tmp_path, n=30), _val_csv(tmp_path, n=10),
                rounds=6, patience=10)
    st = _start_and_wait(
        base, lambda d: d["confirmed_round"] >= 3 and
        d["status"] in ("paused", "completed", "running"))
    # Let it finish so worker state is stable for direct calls.
    st = _wait_status(base, lambda d: d["status"] == "completed")
    task_id, gen, rnd = st["task_id"], st["gen"], st["confirmed_round"]
    w_url = f"http://127.0.0.1:{cluster['w1'].port}"
    state_before = json.loads(
        (cluster["paths"]["w1"] / "state.json").read_text())

    # Correct call: evaluates the durable round with the confirmed weights.
    ok = httpx.post(w_url + "/validate",
                    json={"task_id": task_id, "gen": gen, "round": rnd,
                          "w": st["w"]}, timeout=5)
    assert ok.status_code == 200
    assert ok.json()["n"] > 0 and np.isfinite(ok.json()["loss_sum"])

    # Wrong round / wrong weights / stale generation are all rejected and must
    # never mutate the durable training state (validation is read-only).
    r1 = httpx.post(w_url + "/validate",
                    json={"task_id": task_id, "gen": gen, "round": rnd - 1,
                          "w": st["w"]}, timeout=5)
    assert r1.status_code == 409
    r2 = httpx.post(w_url + "/validate",
                    json={"task_id": task_id, "gen": gen, "round": rnd,
                          "w": [9.9, 9.9, 9.9]}, timeout=5)
    assert r2.status_code == 409
    r3 = httpx.post(w_url + "/validate",
                    json={"task_id": task_id, "gen": gen - 1, "round": rnd,
                          "w": st["w"]}, timeout=5)
    assert r3.status_code == 409
    state_after = json.loads(
        (cluster["paths"]["w1"] / "state.json").read_text())
    assert state_after == state_before
