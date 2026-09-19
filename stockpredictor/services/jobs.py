"""In-process background job manager for long-running analysis.

Long predictions (model training + network sentiment) can take over a minute.
Instead of blocking the HTTP request, ``POST /analyze`` hands the work to a
daemon thread and the browser polls :func:`job_status`. Completed results are
written to ``runtime_state/jobs/`` so the results page survives restarts.

Memory safety: a single AUTO/LSTM/GRU training run already puts real pressure
on the 512MB free-tier worker, so only one heavy job runs at a time (a second
one queues briefly, then fails with a clear "busy" response instead of running
two models at once and getting OOM-killed), and every job releases TensorFlow/
Keras graph state before the next one starts.
"""
from __future__ import annotations

import gc
import json
import logging
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from production_core import STATE_DIR

logger = logging.getLogger("stockpredictor.services.jobs")

JOB_TTL_SECONDS = 4 * 60 * 60
JOBS_DIR = STATE_DIR / "jobs"

# Only one model-training job at a time; a second request waits this long, then
# gets a clean "busy" error rather than doubling memory pressure.
GATE_WAIT_SECONDS = 30
BUSY_MESSAGE = "Another analysis is already in progress here. Please wait a moment and try again."

_STATUSES = {"queued", "running", "complete", "error"}


def _free_model_memory() -> None:
    """Best-effort release of TF/Keras graph state and Python garbage.

    Runs after every job so model/optimizer/session memory does not accumulate
    on the shared worker across sequential /analyze requests. Guarded by
    ``sys.modules`` so a test process that never trains never forces the heavy
    TensorFlow import.
    """
    try:
        if "predictor_core" in sys.modules:
            from predictor_core import release_tf_memory

            release_tf_memory()
    except Exception:  # noqa: BLE001 - cleanup must never mask job outcome
        logger.warning("Could not release model memory", exc_info=True)
    gc.collect()


class JobManager:
    """Thread-safe registry of background jobs keyed by short id."""

    def __init__(self, ttl_seconds: int = JOB_TTL_SECONDS,
                 gate_wait_seconds: int = GATE_WAIT_SECONDS) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._training_gate = threading.Semaphore(1)
        self._gate_wait_seconds = gate_wait_seconds
        try:
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - filesystem edge cases
            logger.warning("Could not create jobs directory %s", JOBS_DIR)

    def submit(
        self,
        fn: Callable,
        *args: Any,
        dedupe_key: Optional[str] = None,
        owner: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Start ``fn(*args, **kwargs)`` on a worker thread.

        When ``dedupe_key`` matches a queued, running, or successfully
        completed job *owned by the same user*, that job's id is returned
        instead of starting a second (expensive) run for identical work.  The
        completed state is included because very short jobs can finish between
        two otherwise concurrent requests. Failed jobs are deliberately not
        deduplicated so callers can retry them. ``owner`` scopes status/result
        reads so one authenticated user can never poll another user's job.
        """
        with self._lock:
            self._prune_locked()
            if dedupe_key:
                for job_id, job in self._jobs.items():
                    if (
                        job.get("dedupe_key") == dedupe_key
                        and job.get("owner") == owner
                        and job.get("status") in ("queued", "running", "complete")
                    ):
                        return job_id

            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {
                "status": "queued",
                "message": "Queued — starting analysis…",
                "dedupe_key": dedupe_key,
                "owner": owner,
                "created": time.time(),
                "result": None,
            }
            thread = threading.Thread(
                target=self._run, args=(job_id, fn, args, kwargs), daemon=True
            )
            thread.start()
            return job_id

    def _run(self, job_id: str, fn: Callable, args: tuple, kwargs: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["message"] = "Fetching data and training the model - this can take a minute..."

        # Single-slot training gate: never run two model builds at once on one
        # worker. A second job queues briefly, then fails cleanly with a busy
        # message instead of doubling memory pressure and crashing.
        if not self._training_gate.acquire(timeout=self._gate_wait_seconds):
            logger.warning("Job %s rejected: another analysis is already running", job_id)
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                job["status"] = "error"
                job["message"] = BUSY_MESSAGE
                job["result"] = {"error": BUSY_MESSAGE}
            return
        try:
            try:
                result = fn(*args, **kwargs)
                if not isinstance(result, dict):
                    result = {"result": result}
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None:
                        return
                    job["status"] = "complete"
                    job["message"] = "Analysis complete"
                    job["result"] = result
                self._write_disk(job_id, result)
            except Exception as exc:  # noqa: BLE001 - surface failure to caller
                logger.exception("Job %s failed", job_id)
                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None:
                        return
                    job["status"] = "error"
                    job["message"] = str(exc)
                    job["result"] = {"error": str(exc)}
        finally:
            _free_model_memory()
            self._training_gate.release()

    def status(self, job_id: str, requester: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Job status for ``requester`` (None = unrestricted/internal use).

        Jobs with an owner are invisible to any other requester; legacy on-disk
        results written before ownership existed stay readable.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            owner = job.get("owner") if job else None
            mem_status = job["status"] if job else None
            mem_message = job["message"] if job else None
            mem_result = job["result"] if job else None

        if owner is not None and requester is not None and requester != owner:
            return None
        if mem_status is not None:
            return {
                "status": mem_status,
                "message": mem_message,
                "result": mem_result,
            }

        disk_owner, disk_result = self._read_disk(job_id)
        if disk_result is None:
            return None
        if disk_owner is not None and requester is not None and requester != disk_owner:
            return None
        return {"status": "complete", "message": "Analysis complete", "result": disk_result}

    def result(self, job_id: str, requester: Optional[str] = None) -> Optional[Dict[str, Any]]:
        status = self.status(job_id, requester=requester)
        if status is None or status["status"] != "complete":
            return None
        return status["result"]

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if now - job.get("created", now) > self._ttl
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
            try:
                (JOBS_DIR / f"{job_id}.json").unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                pass

    def _write_disk(self, job_id: str, result: Dict[str, Any]) -> None:
        try:
            path = JOBS_DIR / f"{job_id}.json"
            with self._lock:
                owner = self._jobs.get(job_id, {}).get("owner")
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump({"owner": owner, "result": result}, fh, default=str)
            tmp.replace(path)
        except Exception:  # pragma: no cover - defensive
            logger.warning("Could not persist job result %s", job_id)

    def _read_disk(self, job_id: str) -> tuple:
        """Return ``(owner, result)`` from the on-disk artifact.

        Legacy files written before ownership existed hold a bare result dict;
        they read back as ``(None, <result>)`` and stay visible to any
        authenticated requester.
        """
        path = JOBS_DIR / f"{job_id}.json"
        if not path.exists():
            return None, None
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:  # pragma: no cover - defensive
            return None, None
        if isinstance(payload, dict) and "result" in payload:
            return payload.get("owner"), payload.get("result")
        return None, payload


job_manager = JobManager()
