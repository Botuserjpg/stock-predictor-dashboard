"""In-process background job manager for long-running analysis.

Long predictions (model training + network sentiment) can take over a minute.
Instead of blocking the HTTP request, ``POST /analyze`` hands the work to a
daemon thread and the browser polls :func:`job_status`. Completed results are
written to ``runtime_state/jobs/`` so the results page survives restarts.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from production_core import STATE_DIR

logger = logging.getLogger("stockpredictor.services.jobs")

JOB_TTL_SECONDS = 4 * 60 * 60
JOBS_DIR = STATE_DIR / "jobs"

_STATUSES = {"queued", "running", "complete", "error"}


class JobManager:
    """Thread-safe registry of background jobs keyed by short id."""

    def __init__(self, ttl_seconds: int = JOB_TTL_SECONDS) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
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
            job["message"] = "Fetching data and training the model — this can take a minute…"
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
