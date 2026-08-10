from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_VERSION = 1
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_CLOCK_SKEW_SECONDS = 300


class ClusterError(RuntimeError):
    pass


def utc_timestamp() -> float:
    return time.time()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def generate_cluster_token() -> str:
    return secrets.token_urlsafe(32)


def load_or_create_token(path: Path) -> tuple[str, bool]:
    destination = path.expanduser().absolute()
    if destination.exists():
        token = destination.read_text(encoding="utf-8").strip()
        if len(token) < 24:
            raise ClusterError("cluster token file is invalid or too short")
        return token, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = generate_cluster_token()
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    os.replace(temporary, destination)
    return token, True


def _body_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def sign_request(token: str, method: str, path: str, body: bytes, timestamp: str, nonce: str) -> str:
    if not token:
        raise ClusterError("cluster token is empty")
    message = "\n".join((method.upper(), path, timestamp, nonce, _body_sha256(body))).encode("utf-8")
    return hmac.new(token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_request_signature(
    token: str,
    method: str,
    path: str,
    body: bytes,
    timestamp: str,
    nonce: str,
    signature: str,
    *,
    now: float | None = None,
) -> None:
    try:
        sent_at = float(timestamp)
    except (TypeError, ValueError) as error:
        raise ClusterError("invalid request timestamp") from error
    current = utc_timestamp() if now is None else float(now)
    if not math.isfinite(sent_at) or abs(current - sent_at) > MAX_CLOCK_SKEW_SECONDS:
        raise ClusterError("request timestamp is outside the accepted clock window")
    if not isinstance(nonce, str) or not 16 <= len(nonce) <= 128:
        raise ClusterError("invalid request nonce")
    expected = sign_request(token, method, path, body, timestamp, nonce)
    if not hmac.compare_digest(expected, signature or ""):
        raise ClusterError("invalid request signature")


def evenly_spaced_indices(total: int, count: int) -> tuple[int, ...]:
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise ClusterError("total must be a positive integer")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= total:
        raise ClusterError("count must be within [1, total]")
    if count == 1:
        return (0,)
    result = [round(index * (total - 1) / (count - 1)) for index in range(count)]
    if len(set(result)) != count:
        raise ClusterError("failed to construct unique evenly spaced indices")
    return tuple(result)


def chunked(values: Sequence[int], size: int) -> Iterable[tuple[int, ...]]:
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ClusterError("chunk size must be a positive integer")
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    spec: Mapping[str, Any]
    spec_sha256: str
    attempts: int
    lease_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "spec": dict(self.spec),
            "spec_sha256": self.spec_sha256,
            "attempts": self.attempts,
            "lease_seconds": self.lease_seconds,
        }


class ClusterState:
    def __init__(self, database_path: Path, *, lease_seconds: int = 600, max_attempts: int = 5) -> None:
        if not isinstance(lease_seconds, int) or not 30 <= lease_seconds <= 86_400:
            raise ClusterError("lease_seconds must be within [30, 86400]")
        if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 100:
            raise ClusterError("max_attempts must be within [1, 100]")
        self.path = database_path.expanduser().absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextlib.contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    meta_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    spec_sha256 TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_until REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS results (
                    job_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('protocol_version', ?)",
                (str(PROTOCOL_VERSION),),
            )

    def add_jobs(self, specs: Sequence[Mapping[str, Any]]) -> int:
        added = 0
        now = utc_timestamp()
        with self._connection() as connection:
            for spec in specs:
                spec_hash = sha256_json(spec)
                job_id = f"job-{spec_hash[:24]}"
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO jobs(
                        job_id, status, spec_json, spec_sha256, created_at, updated_at
                    ) VALUES(?, 'queued', ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        canonical_json_bytes(spec).decode("utf-8"),
                        spec_hash,
                        now,
                        now,
                    ),
                )
                added += int(cursor.rowcount > 0)
        return added

    def register_worker(self, name: str, meta: Mapping[str, Any]) -> str:
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ClusterError("worker name must be non-empty and at most 200 characters")
        worker_id = str(uuid.uuid4())
        now = utc_timestamp()
        meta_json = canonical_json_bytes(meta).decode("utf-8")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO workers(worker_id, name, first_seen, last_seen, meta_json) VALUES(?, ?, ?, ?, ?)",
                (worker_id, name.strip(), now, now, meta_json),
            )
        return worker_id

    def heartbeat(self, worker_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE workers SET last_seen=? WHERE worker_id=?",
                (utc_timestamp(), worker_id),
            )
            if cursor.rowcount != 1:
                raise ClusterError("unknown worker")

    def _requeue_expired(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            """
            UPDATE jobs
            SET status='queued', lease_owner=NULL, lease_until=NULL, updated_at=?
            WHERE status='leased' AND lease_until IS NOT NULL AND lease_until < ? AND attempts < ?
            """,
            (now, now, self.max_attempts),
        )
        connection.execute(
            """
            UPDATE jobs
            SET status='failed', lease_owner=NULL, lease_until=NULL,
                last_error=COALESCE(last_error, 'lease expired too many times'), updated_at=?
            WHERE status='leased' AND lease_until IS NOT NULL AND lease_until < ? AND attempts >= ?
            """,
            (now, now, self.max_attempts),
        )

    def claim_job(self, worker_id: str) -> JobRecord | None:
        now = utc_timestamp()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            worker = connection.execute(
                "SELECT worker_id FROM workers WHERE worker_id=?",
                (worker_id,),
            ).fetchone()
            if worker is None:
                raise ClusterError("unknown worker")
            connection.execute("UPDATE workers SET last_seen=? WHERE worker_id=?", (now, worker_id))
            self._requeue_expired(connection, now)
            row = connection.execute(
                """
                SELECT job_id, spec_json, spec_sha256, attempts
                FROM jobs
                WHERE status='queued' AND attempts < ?
                ORDER BY created_at, job_id
                LIMIT 1
                """,
                (self.max_attempts,),
            ).fetchone()
            if row is None:
                return None
            attempts = int(row["attempts"]) + 1
            connection.execute(
                """
                UPDATE jobs
                SET status='leased', lease_owner=?, lease_until=?, attempts=?, updated_at=?
                WHERE job_id=?
                """,
                (worker_id, now + self.lease_seconds, attempts, now, row["job_id"]),
            )
            return JobRecord(
                job_id=row["job_id"],
                spec=json.loads(row["spec_json"]),
                spec_sha256=row["spec_sha256"],
                attempts=attempts,
                lease_seconds=self.lease_seconds,
            )

    def submit_result(
        self,
        worker_id: str,
        job_id: str,
        spec_sha256: str,
        *,
        ok: bool,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = utc_timestamp()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, lease_owner, spec_sha256, attempts FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise ClusterError("unknown job")
            if row["spec_sha256"] != spec_sha256:
                raise ClusterError("job spec hash mismatch")
            if row["status"] == "done":
                return
            if row["status"] != "leased" or row["lease_owner"] != worker_id:
                raise ClusterError("job is not leased to this worker")
            if ok:
                if result is None:
                    raise ClusterError("successful result is missing payload")
                payload = canonical_json_bytes(result).decode("utf-8")
                result_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                connection.execute(
                    """
                    INSERT OR REPLACE INTO results(job_id, worker_id, result_json, result_sha256, created_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (job_id, worker_id, payload, result_hash, now),
                )
                connection.execute(
                    """
                    UPDATE jobs
                    SET status='done', lease_owner=NULL, lease_until=NULL, last_error=NULL, updated_at=?
                    WHERE job_id=?
                    """,
                    (now, job_id),
                )
            else:
                message = (error or "worker reported an unspecified error")[:4000]
                next_status = "failed" if int(row["attempts"]) >= self.max_attempts else "queued"
                connection.execute(
                    """
                    UPDATE jobs
                    SET status=?, lease_owner=NULL, lease_until=NULL, last_error=?, updated_at=?
                    WHERE job_id=?
                    """,
                    (next_status, message, now, job_id),
                )

    def status(self) -> dict[str, Any]:
        now = utc_timestamp()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._requeue_expired(connection, now)
            job_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
            workers = connection.execute(
                "SELECT worker_id, name, first_seen, last_seen, meta_json FROM workers ORDER BY last_seen DESC"
            ).fetchall()
            result_count = connection.execute("SELECT COUNT(*) AS count FROM results").fetchone()["count"]
        counts = {row["status"]: int(row["count"]) for row in job_rows}
        return {
            "protocol_version": PROTOCOL_VERSION,
            "jobs": {
                "queued": counts.get("queued", 0),
                "leased": counts.get("leased", 0),
                "done": counts.get("done", 0),
                "failed": counts.get("failed", 0),
                "total": sum(counts.values()),
            },
            "results": int(result_count),
            "workers": [
                {
                    "worker_id": row["worker_id"],
                    "name": row["name"],
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "seconds_since_seen": max(0.0, now - float(row["last_seen"])),
                    "meta": json.loads(row["meta_json"]),
                }
                for row in workers
            ],
        }

    def results(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT r.job_id, r.worker_id, r.result_json, r.result_sha256, r.created_at,
                       j.spec_json, j.spec_sha256
                FROM results AS r JOIN jobs AS j ON j.job_id=r.job_id
                ORDER BY r.created_at, r.job_id
                """
            ).fetchall()
        return [
            {
                "job_id": row["job_id"],
                "worker_id": row["worker_id"],
                "created_at": row["created_at"],
                "spec_sha256": row["spec_sha256"],
                "result_sha256": row["result_sha256"],
                "spec": json.loads(row["spec_json"]),
                "result": json.loads(row["result_json"]),
            }
            for row in rows
        ]
