"""Process-isolated, session-sticky HTTP serving for recommendation scorers."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import signal
import subprocess
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import urlopen


MAX_PROXY_BODY_BYTES = 2 * 1024 * 1024
MAX_PROXY_RESPONSE_BYTES = 16 * 1024 * 1024
SESSION_ROUTE_LIMIT = 100_000
CONTROL_PLANE_PATHS = {
    "/api/benchmark", "/api/compare", "/api/config", "/api/dataset/load",
    "/api/mine", "/api/training-confirmation", "/api/tune",
}
HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


def stable_shard(value: str, workers: int) -> int:
    """Return a process-stable shard without Python hash randomization."""
    if workers < 1:
        raise ValueError("workers must be positive")
    digest = hashlib.blake2s(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % workers


def cookie_shard(cookie: str | None, workers: int) -> int | None:
    for item in str(cookie or "").split(";"):
        key, separator, value = item.strip().partition("=")
        if separator and key == "recommendation_shard":
            try:
                shard = int(value)
            except ValueError:
                return None
            return shard if 0 <= shard < workers else None
    return None


@dataclass
class Worker:
    index: int
    port: int
    command: tuple[str, ...]
    process: subprocess.Popen[bytes] | None = None
    restarts: int = 0


class ScorerPool:
    """Own isolated scorer processes and restart failed workers."""

    def __init__(self, workers: list[Worker], *, ready_timeout: float = 180.0):
        if not workers:
            raise ValueError("scorer pool requires at least one worker")
        self.workers = workers
        self.ready_timeout = float(ready_timeout)
        self._lock = threading.RLock()
        self._closing = threading.Event()
        self._monitor: threading.Thread | None = None

    @staticmethod
    def _ready(port: int, timeout: float = 1.0) -> bool:
        try:
            with urlopen(
                f"http://127.0.0.1:{port}/health/ready", timeout=timeout
            ) as response:
                return response.status == 200
        except (OSError, URLError):
            return False

    def _spawn(self, worker: Worker) -> None:
        worker.process = subprocess.Popen(
            worker.command,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    @staticmethod
    def _terminate(worker: Worker) -> None:
        process=worker.process
        if process is None or process.poll() is not None:
            worker.process=None
            return
        try:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=5.0)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)
        finally:
            worker.process=None

    def _wait_ready(self, worker: Worker) -> None:
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            process = worker.process
            if process is None or process.poll() is not None:
                code = None if process is None else process.returncode
                raise RuntimeError(
                    f"scorer worker {worker.index} exited during startup ({code})"
                )
            if self._ready(worker.port):
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"scorer worker {worker.index} was not ready within "
            f"{self.ready_timeout:.1f}s"
        )

    def start(self) -> None:
        try:
            for worker in self.workers:
                self._spawn(worker)
            for worker in self.workers:
                self._wait_ready(worker)
        except BaseException:
            self.close()
            raise
        self._monitor = threading.Thread(
            target=self._monitor_loop,
            name="recommendation-scorer-supervisor",
            daemon=True,
        )
        self._monitor.start()

    def _monitor_loop(self) -> None:
        while not self._closing.wait(1.0):
            for worker in self.workers:
                with self._lock:
                    process = worker.process
                    if process is not None and process.poll() is None:
                        continue
                    if self._closing.is_set():
                        return
                    worker.restarts += 1
                    self._spawn(worker)
                try:
                    self._wait_ready(worker)
                except BaseException:
                    # Do not leave a live-but-unready child in the pool after
                    # its deadline. The next interval starts a clean process.
                    with self._lock:
                        self._terminate(worker)
                    continue

    def status(self) -> list[dict[str, Any]]:
        rows = []
        for worker in self.workers:
            with self._lock:
                process = worker.process
                pid = process.pid if process is not None else None
                alive = bool(process is not None and process.poll() is None)
            rows.append({
                "index": worker.index,
                "port": worker.port,
                "pid": pid,
                "alive": alive,
                "ready": alive and self._ready(worker.port, timeout=0.5),
                "restarts": worker.restarts,
            })
        return rows

    def close(self) -> None:
        self._closing.set()
        for worker in self.workers:
            process = worker.process
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10.0
        for worker in self.workers:
            process = worker.process
            if process is None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5.0)


class BoundedHTTPServer(HTTPServer):
    """HTTP server with a fixed execution pool and bounded admission queue."""

    allow_reuse_address = True
    request_queue_size = 256

    def __init__(self, address, handler, *, threads: int, queue: int):
        super().__init__(address, handler)
        self.executor = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="recommendation-gateway"
        )
        self.capacity = threading.BoundedSemaphore(threads + queue)

    def process_request(self, request, client_address):
        if not self.capacity.acquire(blocking=False):
            body = b'{"error":"server busy"}'
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode("ascii")
                    + b"Connection: close\r\n\r\n" + body
                )
            finally:
                self.shutdown_request(request)
            return
        self.executor.submit(self._bounded_request, request, client_address)

    def _bounded_request(self, request, client_address):
        try:
            self.finish_request(request, client_address)
            self.shutdown_request(request)
        except BaseException:
            self.handle_error(request, client_address)
            self.shutdown_request(request)
        finally:
            self.capacity.release()

    def server_close(self):
        super().server_close()
        self.executor.shutdown(wait=True, cancel_futures=True)


class GatewayState:
    def __init__(self, pool: ScorerPool, *, backend_timeout: float):
        self.pool = pool
        self.backend_timeout = float(backend_timeout)
        self.sessions: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self.lock = threading.RLock()

    def remember_session(self, session: str, shard: int) -> None:
        if not session:
            return
        with self.lock:
            self.sessions.pop(session, None)
            self.sessions[session] = (
                shard,self.pool.workers[shard].restarts,
            )
            while len(self.sessions) > SESSION_ROUTE_LIMIT:
                self.sessions.popitem(last=False)

    def session_shard(self, session: str | None) -> tuple[int | None, bool]:
        if not session:
            return None,False
        with self.lock:
            route = self.sessions.get(session)
            if route is not None:
                shard,generation=route
                if self.pool.workers[shard].restarts!=generation:
                    self.sessions.pop(session,None)
                    return shard,True
                self.sessions.move_to_end(session)
                return shard,False
            return None,False

    def aggregate_state(self) -> dict[str, Any]:
        snapshots=[]
        for worker in self.pool.workers:
            connection=http.client.HTTPConnection(
                "127.0.0.1",worker.port,timeout=self.backend_timeout
            )
            try:
                connection.request("GET","/api/state")
                response=connection.getresponse()
                payload=response.read()
                if response.status!=200:
                    raise RuntimeError(
                        f"scorer shard {worker.index} state failed"
                    )
                snapshots.append(json.loads(payload))
            finally:
                connection.close()
        identities={json.dumps({
            "version":snapshot.get("version"),
            "config":snapshot.get("config"),
            "dataset":snapshot.get("dataset"),
            "serving_model_sha256":snapshot.get("engine",{}).get(
                "serving_model_sha256"
            ),
        },sort_keys=True,separators=(",",":")) for snapshot in snapshots}
        if len(identities)!=1:
            raise RuntimeError("scorer shards have divergent model state")
        aggregate=json.loads(json.dumps(snapshots[0]))
        engines=[snapshot.get("engine",{}) for snapshot in snapshots]
        aggregate["instance_id"]="pool_"+hashlib.sha256(
            "|".join(str(snapshot.get("instance_id",""))
                     for snapshot in snapshots).encode("utf-8")
        ).hexdigest()[:24]
        engine=aggregate.setdefault("engine",{})
        engine["worker_pid"]=[item.get("worker_pid") for item in engines]
        engine["pool_size"]=len(engines)
        for field in (
            "point_case_cache_entries","pair_case_cache_entries",
            "point_channel_cache_entries","pair_channel_cache_entries",
            "point_reasoner_query_calls","pair_reasoner_query_calls",
            "relational_reasoner_query_calls","relational_reasoner_query_roots",
        ):
            engine[field]=sum(int(item.get(field,0) or 0) for item in engines)
        caches=[item.get("feed_rank_cache",{}) for item in engines]
        engine["feed_rank_cache"]={
            field:sum(int(cache.get(field,0) or 0) for cache in caches)
            for field in ("entries","hits","misses")
        }
        aggregate["pool"]={
            "mode":"session_sticky_process_isolation",
            "workers":self.pool.status(),
            "session_routes":len(self.sessions),
        }
        return aggregate


def gateway_handler(state: GatewayState):
    class GatewayHandler(BaseHTTPRequestHandler):
        server_version = "SymbolicRecommendationGateway/1.0"
        protocol_version = "HTTP/1.1"

        def _json(self, payload: Any, status: int) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            raw = self.headers.get("Content-Length")
            if raw is None:
                return b""
            try:
                length = int(raw)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if length < 0 or length > MAX_PROXY_BODY_BYTES:
                raise OverflowError("request body is too large")
            return self.rfile.read(length)

        def _route(self, body: bytes) -> tuple[int, bool, bool]:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            payload = {}
            if body:
                try:
                    candidate = json.loads(body)
                    payload = candidate if isinstance(candidate, dict) else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    payload = {}
            session = (
                payload.get("session")
                or (query.get("session") or [None])[0]
            )
            cursor = (query.get("cursor") or [None])[0]
            if not session and cursor:
                session = str(cursor).split(":", 1)[0]
            remembered,stale = state.session_shard(
                str(session) if session else None
            )
            if remembered is not None:
                return remembered,False,stale
            workers = len(state.pool.workers)
            sticky = cookie_shard(self.headers.get("Cookie"), workers)
            if sticky is not None:
                return sticky,False,stale
            user = (
                payload.get("user")
                or (query.get("user") or [None])[0]
                or self.headers.get("X-User")
            )
            if user:
                return stable_shard(str(user),workers),True,stale
            return 0,True,stale

        @staticmethod
        def _reset_feed_path(path: str) -> str:
            parsed=urlparse(path)
            query=[(key,value) for key,value in parse_qsl(
                parsed.query,keep_blank_values=True
            ) if key not in {"session","cursor"}]
            return urlunparse(parsed._replace(query=urlencode(query)))

        def _proxy(self) -> None:
            try:
                body = self._body()
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return
            except OverflowError as exc:
                self._json({"error": str(exc)}, 413)
                return
            shard,set_cookie,stale_session=self._route(body)
            parsed_path=urlparse(self.path).path
            if stale_session and not (
                    self.command=="GET" and parsed_path=="/api/feed"):
                self._json({
                    "error":"feed session expired after scorer restart",
                    "reset":True,
                },409)
                return
            upstream_path=(self._reset_feed_path(self.path)
                           if stale_session else self.path)
            worker = state.pool.workers[shard]
            process = worker.process
            if process is None or process.poll() is not None:
                self._json({"error": "scorer shard unavailable"}, 503)
                return
            headers = {
                key: value for key, value in self.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
                and key.lower() not in {"host", "content-length"}
            }
            headers["Host"] = f"127.0.0.1:{worker.port}"
            headers["X-Forwarded-For"] = self.client_address[0]
            if body:
                headers["Content-Length"] = str(len(body))
            connection = http.client.HTTPConnection(
                "127.0.0.1", worker.port, timeout=state.backend_timeout
            )
            try:
                connection.request(
                    self.command,upstream_path,body=body,headers=headers
                )
                response = connection.getresponse()
                response_body=response.read(MAX_PROXY_RESPONSE_BYTES+1)
                if len(response_body)>MAX_PROXY_RESPONSE_BYTES:
                    self._json({"error":"scorer response is too large"},502)
                    return
            except (OSError, http.client.HTTPException, TimeoutError):
                self._json({"error": "scorer upstream failed"}, 502)
                return
            finally:
                connection.close()
            content_type = response.getheader("Content-Type", "")
            if (parsed_path == "/api/feed"
                    and "application/json" in content_type):
                try:
                    payload = json.loads(response_body)
                    if stale_session and isinstance(payload,dict):
                        payload["reset"]=True
                        payload["reset_reason"]="scorer_restart"
                        response_body=json.dumps(
                            payload,separators=(",",":")
                        ).encode("utf-8")
                    session = payload.get("session") if isinstance(payload, dict) else None
                    if session:
                        state.remember_session(str(session), shard)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                lowered = key.lower()
                if lowered in HOP_BY_HOP_HEADERS or lowered == "content-length":
                    continue
                self.send_header(key, value)
            if set_cookie:
                self.send_header(
                    "Set-Cookie",
                    f"recommendation_shard={shard}; Path=/; HttpOnly; SameSite=Lax",
                )
            self.send_header("X-Recommendation-Shard", str(shard))
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def _health(self, ready: bool) -> None:
            workers = state.pool.status()
            healthy = all(row["ready" if ready else "alive"] for row in workers)
            self._json({
                "status": "ready" if healthy and ready else "live" if healthy else "degraded",
                "workers": workers,
                "session_routes": len(state.sessions),
            }, 200 if healthy else 503)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health/live":
                self._health(False)
                return
            if path == "/health/ready":
                self._health(True)
                return
            if path == "/api/state":
                try:
                    self._json(state.aggregate_state(),200)
                except BaseException:
                    self._json({"error":"scorer pool state unavailable"},503)
                return
            self._proxy()

        def do_POST(self):
            if urlparse(self.path).path in CONTROL_PLANE_PATHS:
                self._json({
                    "error": (
                        "mutable control-plane operation disabled on serving "
                        "gateway; publish a validated frozen model"
                    )
                }, 409)
                return
            self._proxy()

        def log_message(self, format, *args):
            return

    return GatewayHandler


def serve_pool(
    *, host: str, port: int, worker_commands: list[tuple[str, ...]],
    worker_ports: list[int], ready_timeout: float, backend_timeout: float,
    gateway_threads: int, gateway_queue: int,
) -> None:
    workers = [
        Worker(index=index, port=worker_port, command=command)
        for index, (worker_port, command) in enumerate(
            zip(worker_ports, worker_commands, strict=True)
        )
    ]
    pool = ScorerPool(workers, ready_timeout=ready_timeout)
    pool.start()
    state = GatewayState(pool, backend_timeout=backend_timeout)
    server = BoundedHTTPServer(
        (host, port), gateway_handler(state),
        threads=gateway_threads, queue=gateway_queue,
    )
    print(
        f"Recommendation gateway: http://{host}:{port} "
        f"({len(workers)} scorer processes)", flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        pool.close()
