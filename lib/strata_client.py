"""StrataClient — CLI subprocess wrapper for the Strata database.

Communicates with the `strata` binary via a persistent subprocess pipe
(stdin/stdout) in JSON mode, eliminating Python FFI overhead from benchmark
measurements.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path


class StrataError(Exception):
    """Error returned by the Strata CLI."""


class StrataClient:
    """Spawn ``strata --json --db <path>`` in pipe mode and send commands via stdin."""

    def __init__(
        self,
        db_path: str | os.PathLike,
        *,
        cache: bool = False,
        auto_embed: bool = False,
        binary: str | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self._binary = self._resolve_binary(binary)
        self._lock = threading.Lock()

        args = [self._binary, "--json", "--db", self.db_path]
        if cache:
            args.append("--cache")
        if auto_embed:
            args.append("--auto-embed")

        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        self.kv = KvNamespace(self)
        self.vectors = VectorNamespace(self)
        self.graph = GraphNamespace(self)

    # ------------------------------------------------------------------
    # Binary resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_binary(explicit: str | None) -> str:
        if explicit:
            return explicit
        env = os.environ.get("STRATA_BIN")
        if env:
            return env
        which = shutil.which("strata")
        if which:
            return which
        # Fall back to the release build in a sibling directory
        fallback = Path(__file__).resolve().parent.parent.parent / "strata-core" / "target" / "release" / "strata"
        if fallback.exists():
            return str(fallback)
        raise FileNotFoundError(
            "Cannot find 'strata' binary. Set STRATA_BIN, add it to PATH, "
            "or pass binary= to StrataClient."
        )

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _read_response(self) -> str:
        """Read a complete JSON value from stdout.

        The CLI pretty-prints JSON, so we accumulate lines and attempt to
        parse with ``json.loads`` after each line.  This handles all edge
        cases (braces inside string values, multi-line pretty-printing, bare
        string/number literals) correctly.
        """
        buf: list[str] = []

        while True:
            line = self._proc.stdout.readline()
            if not line:
                # EOF — process died
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise StrataError(f"Strata process exited unexpectedly: {stderr.strip()}")

            # Skip blank lines between responses
            if not line.strip():
                continue

            buf.append(line)
            text = "".join(buf)
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                # Incomplete JSON — keep reading
                continue

    def _send(self, command: str) -> dict | list | str | None:
        """Send a command line and return the parsed, unwrapped response.

        The pipe protocol is line-delimited, so embedded newlines in the
        command are collapsed to spaces to prevent splitting.
        """
        # Collapse newlines — the CLI reads one command per line
        safe_command = command.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise StrataError("Strata process is not running")
            self._proc.stdin.write(safe_command + "\n")
            self._proc.stdin.flush()
            raw = self._read_response()

        parsed = json.loads(raw)
        return _unwrap(parsed)

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: str | None = None,
        primitives: list[str] | None = None,
        expand: bool = False,
        rerank: bool = False,
    ) -> list[dict]:
        parts = ["search", shlex.quote(query), "--k", str(k)]
        if mode:
            parts.extend(["--mode", mode])
        if primitives:
            parts.extend(["--primitives", ",".join(primitives)])
        if expand:
            parts.append("--expand")
        if rerank:
            parts.append("--rerank")
        result = self._send(" ".join(parts))
        return result if isinstance(result, list) else []

    def flush(self) -> None:
        self._send("flush")

    def setup(self) -> None:
        self._send("setup")

    def configure_model(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
    ) -> None:
        parts = ["configure-model", shlex.quote(endpoint), shlex.quote(model)]
        if api_key:
            parts.extend(["--api-key", shlex.quote(api_key)])
        self._send(" ".join(parts))

    def info(self) -> dict:
        result = self._send("info")
        return result if isinstance(result, dict) else {}

    def ping(self) -> str:
        result = self._send("ping")
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get("version", str(result))
        return str(result)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdin and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None


# ======================================================================
# KV Namespace
# ======================================================================

class KvNamespace:
    def __init__(self, client: StrataClient) -> None:
        self._client = client

    def put(self, key: str, value: str) -> None:
        self._client._send(f"kv put {shlex.quote(key)} {shlex.quote(value)}")

    def get(self, key: str) -> str | None:
        return self._client._send(f"kv get {shlex.quote(key)}")

    def list(self, *, prefix: str | None = None, limit: int | None = None) -> list[str]:
        parts = ["kv", "list"]
        if prefix is not None:
            parts.extend(["--prefix", shlex.quote(prefix)])
        if limit is not None:
            parts.extend(["--limit", str(limit)])
        result = self._client._send(" ".join(parts))
        return result if isinstance(result, list) else []

    def delete(self, key: str) -> None:
        self._client._send(f"kv del {shlex.quote(key)}")


# ======================================================================
# Vector Namespace
# ======================================================================

class VectorNamespace:
    def __init__(self, client: StrataClient) -> None:
        self._client = client

    def create(self, name: str, dimension: int, metric: str) -> VectorCollection:
        self._client._send(f"vector create {shlex.quote(name)} {dimension} {shlex.quote(metric)}")
        return VectorCollection(self._client, name)


class VectorCollection:
    def __init__(self, client: StrataClient, name: str) -> None:
        self._client = client
        self._name = name

    def upsert(self, entries: list[dict]) -> None:
        """Batch upsert vectors. Uses a temp file for large payloads."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False,
        ) as f:
            json.dump(entries, f)
            tmp_path = f.name
        try:
            self._client._send(
                f"vector batch-upsert {shlex.quote(self._name)} --file {shlex.quote(tmp_path)}"
            )
        finally:
            os.unlink(tmp_path)

    def search(self, query: list[float], *, k: int = 10) -> list[dict]:
        query_json = json.dumps(query)
        result = self._client._send(
            f"vector search {shlex.quote(self._name)} {shlex.quote(query_json)} {k}"
        )
        return result if isinstance(result, list) else []

    def stats(self) -> dict:
        result = self._client._send(f"vector stats {shlex.quote(self._name)}")
        return result if isinstance(result, dict) else {}


# ======================================================================
# Graph Namespace
# ======================================================================

class GraphNamespace:
    def __init__(self, client: StrataClient) -> None:
        self._client = client

    def create(self, name: str) -> None:
        self._client._send(f"graph create {shlex.quote(name)}")

    def bulk_insert(
        self,
        graph: str,
        *,
        nodes: list[dict] | None = None,
        edges: list[dict] | None = None,
        file_path: str | None = None,
    ) -> None:
        if file_path:
            self._client._send(
                f"graph bulk-insert {shlex.quote(graph)} --file {shlex.quote(file_path)}"
            )
        else:
            payload = {}
            if nodes is not None:
                payload["nodes"] = nodes
            if edges is not None:
                payload["edges"] = edges
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
            ) as f:
                json.dump(payload, f)
                tmp_path = f.name
            try:
                self._client._send(
                    f"graph bulk-insert {shlex.quote(graph)} --file {shlex.quote(tmp_path)}"
                )
            finally:
                os.unlink(tmp_path)

    def bfs(
        self,
        graph: str,
        start: str,
        max_depth: int,
        *,
        direction: str | None = None,
        max_nodes: int | None = None,
    ) -> dict:
        parts = ["graph", "bfs", shlex.quote(graph), shlex.quote(start), str(max_depth)]
        if direction:
            parts.extend(["--direction", direction])
        if max_nodes is not None:
            parts.extend(["--max-nodes", str(max_nodes)])
        result = self._client._send(" ".join(parts))
        return result if isinstance(result, dict) else {}

    def neighbors(
        self,
        graph: str,
        node_id: str,
        *,
        direction: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict]:
        parts = ["graph", "neighbors", shlex.quote(graph), shlex.quote(node_id)]
        if direction:
            parts.extend(["--direction", direction])
        if edge_type:
            parts.extend(["--edge-type", shlex.quote(edge_type)])
        result = self._client._send(" ".join(parts))
        return result if isinstance(result, list) else []

    def list_nodes(self, graph: str) -> list[str]:
        result = self._client._send(f"graph list-nodes {shlex.quote(graph)}")
        return result if isinstance(result, list) else []


# ======================================================================
# Response unwrapping
# ======================================================================

def _unwrap(value):
    """Unwrap serde-serialized ``Output`` enum variants from the CLI."""
    if value is None:
        return None

    if isinstance(value, str):
        if value == "Unit":
            return None
        return value

    if isinstance(value, list):
        return value

    if not isinstance(value, dict):
        return value

    # Single-key enum wrappers
    if len(value) == 1:
        key, inner = next(iter(value.items()))

        if key == "Version":
            return inner

        if key == "MaybeVersioned":
            if inner is None:
                return None
            # inner is {"value": {"String": "val"}, "version": ...}
            raw_val = inner.get("value") if isinstance(inner, dict) else inner
            if isinstance(raw_val, dict) and len(raw_val) == 1:
                return next(iter(raw_val.values()))
            return raw_val

        if key == "Keys":
            return inner

        if key == "SearchResults":
            # v0.6+: inner is {"hits": [...], "stats": {...}}
            # Legacy: inner is a bare list of hits
            if isinstance(inner, dict):
                return inner.get("hits", [])
            return inner

        if key == "VectorMatches":
            return inner

        if key == "VectorStats":
            return inner

        if key == "GraphBfs":
            return inner

        if key == "GraphNeighbors":
            return inner

        if key == "GraphNodes":
            return inner

        if key == "Pong":
            if isinstance(inner, dict):
                return inner.get("version", str(inner))
            return str(inner)

        if key == "Info":
            return inner

        if key == "error":
            raise StrataError(inner)

    # If we don't recognise the shape, return as-is
    return value


# ======================================================================
# Batch execution — no Python overhead in the timing loop
# ======================================================================

def _parse_json_stream(output: str) -> list:
    """Parse a stream of pretty-printed JSON values from CLI stdout.

    The strata CLI outputs one JSON value per command, potentially
    pretty-printed across multiple lines.  This parser accumulates lines
    and attempts ``json.loads`` after each, emitting a result whenever a
    complete JSON value is found.
    """
    results: list = []
    buf: list[str] = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        buf.append(line)
        joined = "\n".join(buf)
        try:
            parsed = json.loads(joined)
            results.append(_unwrap(parsed))
            buf = []
        except json.JSONDecodeError:
            continue
    return results


def batch_execute(
    commands: list[str],
    *,
    db_path: str,
    binary: str | None = None,
    auto_embed: bool = False,
    cache: bool = False,
    parse_responses: bool = True,
) -> tuple[float, list]:
    """Pipe commands to a fresh ``strata`` process and return ``(elapsed_s, responses)``.

    All commands are written to a temporary file and fed to the CLI via
    stdin.  Only the strata process execution time is measured — there is
    **no Python overhead per operation**.  Use this for benchmark timing
    loops instead of the interactive :class:`StrataClient` pipe.

    Parameters
    ----------
    commands:
        One CLI command per element (e.g. ``'kv get "mykey"'``).
    db_path:
        Path to the Strata database directory.
    binary:
        Explicit path to the ``strata`` binary (optional).
    auto_embed:
        Pass ``--auto-embed`` to the CLI process.
    cache:
        Pass ``--cache`` for an ephemeral in-memory database.
    parse_responses:
        If *False*, discard stdout and return an empty list.  Use this
        for load phases where individual responses are not needed.

    Returns
    -------
    (elapsed_seconds, responses)
        Wall-clock time of the CLI process and the list of parsed/unwrapped
        response values (one per command).
    """
    resolved_binary = StrataClient._resolve_binary(binary)
    args = [resolved_binary, "--json", "--db", db_path]
    if cache:
        args.append("--cache")
    if auto_embed:
        args.append("--auto-embed")

    # Write commands to a temp file to avoid pipe buffer limits on
    # large workloads (100K+ commands).  Collapse embedded newlines to
    # spaces — the pipe protocol is line-delimited (one command per line).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False,
    ) as f:
        for cmd in commands:
            safe = cmd.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            f.write(safe + "\n")
        cmd_path = f.name

    try:
        with open(cmd_path) as stdin_file:
            stdout_target = subprocess.PIPE if parse_responses else subprocess.DEVNULL
            t0 = time.perf_counter()
            proc = subprocess.run(
                args,
                stdin=stdin_file,
                stdout=stdout_target,
                stderr=subprocess.PIPE,
                text=True,
            )
            elapsed = time.perf_counter() - t0
    finally:
        os.unlink(cmd_path)

    if proc.returncode != 0 and not (parse_responses and proc.stdout):
        raise StrataError(
            f"Batch execute failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )

    responses = _parse_json_stream(proc.stdout) if parse_responses else []
    return elapsed, responses
