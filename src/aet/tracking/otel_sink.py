"""otel_sink — a minimal OTLP/HTTP (JSON) receiver that captures Claude Code telemetry to a file.

Full-fidelity token/cost/timing capture. Claude Code, with telemetry enabled, exports one
``claude_code.api_request`` log record PER API turn carrying real ``input_tokens / output_tokens /
cache_read_tokens / cache_creation_tokens / cost_usd / duration_ms`` + a real timestamp — plus
``claude_code.token.usage`` / ``cost.usage`` metrics and tool/prompt events. This tiny server accepts
the OTLP/HTTP JSON the CLI POSTs to ``/v1/logs`` and ``/v1/metrics`` and appends each envelope as one
line to ``<out>`` (JSONL), so :func:`aet.trajectory.importers.otel.import_otel` can reconstruct the
exact per-turn trajectory with NO interpolation (real tokens, real cost, real durations, real cache).

Run one sink per agent invocation, alongside the agent. Point the agent's environment at it:

    CLAUDE_CODE_ENABLE_TELEMETRY=1 \\
    OTEL_LOGS_EXPORTER=otlp OTEL_METRICS_EXPORTER=otlp \\
    OTEL_EXPORTER_OTLP_PROTOCOL=http/json \\
    OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:<PORT>

Usage (either form):

    python -m aet.tracking.otel_sink --port 4317 --out otel_logs.jsonl
    aet otel-sink --port 4317 --out otel_logs.jsonl

Only stdlib — no OTel SDK or collector needed. Bind is 127.0.0.1 by default (localhost only).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(out_path: str):
    """Build a request handler that appends each received OTLP envelope to ``out_path`` as JSONL."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass  # quiet — do not spam stderr per request

        def _read_body(self) -> bytes:
            # The CLI's OTLP/HTTP exporter uses chunked transfer-encoding (no Content-Length) and
            # may gzip. Handle both, else the body reads as empty.
            te = (self.headers.get("Transfer-Encoding") or "").lower()
            if "chunked" in te:
                chunks = []
                while True:
                    size_line = self.rfile.readline().strip()
                    try:
                        sz = int(size_line.split(b";")[0], 16)
                    except ValueError:
                        break
                    if sz == 0:
                        self.rfile.readline()  # trailing CRLF
                        break
                    chunks.append(self.rfile.read(sz))
                    self.rfile.readline()      # CRLF after each chunk
                body = b"".join(chunks)
            else:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
            if "gzip" in (self.headers.get("Content-Encoding") or "").lower() and body:
                try:
                    body = gzip.decompress(body)
                except Exception:
                    pass
            return body

        def _ingest(self, kind: str):
            body = self._read_body()
            try:
                payload = json.loads(body.decode("utf-8", "replace")) if body else {}
            except Exception:
                payload = {"_raw": body.decode("utf-8", "replace")[:500]}
            with open(out_path, "a") as f:
                f.write(json.dumps({"kind": kind, "payload": payload}) + "\n")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            if self.path.endswith("/v1/logs"):
                self._ingest("logs")
            elif self.path.endswith("/v1/metrics"):
                self._ingest("metrics")
            elif self.path.endswith("/v1/traces"):
                self._ingest("traces")
            else:
                self._ingest("other")

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
    return H


def serve(port: int, out_path: str, host: str = "127.0.0.1") -> int:
    """Serve forever, appending every received OTLP envelope to ``out_path``. Blocks the caller."""
    open(out_path, "a").close()  # ensure the file exists even before the first request
    srv = ThreadingHTTPServer((host, port), make_handler(out_path))
    print(f"[otel-sink] listening on {host}:{port} -> {out_path}", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aet otel-sink", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True, help="TCP port to listen on")
    ap.add_argument("--out", required=True, help="Output JSONL path (one line per OTLP envelope)")
    ap.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = ap.parse_args(argv)
    return serve(args.port, args.out, host=args.host)


if __name__ == "__main__":
    sys.exit(main())
