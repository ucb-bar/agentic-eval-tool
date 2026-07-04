"""Instrumentation path: the OTLP sink captures an envelope to JSONL, and the 'otel' import source
reconstructs a full-fidelity trajectory from it — the self-contained capture→import→plot loop
documented in docs/instrumentation.md, exercised without any repo-specific harness.
"""
from __future__ import annotations

import json
import threading
import urllib.request

from http.server import ThreadingHTTPServer

from aet.tracking.otel_sink import make_handler
from aet.trajectory.importers import get_importer, IMPORTER_REGISTRY
from aet.trajectory.importers.otel import import_otel


def _api_req(seq, t_s, inp, out, cr, cost, dur_ms):
    B = 1_000_000_000_000_000_000
    S = 1_000_000_000
    return {"timeUnixNano": str(B + t_s * S),
            "body": {"stringValue": "claude_code.api_request"},
            "attributes": [{"key": "event.sequence", "value": {"intValue": seq}},
                           {"key": "input_tokens", "value": {"intValue": inp}},
                           {"key": "output_tokens", "value": {"intValue": out}},
                           {"key": "cache_read_tokens", "value": {"intValue": cr}},
                           {"key": "cost_usd", "value": {"doubleValue": cost}},
                           {"key": "duration_ms", "value": {"intValue": dur_ms}},
                           {"key": "model", "value": {"stringValue": "claude-opus-4-8"}}]}


def _logs_envelope(records):
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": records}]}]}


class TestOtelSink:
    def test_sink_appends_posted_envelope_as_jsonl(self, tmp_path):
        out = tmp_path / "otel_logs.jsonl"
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(out)))
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            env = _logs_envelope([_api_req(2, 3, 1200, 300, 0, 0.05, 3000)])
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/logs",
                                         data=json.dumps(env).encode(),
                                         headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=5)
            assert resp.status == 200
        finally:
            srv.shutdown()
            srv.server_close()
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["kind"] == "logs"
        # the exact envelope the importer parses is preserved
        assert rec["payload"]["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["body"][
            "stringValue"] == "claude_code.api_request"


class TestOtelImportSource:
    def test_registered_as_import_source(self):
        assert "otel" in IMPORTER_REGISTRY
        assert get_importer("otel") is import_otel

    def test_import_source_otel_builds_full_fidelity_trajectory(self, tmp_path):
        # write a capture the way the sink would, then import via the registry with a terminal verdict
        out = tmp_path / "otel_logs.jsonl"
        env = _logs_envelope([_api_req(2, 3, 1200, 300, 0, 0.05, 3000),
                              _api_req(3, 12, 40, 900, 120000, 0.42, 6000)])
        out.write_text(json.dumps({"kind": "logs", "payload": env}) + "\n")

        traj = get_importer("otel")(str(out), run_id="run-a", n_passed=5, n_total=5)
        assert traj is not None and traj.points
        # totals are the exact per-turn sums (input/output/cache/cost), no interpolation
        assert traj.final_input_tokens == 1240
        assert traj.final_output_tokens == 1200
        assert traj.final_cache_tokens == 120000
        assert abs(traj.final_cost_usd - 0.47) < 1e-9
        assert traj.milestones and traj.milestones[-1].n_passed == 5
