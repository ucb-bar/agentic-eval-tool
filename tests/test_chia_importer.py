import json

from aet.trajectory.export import export_agent_profiles
from aet.trajectory.importers.chia import import_chia


def _event(kind, ts, **fields):
    return {"schema": "chia.agent_profile", "schema_version": "1.0",
            "type": kind, "ts": ts, **fields}


def test_chia_import_preserves_hierarchy_cache_and_activity(tmp_path):
    path = tmp_path / "profile.jsonl"
    events = [
        _event("llm_request", 100.0, request_id="r1", attempt=1, agent_id="root",
               session_id="s",
               provider="openai", model="gpt", status="completed", duration_s=2,
               input_tokens=10, output_tokens=4, cache_read_tokens=80,
               cache_write_tokens=10, reasoning_tokens=2),
        _event("tool_activity", 101.0, agent_id="root", tool_name="Bash",
               category="tool", status="completed", duration_s=0.5),
        _event("llm_request", 110.0, request_id="r2", attempt=2, agent_id="child",
               session_id="s",
               parent_agent_id="root", provider="openai", model="gpt", status="completed",
               retry=True, duration_s=1, input_tokens=12, output_tokens=3),
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    traj = import_chia(path, context_windows={"gpt": 200}, default_cache_ttl_s=5)
    assert traj.final_input_tokens == 22
    assert traj.final_cache_read_tokens == 80
    assert traj.final_cache_creation_tokens == 10
    assert traj.final_reasoning_tokens == 2
    assert traj.final_cost_usd is None and traj.rounds[0].cost_usd is None
    assert traj.inferences[0].context_occupancy_ratio == 0.5
    assert traj.inferences[1].ttl_inference == "probable_expiry"
    assert traj.per_agent_rollup()["child"]["parent_agent_id"] == "root"
    assert traj.bands[0].tool_name == "Bash"


def test_agent_exports_label_derived_fields(tmp_path):
    path = tmp_path / "profile.jsonl"
    path.write_text(json.dumps(_event(
        "llm_request", 100.0, request_id="r", agent_id="a", provider="openai",
        model="gpt", status="completed", duration_s=2, input_tokens=10,
        output_tokens=4, cache_read_tokens=5,
    )) + "\n")
    traj = import_chia(path, context_windows={"gpt": 100})
    csv_path, json_path = export_agent_profiles(traj, tmp_path / "profile")
    assert "context_occupancy_provenance" in csv_path.read_text()
    exported = json.loads(json_path.read_text())
    assert exported["provenance"]["context_occupancy"] == "estimated_not_physical_kv_fullness"
    assert exported["activity_seconds"] == {"model": 2.0}
    assert "reasoning" not in exported["activity_seconds"]
