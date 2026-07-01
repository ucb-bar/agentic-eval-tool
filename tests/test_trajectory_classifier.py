"""ActivityClassifier: default categories, pluggable long-wait rules, repo-agnostic core."""
from aet.trajectory.classify import (
    ActivityClassifier, ActivityConfig, LongWaitRule, capsule_bench_config,
)


def test_default_categories():
    c = ActivityClassifier()
    assert c.classify("Read", {"file_path": "/x"})[0] == "read"
    assert c.classify("Edit", {"file_path": "/x"})[0] == "write"
    assert c.classify("Write", {})[0] == "write"
    assert c.classify("NotebookEdit", {})[0] == "write"
    assert c.classify("Bash", {"command": "ls"})[0] == "bash"
    # unknown / MCP tools fall back to the generic bash lane
    assert c.classify("mcp__foo__bar", {})[0] == "bash"


def test_default_weights():
    c = ActivityClassifier()
    assert c.classify("Edit", {})[1] == 1.4
    assert c.classify("Bash", {"command": "ls"})[1] == 1.2
    assert c.weight_for("tool") == 28.0
    assert c.weight_for("think") == 1.0


def test_long_wait_rule_matches_verilator_bash():
    c = ActivityClassifier(capsule_bench_config())
    cat, w = c.classify("Bash", {"command": "run.py --sim verilator --design gemmini"})
    assert cat == "tool" and w == 28.0
    # an ordinary shell call stays in the bash lane
    assert c.classify("Bash", {"command": "ls -la"})[0] == "bash"


def test_circt_rules_added_only_when_requested():
    plain = ActivityClassifier(capsule_bench_config(circt=False))
    circt = ActivityClassifier(capsule_bench_config(circt=True))
    inp = {"command": "python gen_isa.py --rtl_facts facts.json"}
    assert plain.classify("Bash", inp)[0] == "bash"      # no CIRCT rule → ordinary bash
    assert circt.classify("Bash", inp)[0] == "tool"      # CIRCT rule active → long wait


def test_core_is_repo_agnostic_rules_are_data_not_hardcoded():
    """With an empty config, a verilator Bash is ordinary bash — the 'tool' reclassification
    comes ONLY from a supplied rule, never from baked-in harness knowledge."""
    bare = ActivityClassifier(ActivityConfig())          # no long-wait rules
    assert bare.classify("Bash", {"command": "--sim verilator"})[0] == "bash"


def test_config_round_trip():
    cfg = capsule_bench_config(circt=True)
    back = ActivityConfig.from_dict(cfg.to_dict())
    assert back.to_dict() == cfg.to_dict()
    assert len(back.long_wait_rules) == 2


def test_custom_long_wait_rule():
    cfg = ActivityConfig(long_wait_rules=[LongWaitRule("Bash", ["pytest"], category="tool", weight=9.0)])
    c = ActivityClassifier(cfg)
    cat, w = c.classify("Bash", {"command": "pytest tests/"})
    assert cat == "tool" and w == 9.0
