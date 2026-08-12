"""Tests for aet.isolation — sandbox builder, allow-list audit, file-access ledger."""
import json
import re
import shutil
import subprocess

import pytest

from aet.isolation import (
    AuditPolicy,
    SandboxSpec,
    audit_run,
    bwrap_argv,
    file_access_ledger,
    wrap_command,
)
from aet.isolation.sandbox import DNS_DIR


def _txn(*tool_uses):
    """One assistant record with the given tool_use blocks."""
    return json.dumps({"type": "assistant", "message": {"content": list(tool_uses)}})


def _use(name, **inp):
    return {"type": "tool_use", "name": name, "id": inp.pop("_id", "t1"), "input": inp}


class TestSandboxArgv:
    def test_allow_then_deny_order(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        allowed = tmp_path / "corpus"
        allowed.mkdir()
        denied = allowed / "hidden"
        denied.mkdir()
        argv = bwrap_argv(SandboxSpec(workspace=ws, allow=[allowed], deny=[denied]))
        s = " ".join(argv)
        assert f"--bind {ws} {ws}" in s
        assert f"--ro-bind {allowed} {allowed}" in s
        assert argv.index(str(denied)) > argv.index(str(allowed))   # deny applied after allow
        assert "--tmpfs" in argv and str(denied) in argv

    def test_rw_bind_after_allow_and_before_deny(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        home = tmp_path / "home"
        home.mkdir()
        state = home / ".claude"        # writable state dir under an otherwise ro-bound home
        state.mkdir()
        answers = state / "answers"     # a denied sub-path even inside the writable state dir
        answers.mkdir()
        argv = bwrap_argv(
            SandboxSpec(workspace=ws, allow=[home], rw_binds=[state], deny=[answers])
        )
        s = " ".join(argv)
        assert f"--ro-bind {home} {home}" in s               # home granted read-only
        assert f"--bind {state} {state}" in s                # state writable (rw override)
        # rw bind comes AFTER the ro allow of its parent so it wins ...
        assert argv.index(str(state)) > argv.index(str(home))
        # ... and BEFORE the deny mask so deny still wins over the rw bind
        assert argv.index(str(answers)) > argv.index(str(state))
        assert "--tmpfs" in argv and str(answers) in argv

    def test_mask_files_and_unsetenv_and_dns(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        gold = tmp_path / "g.yaml"
        gold.write_text("secret")
        argv = bwrap_argv(SandboxSpec(workspace=ws, mask_files=[gold], unsetenv=["FOO", "BAR"], dns=False))
        assert argv[argv.index(str(gold)) - 1] == "/dev/null"       # per-file /dev/null overlay
        assert "--unsetenv" in argv and "FOO" in argv and "BAR" in argv
        assert "/run/systemd/resolve" not in argv                   # dns disabled

    def test_permission_safe_on_locked_dir(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        locked = tmp_path / "locked"
        locked.mkdir()
        (locked / "child").mkdir()
        locked.chmod(0o000)
        try:
            argv = bwrap_argv(SandboxSpec(workspace=ws, deny=[locked / "child"]))
            assert str(locked / "child") in argv                    # treated as present -> masked
        finally:
            locked.chmod(0o755)

    def test_wrap_command_shape(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        cmd = wrap_command("echo hi", SandboxSpec(workspace=ws), env_prefix="export X=1;")
        assert cmd.startswith("bwrap ")
        assert "bash -c 'export X=1; echo hi'" in cmd

    @pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not available")
    def test_functional_allow_visible_deny_masked(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "input.txt").write_text("ok")
        secret = corpus / "answers"
        secret.mkdir()
        (secret / "key").write_text("42")
        spec = SandboxSpec(workspace=ws, allow=[corpus], deny=[secret], tmpfs=["/tmp"])
        probe = (
            f'test -e {corpus}/input.txt && echo INPUT_OK; '
            f'test "$(ls {secret} 2>/dev/null | wc -l)" = 0 && echo ANSWERS_MASKED'
        )
        out = subprocess.run(
            ["bash", "-c", wrap_command(probe, spec)], capture_output=True, text=True, timeout=60
        ).stdout
        assert "INPUT_OK" in out
        assert "ANSWERS_MASKED" in out


class TestSandboxNetEnvGit:
    """Three ways a filesystem allow-list leaks anyway: the network, the environment, and .git."""

    def test_unshare_net_is_opt_in(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        assert "--unshare-net" not in bwrap_argv(SandboxSpec(workspace=ws))
        assert "--unshare-net" in bwrap_argv(SandboxSpec(workspace=ws, unshare_net=True))

    def test_dns_bind_is_dropped_when_the_network_is_unshared(self, tmp_path):
        """Binding the resolver stub into a netns with no interfaces resolves nothing. Leaving the
        bind in would suggest the sandbox has networking when it does not."""
        ws = tmp_path / "ws"
        ws.mkdir()
        s = " ".join(bwrap_argv(SandboxSpec(workspace=ws, unshare_net=True, dns=True)))
        assert DNS_DIR not in s

    def test_clearenv_precedes_every_setenv(self, tmp_path):
        """--clearenv after a --setenv would wipe it. Order is the whole correctness argument."""
        ws = tmp_path / "ws"
        ws.mkdir()
        argv = bwrap_argv(SandboxSpec(workspace=ws, clearenv=True, env_allow=["PATH", "HOME"]))
        assert "--clearenv" in argv
        assert all(argv.index("--clearenv") < i
                   for i, a in enumerate(argv) if a == "--setenv")

    def test_clearenv_exports_only_what_was_named(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setenv("KEEP_ME", "yes")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
        argv = bwrap_argv(SandboxSpec(workspace=ws, clearenv=True, env_allow=["KEEP_ME"]))
        s = " ".join(argv)
        assert "--setenv KEEP_ME yes" in s
        assert "sk-secret" not in s, "an unnamed variable must not survive --clearenv"

    def test_clearenv_skips_names_that_are_not_set(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.delenv("DEFINITELY_UNSET", raising=False)
        argv = bwrap_argv(SandboxSpec(workspace=ws, clearenv=True, env_allow=["DEFINITELY_UNSET"]))
        assert "DEFINITELY_UNSET" not in argv, "exporting an empty value is not the same as absence"

    def test_mask_git_covers_workspace_and_granted_roots(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / ".git").mkdir(parents=True)
        corpus = tmp_path / "corpus"
        (corpus / ".git").mkdir(parents=True)
        argv = bwrap_argv(SandboxSpec(workspace=ws, allow=[corpus], mask_git=True))
        s = " ".join(argv)
        assert f"--tmpfs {ws / '.git'}" in s
        assert f"--tmpfs {corpus / '.git'}" in s

    def test_mask_git_finds_nested_submodules(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        corpus = tmp_path / "corpus"
        (corpus / "third_party" / "dep" / ".git").mkdir(parents=True)
        s = " ".join(bwrap_argv(SandboxSpec(workspace=ws, allow=[corpus], mask_git=True)))
        assert f"--tmpfs {corpus / 'third_party' / 'dep' / '.git'}" in s

    def test_mask_git_is_applied_after_the_allow_binds(self, tmp_path):
        """Same deny-wins ordering as `deny`: a granted repo root must not re-expose its history."""
        ws = tmp_path / "ws"
        ws.mkdir()
        corpus = tmp_path / "corpus"
        (corpus / ".git").mkdir(parents=True)
        argv = bwrap_argv(SandboxSpec(workspace=ws, allow=[corpus], mask_git=True))
        assert argv.index(str(corpus / ".git")) > argv.index(str(corpus))

    def test_mask_git_is_off_by_default(self, tmp_path):
        ws = tmp_path / "ws"
        (ws / ".git").mkdir(parents=True)
        assert str(ws / ".git") not in bwrap_argv(SandboxSpec(workspace=ws))

    @pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not available")
    def test_functional_git_history_is_unreadable(self, tmp_path):
        """The failure this exists to prevent: a file withheld from the working tree, still readable
        from the history of a granted checkout."""
        ws = tmp_path / "ws"
        ws.mkdir()
        corpus = tmp_path / "corpus"
        (corpus / ".git").mkdir(parents=True)
        (corpus / ".git" / "answer").write_text("42")
        (corpus / "input.txt").write_text("ok")
        spec = SandboxSpec(workspace=ws, allow=[corpus], mask_git=True)
        probe = (
            f'test -e {corpus}/input.txt && echo INPUT_OK; '
            f'test "$(ls {corpus}/.git 2>/dev/null | wc -l)" = 0 && echo GIT_MASKED'
        )
        out = subprocess.run(
            ["bash", "-c", wrap_command(probe, spec)], capture_output=True, text=True, timeout=60
        ).stdout
        assert "INPUT_OK" in out
        assert "GIT_MASKED" in out

    @pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap not available")
    def test_functional_env_is_empty_but_for_the_allowlist(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        ws.mkdir()
        monkeypatch.setenv("SECRET_TOKEN", "sk-do-not-leak")
        spec = SandboxSpec(workspace=ws, clearenv=True, env_allow=["PATH"])
        out = subprocess.run(
            ["bash", "-c", wrap_command("echo TOKEN=[$SECRET_TOKEN]; test -n \"$PATH\" && echo PATH_OK", spec)],
            capture_output=True, text=True, timeout=60,
        ).stdout
        assert "TOKEN=[]" in out
        assert "PATH_OK" in out


class TestAudit:
    def _policy(self):
        return AuditPolicy(
            cheats={"golden": re.compile(r"golden\.yaml"), "oracle": re.compile(r"import\s+oracle\b")},
            contaminants={"other_project": re.compile(r"/other/[^ \"']+")},
            warns={"oracle_src": re.compile(r"oracle\.py")},
        )

    def test_clean_run(self, tmp_path):
        rd = tmp_path / "run"
        (rd / "rounds").mkdir(parents=True)
        (rd / "rounds" / "round_00.transcript.jsonl").write_text(
            _txn(_use("Bash", command="ls input/ && python build.py"))
        )
        r = audit_run(rd, self._policy())
        assert not r["disqualified"] and r["isolation_clean"] and not r["warnings"]

    def test_hard_cheat_disqualifies(self, tmp_path):
        rd = tmp_path / "run"
        (rd / "rounds").mkdir(parents=True)
        (rd / "rounds" / "round_00.transcript.jsonl").write_text(
            _txn(_use("Bash", command="cat corpus/golden.yaml"))
        )
        r = audit_run(rd, self._policy())
        assert r["disqualified"] and "golden" in r["cheat_hits"]

    def test_soft_contaminant_and_warn(self, tmp_path):
        rd = tmp_path / "run"
        (rd / "rounds").mkdir(parents=True)
        (rd / "rounds" / "round_00.transcript.jsonl").write_text(
            _txn(_use("Read", file_path="/other/proj/notes.md"), _use("Bash", command="less oracle.py"))
        )
        r = audit_run(rd, self._policy())
        assert not r["disqualified"]
        assert not r["isolation_clean"] and "other_project" in r["out_of_scope_reads"]
        assert r["warnings"].get("oracle_src") == 1


class TestLedger:
    def test_enumerates_and_classifies(self, tmp_path):
        rd = tmp_path / "run"
        (rd / "rounds").mkdir(parents=True)
        recs = [
            _txn(_use("Read", _id="r1", file_path="src/a.py")),
            _txn(_use("Write", _id="w1", file_path="out/b.py", content="xyz")),
            _txn(_use("Bash", _id="b1", command="cat /other/secret.txt && echo done")),
        ]
        (rd / "rounds" / "round_00.transcript.jsonl").write_text("\n".join(recs))

        def classify(p):
            return "out_of_scope" if p.startswith("/other") else "in_scope"

        ledger = file_access_ledger(rd, classify=classify)
        assert ledger["files_read"] == ["src/a.py"]
        assert ledger["files_written"] == ["out/b.py"]
        assert ledger["n_bash"] == 1
        assert any(
            "/other/secret.txt" in r["path"]
            for e in ledger["out_of_scope_events"]
            for r in e.get("refs", [])
        )
