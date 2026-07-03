"""Stress tests — concurrent runs to verify no cross-run pollution or file-write races."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path


from aet.core.run_spec import RunSpec
from aet.core.run_paths import RunPaths
from aet.suites import get_suite
from aet.tracking.run_logger import EvalRunLogger


def _do_run(tmp_root: Path, seed: int) -> dict:
    """One full init-run + validate cycle, isolated in tmp_root/runs/default/<run_id>."""
    run_id = f"stress_seed{seed:03d}"
    spec = RunSpec(
        project="stress",
        suite="default",
        method="stress_method",
        seed=seed,
        project_root=tmp_root,
    )
    paths = RunPaths.from_spec(spec, run_id)
    for d in (paths.run_path, paths.logs, paths.metrics, paths.generated, paths.patches, paths.contracts, paths.artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = EvalRunLogger.start(
        project="stress", suite="default", target="",
        method="stress_method", seed=seed, run_id=run_id,
        run_path=paths.run_path, tracking_mode="local",
    )

    suite = get_suite("default")
    suite.init_run(spec, paths, logger)
    report = suite.validate(spec, paths, logger)
    logger.finish(status=report.get("status", "unknown"))
    return report


class TestConcurrentRuns:
    def test_10_concurrent_default_runs(self, tmp_path):
        """10 concurrent default-suite runs must all succeed with no cross-run pollution."""
        seeds = list(range(10))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_do_run, tmp_path, seed): seed for seed in seeds}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                seed = futures[future]
                results[seed] = future.result()

        assert len(results) == 10
        for seed, report in results.items():
            assert report["status"] in ("pass", "partial"), (
                f"seed={seed} got status={report['status']!r}: {report.get('errors')}"
            )

    def test_run_ids_are_unique(self, tmp_path):
        """Each concurrent run gets its own isolated directory."""
        seeds = list(range(5))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_do_run, tmp_path, seed) for seed in seeds]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        run_dirs = list((tmp_path / "runs" / "default").iterdir())
        assert len(run_dirs) == 5
        names = {d.name for d in run_dirs}
        assert len(names) == 5  # all unique

    def test_no_cross_run_pollution_in_metrics(self, tmp_path):
        """Each run's summary_metrics.json must contain its own seed."""
        seeds = [42, 43, 44]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_do_run, tmp_path, seed) for seed in seeds]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        import json
        for seed in seeds:
            run_id = f"stress_seed{seed:03d}"
            metrics_path = tmp_path / "runs" / "default" / run_id / "metrics" / "summary_metrics.json"
            assert metrics_path.exists(), f"missing metrics for seed={seed}"
            data = json.loads(metrics_path.read_text())
            assert data["seed"] == seed, f"wrong seed in metrics: expected {seed}, got {data['seed']}"


class TestConcurrentTargetgenRuns:
    def test_5_concurrent_targetgen_runs(self, tmp_path):
        """5 concurrent targetgen-suite runs (no external deps) must all complete."""
        def _do_targetgen_run(seed: int) -> dict:
            run_id = f"tg_stress_seed{seed:03d}"
            spec = RunSpec(
                project="stress",
                suite="targetgen",
                method="stress",
                seed=seed,
                target="gemmini",
                project_root=tmp_path,
            )
            paths = RunPaths.from_spec(spec, run_id)
            for d in (paths.run_path, paths.logs, paths.metrics, paths.generated, paths.patches, paths.contracts, paths.artifacts_dir):
                d.mkdir(parents=True, exist_ok=True)

            logger = EvalRunLogger.start(
                project="stress", suite="targetgen", target="gemmini",
                method="stress", seed=seed, run_id=run_id,
                run_path=paths.run_path, tracking_mode="local",
            )
            suite = get_suite("targetgen")
            suite.init_run(spec, paths, logger)
            report = suite.validate(spec, paths, logger)
            logger.finish(status=report.get("overall", "unknown"))
            return report

        seeds = list(range(5))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_do_targetgen_run, seed) for seed in seeds]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 5
        for report in results:
            # targetgen reports overall, not status
            assert "overall" in report
