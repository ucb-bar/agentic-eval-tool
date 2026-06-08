# Ray Execution Backend (Planned)

## Current Status

The `RayExecutor` class exists but is not implemented. Both `submit` and `run_many` raise `NotImplementedError`. Use `--execution local` for all current work.

```
pip install 'aet[ray]'   # installs Ray but the executor still raises NotImplementedError
aet run-suite ... --execution local   # use this
```

## Motivation

The `run-suite` command runs `init-run + validate` for every `(method, seed)` combination sequentially. For large sweeps (many methods, many seeds, expensive validators) this is too slow. The Ray backend will parallelize across combinations while keeping each individual run's tracking and local-write semantics intact.

## Planned Interface

The `RayExecutor` will implement the same `ExecutionBackend` interface as `LocalExecutor`:

```python
class ExecutionBackend:
    def submit(self, spec: RunSpec, fn: Callable) -> Any:
        """Submit a single run spec for execution. Returns a future or result."""
        ...

    def run_many(self, specs: list[RunSpec], fn: Callable) -> list[Any]:
        """Submit all specs and collect results in submission order."""
        ...
```

Planned Ray implementation sketch:

```python
class RayExecutor(ExecutionBackend):
    def __init__(self, ray_address: str | None = None) -> None:
        import ray
        if not ray.is_initialized():
            ray.init(address=ray_address)
        self._ray_address = ray_address

    def submit(self, spec: RunSpec, fn: Callable) -> ray.ObjectRef:
        remote_fn = ray.remote(fn)
        return remote_fn.remote(spec)

    def run_many(self, specs: list[RunSpec], fn: Callable) -> list[Any]:
        remote_fn = ray.remote(fn)
        refs = [remote_fn.remote(spec) for spec in specs]
        return ray.get(refs)
```

## Design Constraints

When implemented, the Ray backend must:

1. Write the local `run_manifest.yaml` and `metrics/` files from within the Ray task, not the driver. Each task is responsible for its own canonical record.
2. Collect tracking warnings per-task and merge them into the driver-side `logs/tracking_warnings.jsonl` after all tasks complete.
3. Not share MLflow or OTel client objects across Ray tasks. Each task constructs its own backend connections.
4. Preserve submission order in the results returned by `run_many`. `ray.get(refs)` preserves order.
5. Report failures per-run, not as a batch failure. A single run failure must not cancel the rest of the sweep.

## Installation

```
pip install 'aet[ray]'
```

This installs `ray` as an optional dependency. The `RayExecutor` class is always importable from `aet.execution`; the `ray` package is only imported at construction time.
