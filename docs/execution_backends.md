# Execution Backends

`aet` supports two execution backends, selected with the `--execution` flag on `init-run`, `validate`, and `run-suite`.

## LocalExecutor (default)

The local backend runs experiment specs sequentially, in-process. It is the default and the only fully-supported backend.

```
aet run-suite --suite targetgen --methods agent_v1 --seeds 1,2,3 --execution local
```

Properties:
- Sequential: runs are executed one after another in the order `(method, seed)` combinations are generated.
- In-process: each `validate` call runs in the same Python process. No subprocess overhead.
- No dependencies: ships with the base `aet` package.
- Deterministic ordering: the cross-product of `--methods` and `--seeds` is traversed in stable order.

The `LocalExecutor` implementation:

```python
class LocalExecutor(ExecutionBackend):
    def submit(self, spec: RunSpec, fn: Callable) -> Any:
        return fn(spec)

    def run_many(self, specs: list[RunSpec], fn: Callable) -> list[Any]:
        return [fn(spec) for spec in specs]
```

## RayExecutor (skeleton)

The Ray backend is a skeleton. It is not yet implemented. Any call to `submit` or `run_many` raises `NotImplementedError`.

```
aet run-suite --suite targetgen --methods agent_v1 --seeds 1,2,3 --execution ray
# raises: NotImplementedError — use --execution local
```

Attempting to use `--execution ray` with `run-suite` will also raise `NotImplementedError` at the CLI level, before any run is started.

To install the Ray extra (for future use):

```
pip install 'aet[ray]'
```

The `RayExecutor` class will import `ray` at construction time and raise `AetError` if it is not installed, rather than at call time.

See `docs/ray_backend.md` for the planned interface.

## CLI Flag Reference

| Flag | Values | Default | Applies to |
|------|--------|---------|------------|
| `--execution` | `local`, `ray` | `local` | `init-run`, `validate`, `run-suite` |

Use `--execution local` for all current work.
