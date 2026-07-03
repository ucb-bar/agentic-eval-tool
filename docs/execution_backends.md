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

## CLI Flag Reference

| Flag | Values | Default | Applies to |
|------|--------|---------|------------|
| `--execution` | `local` | `local` | `init-run`, `validate`, `run-suite` |

The local executor is the only implemented backend.
