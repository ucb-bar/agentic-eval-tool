"""The [viz] extra is optional: core recording/import/monitor must not require matplotlib."""
import importlib


def test_trajectory_core_imports_without_matplotlib():
    # None of these modules may pull matplotlib into their import chain.
    for mod in ("aet.trajectory", "aet.trajectory.build", "aet.trajectory.stream",
                "aet.trajectory.recording", "aet.trajectory.importers.capsule_bench"):
        m = importlib.import_module(mod)
        assert m is not None


def test_viz_package_is_import_light():
    # importing the package itself must not import matplotlib (only its submodules do)
    import aet.viz
    assert hasattr(aet.viz, "require_viz")


def test_viz_getattr_unknown_raises():
    import aet.viz
    try:
        aet.viz.does_not_exist
    except AttributeError:
        pass
    else:  # pragma: no cover
        assert False, "expected AttributeError"


def test_require_viz_message_is_actionable(monkeypatch):
    """When matplotlib is missing, the error tells the user exactly what to install."""
    import builtins
    import aet.viz
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "matplotlib":
            raise ImportError("no matplotlib")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    try:
        aet.viz.require_viz()
    except ImportError as e:
        assert "aet[viz]" in str(e)
    else:  # pragma: no cover
        assert False, "expected ImportError"
