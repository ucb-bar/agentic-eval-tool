from pathlib import Path
import yaml

def load_yaml(path: Path) -> dict:
    """Load YAML file, return empty dict if file missing or empty."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

def dump_yaml(data: dict, path: Path) -> None:
    """Dump dict to YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=True, allow_unicode=True)
