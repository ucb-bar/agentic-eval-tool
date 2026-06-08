from pathlib import Path

def validate_json_schema(instance: dict, schema_path: Path) -> list[str]:
    """Validate instance against JSON schema. Returns list of error strings.
    Returns empty list if jsonschema not installed (degrades gracefully)."""
    try:
        import jsonschema
        import json
        with open(schema_path) as f:
            schema = json.load(f)
        errors = []
        validator = jsonschema.Draft7Validator(schema)
        for error in validator.iter_errors(instance):
            errors.append(error.message)
        return errors
    except ImportError:
        return []
    except FileNotFoundError:
        return [f"Schema file not found: {schema_path}"]
    except Exception as e:
        return [f"Schema validation error: {e}"]
