"""Cassette storage — save and load agent traces as YAML files.

Cassettes are the on-disk format for recorded agent traces. They're
human-readable YAML files that can be committed to version control.

Naming follows the VCR.py convention: cassettes live in a cassettes/
directory next to the test file, named after the test function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from trace_ops._types import Trace


class CassetteNotFoundError(FileNotFoundError):
    """Raised when a cassette file doesn't exist during replay."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Cassette not found: {path}\n"
            f"Run with --record to create it, or check the cassette path."
        )


class CassetteMismatchError(AssertionError):
    """Raised when replay diverges from the recorded cassette."""

    def __init__(
        self,
        message: str,
        expected_event: dict[str, Any] | None = None,
        actual_event: dict[str, Any] | None = None,
    ) -> None:
        self.expected_event = expected_event
        self.actual_event = actual_event
        super().__init__(message)


def _yaml_representer_str(dumper: yaml.Dumper, data: str) -> Any:
    """Use literal block style for multi-line strings."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _get_dumper() -> type:
    """Get a YAML dumper with custom representers."""
    dumper = yaml.Dumper
    dumper.add_representer(str, _yaml_representer_str)
    return dumper


def save_cassette(trace: Trace, path: str | Path) -> Path:
    """Save a trace as a YAML cassette file.

    Args:
        trace: The recorded trace to save.
        path: Path to the cassette file.

    Returns:
        The path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = trace.to_dict()

    # Redact sensitive headers/keys before saving
    data = _redact_sensitive(data)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            Dumper=_get_dumper(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    return path


def load_cassette(path: str | Path) -> Trace:
    """Load a trace from a YAML cassette file.

    Args:
        path: Path to the cassette file.

    Returns:
        The deserialized Trace.

    Raises:
        CassetteNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise CassetteNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Trace.from_dict(data)


def _redact_sensitive(data: dict[str, Any]) -> dict[str, Any]:
    """Redact API keys, tokens, and other sensitive values from trace data.

    Redacts in-place and returns the same dict.
    """
    sensitive_keys = {
        "api_key", "api-key", "authorization", "x-api-key",
        "openai-api-key", "anthropic-api-key", "bearer",
    }

    def _redact_dict(d: Any) -> Any:
        if isinstance(d, dict):
            return {
                k: ("[REDACTED]" if k.lower() in sensitive_keys else _redact_dict(v))
                for k, v in d.items()
            }
        elif isinstance(d, list):
            return [_redact_dict(item) for item in d]
        elif isinstance(d, str):
            # Redact strings that look like API keys
            if d.startswith(("sk-", "sk-ant-", "xai-", "gsk_")) and len(d) > 20:
                return f"{d[:8]}...[REDACTED]"
            return d
        return d

    return _redact_dict(data)


def cassette_path_for_test(
    test_file: str | Path,
    test_name: str,
    cassette_dir: str = "cassettes",
) -> Path:
    """Compute the default cassette path for a test function.

    Convention: cassettes/{module_name}/{test_name}.yaml
    """
    test_file = Path(test_file)
    module_name = test_file.stem  # e.g., "test_my_agent"
    safe_name = test_name.replace("[", "_").replace("]", "_").replace("/", "_")
    return test_file.parent / cassette_dir / module_name / f"{safe_name}.yaml"
