"""Tests for tuner.__version__"""

import importlib
import importlib.metadata
import re
from pathlib import Path

import tuner

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_installed_metadata():
    assert tuner.__version__ == importlib.metadata.version("tuner-livekit-sdk")


def test_version_matches_pyproject():
    match = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.MULTILINE)
    assert match is not None
    assert tuner.__version__ == match.group(1)


def test_version_falls_back_when_not_installed(monkeypatch):
    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    try:
        importlib.reload(tuner)
        assert tuner.__version__ == "0.0.0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(tuner)
