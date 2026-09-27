"""Report downloads are verified against SHA-256 checksums."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

from copilot import config


def load_script():
    spec = importlib.util.spec_from_file_location("download_data", config.ROOT / "scripts" / "download_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_configured_report_has_a_checksum() -> None:
    reports = load_script().REPORTS
    assert set(reports) == set(config.DOCUMENTS)
    assert all(len(checksum) == 64 for _, checksum in reports.values())


def test_verify_rejects_a_changed_file(tmp_path: Path) -> None:
    script = load_script()
    report = tmp_path / "report.pdf"
    report.write_bytes(b"original")
    script.verify(report, hashlib.sha256(b"original").hexdigest())
    report.write_bytes(b"changed")
    with pytest.raises(script.ChecksumError, match="has changed"):
        script.verify(report, hashlib.sha256(b"original").hexdigest())


def test_local_reports_match_their_checksums(data_available: bool) -> None:
    if not data_available:
        pytest.skip("reports not downloaded")
    script = load_script()
    for name, (_, checksum) in script.REPORTS.items():
        script.verify(config.DATA_DIR / name, checksum)
