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


def test_download_falls_back_to_the_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from urllib.error import HTTPError

    script = load_script()
    content = b"the exact report"
    tried: list[str] = []

    def fake_fetch(url: str) -> bytes:
        tried.append(url)
        if "original.example" in url:
            raise HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)
        return content

    monkeypatch.setattr(script, "fetch", fake_fetch)
    monkeypatch.setattr(script, "MIRROR", "https://mirror.example/reports")
    script.download("report.pdf", "https://original.example/r.pdf", hashlib.sha256(content).hexdigest(), tmp_path)
    assert tried == ["https://original.example/r.pdf", "https://mirror.example/reports/report.pdf"]
    assert (tmp_path / "report.pdf").read_bytes() == content


def test_download_rejects_a_copy_with_the_wrong_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script()
    monkeypatch.setattr(script, "fetch", lambda url: b"a different file")
    monkeypatch.setattr(script, "MIRROR", "https://mirror.example/reports")
    with pytest.raises(script.DownloadError, match="original.example.*mirror.example"):
        script.download("report.pdf", "https://original.example/r.pdf", hashlib.sha256(b"real").hexdigest(), tmp_path)
    assert not (tmp_path / "report.pdf").exists()  # nothing unverified is left behind


def test_no_mirror_means_original_host_only() -> None:
    script = load_script()
    script.MIRROR = ""
    assert script.sources("report.pdf", "https://original.example/r.pdf") == ["https://original.example/r.pdf"]
