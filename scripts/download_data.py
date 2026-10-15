"""Download the five athletic apparel & footwear annual reports into data/.

Every file is checked against a SHA-256 checksum, so the evaluation always runs
on exactly the documents it was built from. If an original host refuses the
download, the script tries a mirror copy; the checksum proves it is the same file.

Run: python scripts/download_data.py
"""

import hashlib
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# SEC asks automated clients to identify themselves in the User-Agent.
USER_AGENT = "consulting-research-copilot admin@example.com"

# Byte-identical copies of the five reports, tried when an original host refuses the download
# (stocklight.com began returning 403 to CI runners in October 2026). Set the variable to an
# empty string to use the original hosts only.
MIRROR = os.getenv(
    "COPILOT_REPORTS_MIRROR", "https://github.com/Ohm-Kel/consulting-research-copilot/releases/download/reports-v1"
)

# file name -> (URL, SHA-256 of the file the evaluation set was built from)
REPORTS: dict[str, tuple[str, str]] = {
    "Nike_FY2025_10K.pdf": (
        "https://s1.q4cdn.com/806093406/files/doc_financials/2025/ar/Nike-Inc-2025_10K.pdf",
        "7b1b8ae4da5893c7ab2fdf8a74008e9e0a7afb9f58f13fa14c82cf80e44ad9fc",
    ),
    # Lululemon and Under Armour's designed annual reports use fonts without a text
    # mapping (pypdf extracts gibberish), so we use print copies of their 10-K filings.
    "Lululemon_FY2024_10K.pdf": (
        "https://stocklight.com/stocks/us/nasdaq-lulu/lululemon-athletica/annual-reports/nasdaq-lulu-2025-10K-25779399.pdf",
        "ca60d873c6fcb67d2b58f7c723565ef2ec8f0e37b6359ebc7aa8ed7d6f4be2a8",
    ),
    "UnderArmour_FY2025_10K.pdf": (
        "https://stocklight.com/stocks/us/nyse-ua/under-armour-inc-c/annual-reports/nyse-ua-2025-10K-25977723.pdf",
        "851da2196b285fb523bdde8cd93e4485da3bf8d2f3671a4079e0a0f63473729d",
    ),
    "Columbia_FY2024_10K.pdf": (
        "https://investor.columbia.com/sec-filings/annual-reports/content/0001050797-25-000023/0001050797-25-000023.pdf",
        "853a6c92ad8bc8360a00b0a2ed065f5d2606217950c4e10fdadc4bad44aaa640",
    ),
    "Deckers_FY2025_AR.pdf": (
        "https://www.sec.gov/Archives/edgar/data/910521/000091052125000034/deckglossyannualreportweb_.pdf",
        "9900996aa783da230725934bf3e209272a632e6b679118b43f4b8e992176d017",
    ),
}


class ChecksumError(RuntimeError):
    """A report does not match the checksum the evaluation set was built from."""


class DownloadError(RuntimeError):
    """No source delivered a report with the expected checksum."""


def sha256(path: Path) -> str:
    """Hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path: Path, expected: str) -> None:
    """Raise ChecksumError if `path` does not have the expected SHA-256."""
    actual = sha256(path)
    if actual != expected:
        raise ChecksumError(
            f"{path.name} has SHA-256 {actual[:12]}..., expected {expected[:12]}.... The source file has "
            "changed, so evaluation results would no longer be comparable. Delete it to re-download, or "
            "update the checksum and the evaluation set together."
        )


def sources(filename: str, url: str) -> list[str]:
    """The original URL first, then the mirror copy if a mirror is configured."""
    return [url, f"{MIRROR.rstrip('/')}/{filename}"] if MIRROR else [url]


def fetch(url: str) -> bytes:
    """Download `url` and return its bytes."""
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as response:
        data: bytes = response.read()
        return data


def download(filename: str, url: str, expected: str, data_dir: Path = DATA_DIR) -> None:
    """Download one report unless it is already present, trying each source until one passes the checksum."""
    target = data_dir / filename
    if target.exists():
        verify(target, expected)
        print(f"ok     {filename} (already downloaded, checksum verified)")
        return
    problems: list[str] = []
    for source in sources(filename, url):
        try:
            target.write_bytes(fetch(source))
            verify(target, expected)
        except (URLError, TimeoutError, ChecksumError) as exc:  # HTTPError is a URLError
            target.unlink(missing_ok=True)  # never leave a partial or unexpected file behind
            problems.append(f"{urlparse(source).netloc}: {exc}")
            continue
        size = target.stat().st_size / 1e6
        print(f"saved  {filename} from {urlparse(source).netloc} ({size:.1f} MB, checksum verified)")
        return
    raise DownloadError(f"could not get {filename}: " + "; ".join(problems))


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    try:
        for name, (link, checksum) in REPORTS.items():
            download(name, link, checksum)
    except (ChecksumError, DownloadError) as exc:
        sys.exit(f"error  {exc}")
