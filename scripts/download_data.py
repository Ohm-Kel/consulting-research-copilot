"""Download the five athletic apparel & footwear annual reports into data/.

Run: python scripts/download_data.py
"""

from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# SEC asks automated clients to identify themselves in the User-Agent.
USER_AGENT = "consulting-copilot research project admin@example.com"

REPORTS: dict[str, str] = {
    "Nike_FY2025_10K.pdf": "https://s1.q4cdn.com/806093406/files/doc_financials/2025/ar/Nike-Inc-2025_10K.pdf",
    # Lululemon and Under Armour's designed annual reports use fonts without a text
    # mapping (pypdf extracts gibberish), so we use print copies of their 10-K filings.
    "Lululemon_FY2024_10K.pdf": "https://stocklight.com/stocks/us/nasdaq-lulu/lululemon-athletica/annual-reports/nasdaq-lulu-2025-10K-25779399.pdf",
    "UnderArmour_FY2025_10K.pdf": "https://stocklight.com/stocks/us/nyse-ua/under-armour-inc-c/annual-reports/nyse-ua-2025-10K-25977723.pdf",
    "Columbia_FY2024_10K.pdf": "https://investor.columbia.com/sec-filings/annual-reports/content/0001050797-25-000023/0001050797-25-000023.pdf",
    "Deckers_FY2025_AR.pdf": "https://www.sec.gov/Archives/edgar/data/910521/000091052125000034/deckglossyannualreportweb_.pdf",
}


def download(filename: str, url: str) -> None:
    """Download one report unless it is already present."""
    target = DATA_DIR / filename
    if target.exists():
        print(f"skip   {filename} (already downloaded)")
        return
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as response:
        target.write_bytes(response.read())
    print(f"saved  {filename} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name, link in REPORTS.items():
        download(name, link)
