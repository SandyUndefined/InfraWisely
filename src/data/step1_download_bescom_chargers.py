"""Download the official BESCOM EV charger list PDF."""

from __future__ import annotations

from pathlib import Path

import requests


PDF_URL = "https://bescom.karnataka.gov.in/storage/pdf-files/EV/Chargerlist.pdf"
OUTPUT_PATH = Path("data/raw/bescom_chargers.pdf")
TIMEOUT_SECONDS = 60


def ensure_directories() -> None:
    """Create required data folders if they do not already exist."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def download_pdf(url: str = PDF_URL, output_path: Path = OUTPUT_PATH) -> Path:
    """Download the BESCOM charger PDF and save it locally."""
    ensure_directories()

    print(f"Download URL: {url}")
    print(f"Output path: {output_path}")

    try:
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download BESCOM charger PDF from {url}: {exc}") from exc

    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise RuntimeError(
            "Downloaded response does not look like a PDF. "
            f"Content-Type was: {content_type or 'unknown'}"
        )

    output_path.write_bytes(response.content)
    file_size_kb = output_path.stat().st_size / 1024

    print(f"HTTP status code: {response.status_code}")
    print(f"File size: {file_size_kb:.2f} KB")
    print("Download complete.")

    return output_path


if __name__ == "__main__":
    download_pdf()
