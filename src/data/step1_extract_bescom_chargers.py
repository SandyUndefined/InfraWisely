"""Extract and clean BESCOM EV charger records from the downloaded PDF."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pdfplumber


PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.area_mapper import infer_area


PDF_PATH = Path("data/raw/bescom_chargers.pdf")
RAW_ROWS_PATH = Path("data/interim/bescom_chargers_raw_rows.csv")
CLEAN_PATH = Path("data/processed/bescom_chargers_clean.csv")
AREA_COUNTS_PATH = Path("data/processed/charger_counts_by_area.csv")
REPORT_PATH = Path("reports/step1_data_quality_report.md")
SOURCE_PDF_URL = "https://bescom.karnataka.gov.in/storage/pdf-files/EV/Chargerlist.pdf"

SERIAL_ROW_PATTERN = re.compile(r"^\s*(\d{1,4})[\).\-\s]+(.+)$")
CHARGER_CODE_PATTERN = re.compile(r"\b(BES[A-Z0-9/-]{3,})\b", flags=re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")

MODEL_KEYWORDS = (
    "GB/TIECACCOMBO",
    "CCSCHADEMOCOMBO",
    "IECAC10KW",
    "EVRE-WX-3.3KW",
    "PENTA IEC 3.3KW",
    "PENTA IEC 3.3KW",
    "PENTA",
    "CHADEMO",
    "COMBO",
    "GB/T",
    "CCS",
    "IEC",
    "AC",
    "DC",
    "KW",
)


def ensure_directories() -> None:
    """Create required output folders."""
    for path in (RAW_ROWS_PATH, CLEAN_PATH, AREA_COUNTS_PATH, REPORT_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    Path("notebooks").mkdir(parents=True, exist_ok=True)
    Path("src/data").mkdir(parents=True, exist_ok=True)
    Path("src/utils").mkdir(parents=True, exist_ok=True)


def normalize_text(value: str) -> str:
    """Normalize spacing and common PDF extraction artifacts."""
    if not isinstance(value, str):
        return ""
    cleaned = value.replace("\u00a0", " ")
    cleaned = SPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip(" -,\t")


def extract_raw_rows(pdf_path: Path = PDF_PATH) -> pd.DataFrame:
    """Extract rows that start with a serial number from all PDF pages."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Missing PDF at {pdf_path}. Run python src/data/step1_download_bescom_chargers.py first."
        )

    raw_rows: list[dict[str, object]] = []

    print(f"Reading PDF: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Total pages found: {len(pdf.pages)}")
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                for line in text.splitlines():
                    line = normalize_text(line)
                    match = SERIAL_ROW_PATTERN.match(line)
                    if match:
                        raw_rows.append(
                            {
                                "sl_no": int(match.group(1)),
                                "raw_text": line,
                                "source_page": page_number,
                            }
                        )
    except Exception as exc:
        raise RuntimeError(f"Failed while extracting rows from {pdf_path}: {exc}") from exc

    raw_df = pd.DataFrame(raw_rows, columns=["sl_no", "raw_text", "source_page"])
    raw_df.to_csv(RAW_ROWS_PATH, index=False)
    print(f"Raw rows extracted: {len(raw_df)}")
    print(f"Saved raw rows to: {RAW_ROWS_PATH}")
    return raw_df


def detect_charger_model(text_after_code: str) -> tuple[str, str]:
    """Split station text and charger model from the text after charger code."""
    text_after_code = normalize_text(text_after_code)
    if not text_after_code:
        return "", ""

    upper_text = text_after_code.upper()
    candidate_positions: list[int] = []

    for keyword in MODEL_KEYWORDS:
        position = upper_text.find(keyword.upper())
        if position >= 0:
            candidate_positions.append(position)

    if not candidate_positions:
        return text_after_code, ""

    model_start = min(candidate_positions)
    station_name = normalize_text(text_after_code[:model_start])
    charger_model = normalize_text(text_after_code[model_start:])
    return station_name, charger_model


def parse_raw_row(row: pd.Series) -> dict[str, object]:
    """Parse one raw charger row into structured fields."""
    raw_text = normalize_text(str(row["raw_text"]))
    body_match = SERIAL_ROW_PATTERN.match(raw_text)
    body = normalize_text(body_match.group(2) if body_match else raw_text)

    charger_code = ""
    station_name = ""
    charger_model = ""
    parse_status = "failed"

    code_match = CHARGER_CODE_PATTERN.search(body)
    if code_match:
        charger_code = code_match.group(1).upper()
        text_after_code = normalize_text(body[code_match.end() :])
        station_name, charger_model = detect_charger_model(text_after_code)
    else:
        station_name, charger_model = detect_charger_model(body)

    station_name = normalize_text(station_name)
    charger_model = normalize_text(charger_model)
    area = infer_area(station_name)

    if charger_code and station_name and charger_model:
        parse_status = "parsed"
    elif charger_code or station_name or charger_model:
        parse_status = "partial"

    return {
        "sl_no": row["sl_no"],
        "charger_code": charger_code,
        "station_name": station_name,
        "charger_model": charger_model,
        "area": area,
        "source_page": row["source_page"],
        "raw_text": raw_text,
        "parse_status": parse_status,
    }


def clean_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw rows and save the cleaned charger dataset."""
    cleaned_records = [parse_raw_row(row) for _, row in raw_df.iterrows()]
    clean_df = pd.DataFrame(
        cleaned_records,
        columns=[
            "sl_no",
            "charger_code",
            "station_name",
            "charger_model",
            "area",
            "source_page",
            "raw_text",
            "parse_status",
        ],
    )
    clean_df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned rows: {len(clean_df)}")
    print(f"Saved cleaned charger data to: {CLEAN_PATH}")
    return clean_df


def create_area_counts(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Create charger counts by inferred area."""
    area_counts = (
        clean_df.groupby("area", dropna=False)
        .size()
        .reset_index(name="existing_chargers")
        .sort_values(["existing_chargers", "area"], ascending=[False, True])
    )
    area_counts.to_csv(AREA_COUNTS_PATH, index=False)
    print(f"Saved charger counts by area to: {AREA_COUNTS_PATH}")
    return area_counts


def create_quality_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, area_counts: pd.DataFrame) -> None:
    """Write a Markdown data quality report for Step 1."""
    status_counts = Counter(clean_df["parse_status"])
    unknown_area_count = int((clean_df["area"] == "Unknown").sum())
    top_areas = area_counts.head(10)

    top_area_lines = [
        f"| {record.area} | {record.existing_chargers} |"
        for record in top_areas.itertuples(index=False)
    ]
    top_area_table = "\n".join(["| Area | Existing chargers |", "|---|---:|", *top_area_lines])

    report = f"""# Step 1 Data Quality Report

## Step Name

BESCOM EV charger PDF extraction, cleaning, area inference, and charger count aggregation.

## Source

Source PDF URL: {SOURCE_PDF_URL}

## Extraction Summary

- Raw rows extracted count: {len(raw_df)}
- Clean parsed rows count: {len(clean_df)}
- Parsed count: {status_counts.get("parsed", 0)}
- Partial count: {status_counts.get("partial", 0)}
- Failed count: {status_counts.get("failed", 0)}
- Unknown area count: {unknown_area_count}

## Top 10 Areas By Charger Count

{top_area_table}

## Notes For ML

- `existing_chargers` becomes a feature for EV demand prediction.
- `charger_counts_by_area.csv` becomes an input for infrastructure gap scoring.
- `Unknown` area rows are retained so they can be manually reviewed or improved with better geospatial matching later.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved data quality report to: {REPORT_PATH}")


def print_summary(clean_df: pd.DataFrame, area_counts: pd.DataFrame) -> None:
    """Print useful extraction and cleaning outputs for quick inspection."""
    print("\nParse status value counts:")
    print(clean_df["parse_status"].value_counts(dropna=False).to_string())

    print("\nFirst 10 cleaned rows:")
    print(clean_df.head(10).to_string(index=False))

    print("\nCharger counts by area:")
    print(area_counts.to_string(index=False))


def main() -> None:
    """Run Step 1 extraction, cleaning, aggregation, and reporting."""
    ensure_directories()
    raw_df = extract_raw_rows()
    clean_df = clean_rows(raw_df)
    area_counts = create_area_counts(clean_df)
    create_quality_report(raw_df, clean_df, area_counts)
    print_summary(clean_df, area_counts)
    print("\nStep 1 extraction pipeline complete.")


if __name__ == "__main__":
    main()
