"""Create the zone-level ML master dataset for GridCharge AI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CHARGER_COUNTS_PATH = Path("data/processed/charger_counts_by_area.csv")
ZONES_MASTER_PATH = Path("data/processed/zones_master.csv")
REPORT_PATH = Path("reports/step2_zone_master_report.md")
RANDOM_SEED = 42

ZONE_DEFINITIONS = [
    ("Whitefield", 12.9698, 77.7500, "mixed", 90),
    ("Indiranagar", 12.9719, 77.6412, "residential", 85),
    ("Koramangala", 12.9352, 77.6245, "mixed", 88),
    ("Electronic City", 12.8452, 77.6602, "industrial", 82),
    ("MG Road", 12.9756, 77.6068, "commercial", 92),
    ("Jayanagar", 12.9250, 77.5938, "residential", 75),
    ("Hebbal", 13.0358, 77.5970, "mixed", 95),
    ("Yelahanka", 13.1007, 77.5963, "residential", 70),
    ("BTM Layout", 12.9166, 77.6101, "residential", 80),
    ("Rajajinagar", 12.9915, 77.5550, "mixed", 78),
    ("Marathahalli", 12.9591, 77.6974, "mixed", 94),
    ("Banashankari", 12.9255, 77.5468, "residential", 72),
    ("HSR Layout", 12.9116, 77.6473, "residential", 84),
    ("KR Puram", 13.0075, 77.6950, "mixed", 87),
    ("Malleshwaram", 13.0031, 77.5643, "residential", 73),
    ("Sarjapur Road", 12.9081, 77.6812, "mixed", 90),
    ("Peenya", 13.0285, 77.5197, "industrial", 75),
    ("Majestic", 12.9767, 77.5713, "commercial", 96),
    ("Bellandur", 12.9358, 77.6788, "mixed", 93),
    ("Kengeri", 12.9087, 77.4871, "residential", 68),
]

POPULATION_DENSITY_RANGES = {
    "residential": (9000, 18000),
    "commercial": (6000, 14000),
    "mixed": (8000, 17000),
    "industrial": (3000, 9000),
}

EV_MULTIPLIERS = {
    "residential": 0.28,
    "commercial": 0.18,
    "mixed": 0.25,
    "industrial": 0.12,
}

HIGH_GROWTH_ZONES = {
    "Whitefield",
    "Bellandur",
    "Sarjapur Road",
    "Marathahalli",
    "Electronic City",
}
MID_GROWTH_ZONES = {"Indiranagar", "Koramangala", "HSR Layout", "KR Puram", "Hebbal"}


def ensure_directories() -> None:
    """Create required output folders."""
    ZONES_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_charger_counts(path: Path = CHARGER_COUNTS_PATH) -> pd.DataFrame:
    """Read Step 1 charger counts by area."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}. Run Step 1 extraction before Step 2."
        )

    charger_counts = pd.read_csv(path)
    expected_columns = {"area", "existing_chargers"}
    missing_columns = expected_columns.difference(charger_counts.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing_columns)}"
        )

    charger_counts = charger_counts[["area", "existing_chargers"]].copy()
    charger_counts["existing_chargers"] = (
        pd.to_numeric(charger_counts["existing_chargers"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    print(f"Loaded charger counts from: {path}")
    print(f"Charger count rows: {len(charger_counts)}")
    return charger_counts


def growth_rate_for_zone(zone_name: str, zone_type: str, rng: np.random.Generator) -> float:
    """Generate realistic demand growth rates by zone category."""
    if zone_name in HIGH_GROWTH_ZONES:
        low, high = 0.28, 0.38
    elif zone_name in MID_GROWTH_ZONES:
        low, high = 0.20, 0.32
    elif zone_type == "commercial":
        low, high = 0.14, 0.25
    elif zone_type == "industrial":
        low, high = 0.12, 0.22
    else:
        low, high = 0.10, 0.24

    return round(float(rng.uniform(low, high)), 3)


def coverage_level(existing_chargers: int, charger_gap_score: float) -> str:
    """Classify charger coverage from existing chargers and gap score."""
    if existing_chargers == 0:
        return "No Coverage"
    if charger_gap_score > 1200:
        return "Low Coverage"
    if 600 <= charger_gap_score <= 1200:
        return "Medium Coverage"
    return "Good Coverage"


def create_base_zones(rng: np.random.Generator) -> pd.DataFrame:
    """Create the manually selected 20-zone Bengaluru feature table."""
    records = []

    for idx, (zone_name, latitude, longitude, zone_type, traffic_score) in enumerate(
        ZONE_DEFINITIONS, start=1
    ):
        density_low, density_high = POPULATION_DENSITY_RANGES[zone_type]
        population_density = int(rng.integers(density_low, density_high + 1))
        ev_noise = int(rng.integers(-300, 501))
        ev_count_estimate = int(
            round(
                max(
                    500,
                    population_density * EV_MULTIPLIERS[zone_type]
                    + traffic_score * 25
                    + ev_noise,
                )
            )
        )

        records.append(
            {
                "zone_id": f"Z{idx:03d}",
                "zone_name": zone_name,
                "latitude": latitude,
                "longitude": longitude,
                "zone_type": zone_type,
                "population_density": population_density,
                "traffic_score": traffic_score,
                "ev_count_estimate": ev_count_estimate,
                "demand_growth_rate": growth_rate_for_zone(zone_name, zone_type, rng),
            }
        )

    return pd.DataFrame(records)


def create_zone_master(charger_counts: pd.DataFrame) -> pd.DataFrame:
    """Merge zone features with Step 1 existing charger counts."""
    rng = np.random.default_rng(RANDOM_SEED)
    zones = create_base_zones(rng)

    zones_master = zones.merge(
        charger_counts,
        left_on="zone_name",
        right_on="area",
        how="left",
    ).drop(columns=["area"])

    zones_master["existing_chargers"] = (
        zones_master["existing_chargers"].fillna(0).astype(int)
    )
    charger_denominator = zones_master["existing_chargers"].clip(lower=1)
    zones_master["charger_gap_score"] = (
        zones_master["ev_count_estimate"] / charger_denominator
    ).round(2)
    zones_master["charger_coverage_level"] = zones_master.apply(
        lambda row: coverage_level(
            int(row["existing_chargers"]), float(row["charger_gap_score"])
        ),
        axis=1,
    )

    final_columns = [
        "zone_id",
        "zone_name",
        "latitude",
        "longitude",
        "zone_type",
        "population_density",
        "traffic_score",
        "ev_count_estimate",
        "demand_growth_rate",
        "existing_chargers",
        "charger_gap_score",
        "charger_coverage_level",
    ]
    return zones_master[final_columns]


def create_report(zones_master: pd.DataFrame) -> None:
    """Create the Step 2 Markdown report."""
    total_existing_chargers = int(zones_master["existing_chargers"].sum())
    zero_charger_zones = zones_master.loc[
        zones_master["existing_chargers"] == 0, "zone_name"
    ].tolist()
    top_ev = zones_master.nlargest(5, "ev_count_estimate")[
        ["zone_name", "ev_count_estimate"]
    ]
    top_gap = zones_master.nlargest(5, "charger_gap_score")[
        ["zone_name", "charger_gap_score", "existing_chargers"]
    ]

    zero_charger_text = ", ".join(zero_charger_zones) if zero_charger_zones else "None"
    top_ev_lines = [
        f"| {row.zone_name} | {row.ev_count_estimate} |"
        for row in top_ev.itertuples(index=False)
    ]
    top_gap_lines = [
        f"| {row.zone_name} | {row.charger_gap_score:.2f} | {row.existing_chargers} |"
        for row in top_gap.itertuples(index=False)
    ]

    report = f"""# Step 2 Zone Master Report

## Step Name

Create Bengaluru zone-level ML master dataset.

## Inputs And Outputs

- Input used: `data/processed/charger_counts_by_area.csv`
- Output generated: `data/processed/zones_master.csv`

## Summary

- Number of zones: {len(zones_master)}
- Total existing chargers mapped to selected zones: {total_existing_chargers}
- Zones with zero chargers: {zero_charger_text}

## Top 5 Zones By EV Count Estimate

| Zone | EV count estimate |
|---|---:|
{chr(10).join(top_ev_lines)}

## Top 5 Zones By Charger Gap Score

| Zone | Charger gap score | Existing chargers |
|---|---:|---:|
{chr(10).join(top_gap_lines)}

## Notes For ML

- `zones_master.csv` becomes the base feature table.
- `existing_chargers` and `charger_gap_score` help infrastructure planning.
- `traffic_score` and `population_density` are demand proxies.
- `ev_count_estimate` and `demand_growth_rate` support demand forecasting.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved Step 2 report to: {REPORT_PATH}")


def print_summary(zones_master: pd.DataFrame) -> None:
    """Print Step 2 inspection outputs."""
    total_existing_chargers = int(zones_master["existing_chargers"].sum())
    no_coverage = zones_master.loc[
        zones_master["charger_coverage_level"] == "No Coverage", "zone_name"
    ].tolist()
    top_gap = zones_master.nlargest(5, "charger_gap_score")[
        ["zone_name", "charger_gap_score", "existing_chargers", "ev_count_estimate"]
    ]

    print(f"\nzones_master shape: {zones_master.shape}")
    print("\nFirst 10 rows:")
    print(zones_master.head(10).to_string(index=False))
    print(f"\nTotal existing chargers mapped: {total_existing_chargers}")
    print(
        "Zones with no charger coverage: "
        + (", ".join(no_coverage) if no_coverage else "None")
    )
    print("\nTop 5 charger gap zones:")
    print(top_gap.to_string(index=False))


def main() -> None:
    """Run Step 2 zone master creation."""
    try:
        ensure_directories()
        charger_counts = read_charger_counts()
        zones_master = create_zone_master(charger_counts)
        zones_master.to_csv(ZONES_MASTER_PATH, index=False)
        print(f"Saved zone master dataset to: {ZONES_MASTER_PATH}")
        create_report(zones_master)
        print_summary(zones_master)
        print("\nStep 2 zone master pipeline complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 2 failed: {exc}") from exc


if __name__ == "__main__":
    main()
