"""Create synthetic grid capacity constraints for GridCharge AI zones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ZONES_MASTER_PATH = Path("data/processed/zones_master.csv")
GRID_CAPACITY_PATH = Path("data/processed/grid_capacity.csv")
REPORT_PATH = Path("reports/step3_grid_capacity_report.md")
RANDOM_SEED = 42

CAPACITY_RANGES_KW = {
    "residential": (900, 1800),
    "commercial": (1400, 2600),
    "mixed": (1200, 2400),
    "industrial": (1800, 3200),
}

BASE_LOAD_RANGES = {
    "residential": (0.45, 0.65),
    "commercial": (0.50, 0.75),
    "mixed": (0.50, 0.70),
    "industrial": (0.55, 0.78),
}

PEAK_LOAD_RANGES = {
    "residential": (0.65, 0.92),
    "commercial": (0.70, 0.95),
    "mixed": (0.68, 0.94),
    "industrial": (0.72, 0.96),
}

SUBSTATION_NAMES = {
    "Whitefield": "Whitefield S/S",
    "Indiranagar": "Indiranagar S/S",
    "Koramangala": "Koramangala S/S",
    "Electronic City": "Electronic City S/S",
    "MG Road": "MG Road S/S",
    "Jayanagar": "Jayanagar S/S",
    "Hebbal": "Hebbal S/S",
    "Yelahanka": "Yelahanka S/S",
    "BTM Layout": "BTM S/S",
    "Rajajinagar": "Rajajinagar S/S",
    "Marathahalli": "Marathahalli S/S",
    "Banashankari": "Banashankari S/S",
    "HSR Layout": "HSR S/S",
    "KR Puram": "KR Puram S/S",
    "Malleshwaram": "Malleshwaram S/S",
    "Sarjapur Road": "Sarjapur S/S",
    "Peenya": "Peenya S/S",
    "Majestic": "Majestic S/S",
    "Bellandur": "Bellandur S/S",
    "Kengeri": "Kengeri S/S",
}


def ensure_directories() -> None:
    """Create required output directories."""
    GRID_CAPACITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_zones_master(path: Path = ZONES_MASTER_PATH) -> pd.DataFrame:
    """Read the Step 2 zone master table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}. Run Step 2 before creating grid capacity data."
        )

    zones = pd.read_csv(path)
    required_columns = {"zone_id", "zone_name", "zone_type"}
    missing_columns = required_columns.difference(zones.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    unknown_types = set(zones["zone_type"]) - set(CAPACITY_RANGES_KW)
    if unknown_types:
        raise ValueError(f"Unsupported zone_type values found: {sorted(unknown_types)}")

    print(f"Loaded zones master from: {path}")
    print(f"Zone rows: {len(zones)}")
    return zones


def classify_grid_status(load_ratio: float) -> str:
    """Convert current load ratio into an explainable grid capacity status."""
    if load_ratio < 0.70:
        return "Healthy"
    if load_ratio < 0.85:
        return "Moderate"
    if load_ratio < 0.95:
        return "Constrained"
    return "Critical"


def create_grid_capacity(zones: pd.DataFrame) -> pd.DataFrame:
    """Generate reproducible synthetic feeder and transformer constraints."""
    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for idx, row in enumerate(zones.itertuples(index=False), start=1):
        zone_type = row.zone_type
        capacity_low, capacity_high = CAPACITY_RANGES_KW[zone_type]
        base_low, base_high = BASE_LOAD_RANGES[zone_type]
        peak_low, peak_high = PEAK_LOAD_RANGES[zone_type]

        transformer_capacity_kw = int(rng.integers(capacity_low, capacity_high + 1))
        base_load_kw = int(round(transformer_capacity_kw * rng.uniform(base_low, base_high)))
        current_peak_load_kw = int(
            round(transformer_capacity_kw * rng.uniform(peak_low, peak_high))
        )
        current_peak_load_kw = max(current_peak_load_kw, base_load_kw)

        spare_capacity_kw = transformer_capacity_kw - current_peak_load_kw
        load_ratio = round(current_peak_load_kw / transformer_capacity_kw, 3)
        grid_health_score = round((spare_capacity_kw / transformer_capacity_kw) * 100, 2)

        records.append(
            {
                "zone_id": row.zone_id,
                "zone_name": row.zone_name,
                "feeder_id": f"FDR-{idx:03d}",
                "transformer_id": f"DTR-{idx:03d}",
                "substation_name": SUBSTATION_NAMES.get(row.zone_name, f"{row.zone_name} S/S"),
                "zone_type": zone_type,
                "transformer_capacity_kw": transformer_capacity_kw,
                "base_load_kw": base_load_kw,
                "current_peak_load_kw": current_peak_load_kw,
                "spare_capacity_kw": spare_capacity_kw,
                "load_ratio": load_ratio,
                "grid_health_score": grid_health_score,
                "grid_capacity_status": classify_grid_status(load_ratio),
            }
        )

    return pd.DataFrame(
        records,
        columns=[
            "zone_id",
            "zone_name",
            "feeder_id",
            "transformer_id",
            "substation_name",
            "zone_type",
            "transformer_capacity_kw",
            "base_load_kw",
            "current_peak_load_kw",
            "spare_capacity_kw",
            "load_ratio",
            "grid_health_score",
            "grid_capacity_status",
        ],
    )


def create_report(grid_capacity: pd.DataFrame) -> None:
    """Write the Step 3 grid capacity report."""
    total_capacity = int(grid_capacity["transformer_capacity_kw"].sum())
    average_health = float(grid_capacity["grid_health_score"].mean())
    status_counts = grid_capacity["grid_capacity_status"].value_counts().sort_index()
    constrained = grid_capacity.nlargest(5, "load_ratio")[
        ["zone_name", "load_ratio", "grid_capacity_status"]
    ]
    spare_capacity = grid_capacity.nlargest(5, "spare_capacity_kw")[
        ["zone_name", "spare_capacity_kw", "grid_health_score"]
    ]

    status_lines = [
        f"| {status} | {count} |" for status, count in status_counts.items()
    ]
    constrained_lines = [
        f"| {row.zone_name} | {row.load_ratio:.3f} | {row.grid_capacity_status} |"
        for row in constrained.itertuples(index=False)
    ]
    spare_lines = [
        f"| {row.zone_name} | {row.spare_capacity_kw} | {row.grid_health_score:.2f} |"
        for row in spare_capacity.itertuples(index=False)
    ]

    report = f"""# Step 3 Grid Capacity Report

## Step Name

Create synthetic masked grid capacity constraints for Bengaluru zones.

## Inputs And Outputs

- Input used: `data/processed/zones_master.csv`
- Output generated: `data/processed/grid_capacity.csv`

## Summary

- Number of zones: {len(grid_capacity)}
- Total transformer capacity: {total_capacity} kW
- Average grid health score: {average_health:.2f}

## Count By Grid Capacity Status

| Status | Count |
|---|---:|
{chr(10).join(status_lines)}

## Top 5 Most Constrained Zones By Load Ratio

| Zone | Load ratio | Status |
|---|---:|---|
{chr(10).join(constrained_lines)}

## Top 5 Zones By Spare Capacity

| Zone | Spare capacity kW | Grid health score |
|---|---:|---:|
{chr(10).join(spare_lines)}

## Notes For ML

- `transformer_capacity_kw` and `current_peak_load_kw` enable grid stress scoring.
- `spare_capacity_kw` supports station feasibility checks.
- `grid_capacity_status` helps explain recommendations.
- This synthetic layer mimics masked utility constraints without exposing sensitive data.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved Step 3 report to: {REPORT_PATH}")


def print_summary(grid_capacity: pd.DataFrame) -> None:
    """Print useful Step 3 inspection outputs."""
    constrained = grid_capacity.nlargest(5, "load_ratio")[
        [
            "zone_name",
            "transformer_capacity_kw",
            "current_peak_load_kw",
            "load_ratio",
            "grid_capacity_status",
        ]
    ]

    print(f"\ngrid_capacity shape: {grid_capacity.shape}")
    print("\nFirst 10 rows:")
    print(grid_capacity.head(10).to_string(index=False))
    print("\ngrid_capacity_status value counts:")
    print(grid_capacity["grid_capacity_status"].value_counts().to_string())
    print("\nTop 5 most constrained zones:")
    print(constrained.to_string(index=False))
    print(f"\nAverage grid health score: {grid_capacity['grid_health_score'].mean():.2f}")


def main() -> None:
    """Run Step 3 grid capacity creation."""
    try:
        ensure_directories()
        zones = read_zones_master()
        grid_capacity = create_grid_capacity(zones)
        grid_capacity.to_csv(GRID_CAPACITY_PATH, index=False)
        print(f"Saved grid capacity dataset to: {GRID_CAPACITY_PATH}")
        create_report(grid_capacity)
        print_summary(grid_capacity)
        print("\nStep 3 grid capacity pipeline complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 3 failed: {exc}") from exc


if __name__ == "__main__":
    main()
