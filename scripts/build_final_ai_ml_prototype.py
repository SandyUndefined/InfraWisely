"""Build a self-contained final AI/ML prototype bundle.

This script does not modify the original root pipeline outputs. It copies the
finished AI/ML artifacts into final_prototype/, trains a separate final model
copy, and rebuilds cleaned dashboard JSON for the final demo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "final_prototype"
RANDOM_SEED = 42

SOURCE_PROCESSED = ROOT / "data" / "processed"
SOURCE_REPORTS = ROOT / "reports"
SOURCE_SRC = ROOT / "src"

FINAL_DATA = FINAL / "data"
FINAL_PROCESSED = FINAL_DATA / "processed"
FINAL_REPORTS = FINAL / "reports"
FINAL_MODELS = FINAL / "models"
FINAL_SRC = FINAL / "src"

TARGET_COLUMN = "predicted_ev_load_kw"
FEATURE_COLUMNS = [
    "hour",
    "time_period",
    "is_peak_hour",
    "zone_type",
    "ev_count_estimate",
    "traffic_score",
    "existing_chargers",
    "demand_growth_rate",
    "transformer_capacity_kw",
    "base_grid_load_kw",
]
CATEGORICAL_COLUMNS = ["time_period", "zone_type"]
NUMERIC_COLUMNS = [
    "hour",
    "is_peak_hour",
    "ev_count_estimate",
    "traffic_score",
    "existing_chargers",
    "demand_growth_rate",
    "transformer_capacity_kw",
    "base_grid_load_kw",
]


def ensure_dirs() -> None:
    for path in (FINAL_DATA, FINAL_PROCESSED, FINAL_REPORTS, FINAL_MODELS, FINAL_SRC):
        path.mkdir(parents=True, exist_ok=True)


def copy_tree_contents(source: Path, target: Path, pattern: str = "*") -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob(pattern):
        if path.is_file():
            shutil.copy2(path, target / path.name)


def copy_pipeline_artifacts() -> None:
    copy_tree_contents(SOURCE_PROCESSED, FINAL_PROCESSED, "*.csv")
    copy_tree_contents(SOURCE_REPORTS, FINAL_REPORTS, "*.md")
    if FINAL_SRC.exists():
        shutil.rmtree(FINAL_SRC)
    shutil.copytree(SOURCE_SRC, FINAL_SRC, ignore=shutil.ignore_patterns("__pycache__"))


def build_model_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_SEED,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_final_model() -> dict[str, object]:
    hourly_path = FINAL_PROCESSED / "hourly_demand.csv"
    if not hourly_path.exists():
        raise FileNotFoundError(f"Missing training data: {hourly_path}")

    data = pd.read_csv(hourly_path)
    data["is_peak_hour"] = data["is_peak_hour"].astype(bool)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    baseline = np.full(len(y_test), float(y_train.mean()))

    model_mae = float(mean_absolute_error(y_test, y_pred))
    model_rmse = rmse(y_test, y_pred)
    baseline_mae = float(mean_absolute_error(y_test, baseline))
    baseline_rmse = rmse(y_test, baseline)
    metrics = {
        "model_name": "RandomForestRegressor",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "row_count": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": round(model_mae, 4),
        "rmse": round(model_rmse, 4),
        "r2_score": round(float(r2_score(y_test, y_pred)), 6),
        "baseline_mae": round(baseline_mae, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "baseline_r2": round(float(r2_score(y_test, baseline)), 6),
        "improvement_over_baseline_mae_percent": round(
            ((baseline_mae - model_mae) / baseline_mae) * 100, 2
        ),
        "improvement_over_baseline_rmse_percent": round(
            ((baseline_rmse - model_rmse) / baseline_rmse) * 100, 2
        ),
    }

    joblib.dump(pipeline, FINAL_MODELS / "ev_demand_model.pkl")
    (FINAL_MODELS / "ev_demand_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def write_final_summary(metrics: dict[str, object]) -> None:
    summary = json.loads((FINAL / "data" / "dashboard" / "summary_metrics.json").read_text())
    text = f"""# GridCharge AI Final Prototype - AI/ML Bundle

This folder is the final self-contained AI/ML prototype bundle. The original root pipeline is preserved, while this folder contains copied final datasets, reports, dashboard JSON, frontend files, and a separately saved final model.

## Final Model

- Model: Random Forest Regressor
- Target: `predicted_ev_load_kw`
- R2: {metrics['r2_score']}
- MAE: {metrics['mae']} kW
- RMSE: {metrics['rmse']} kW
- Baseline MAE improvement: {metrics['improvement_over_baseline_mae_percent']}%

## Risk Reduction Metrics

- Critical risk: {summary['critical_hours_before']} -> {summary['critical_hours_after']}
- Overload risk: {summary['overloaded_hours_before']} -> {summary['overloaded_hours_after']}
- Meaning: counts are based on risky zone-time records across the 20-zone, 24-hour demo.

## Final Impact

- Peak load reduction: {summary['peak_load_reduction_percent']}%
- Recommended chargers: {summary['total_recommended_chargers']}
- Top station zone: {summary['top_priority_station_zone']}
- Highest risk zone: {summary['highest_risk_zone']}
"""
    (FINAL_REPORTS / "final_prototype_ai_ml_bundle.md").write_text(text, encoding="utf-8")


def rebuild_dashboard_json() -> None:
    script = FINAL / "scripts" / "build_final_dashboard_data.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def main() -> None:
    ensure_dirs()
    copy_pipeline_artifacts()
    rebuild_dashboard_json()
    metrics = train_final_model()
    retrain_script = FINAL / "scripts" / "retrain_final_model_with_expanded_data.py"
    if retrain_script.exists():
        subprocess.run([sys.executable, str(retrain_script)], cwd=ROOT, check=True)
        v2_metrics_path = FINAL_MODELS / "ev_demand_model_v2_metrics.json"
        if v2_metrics_path.exists():
            metrics = json.loads(v2_metrics_path.read_text(encoding="utf-8"))
    write_final_summary(metrics)
    print("Final AI/ML prototype bundle created.")
    print(f"Final model: {FINAL_MODELS / 'ev_demand_model_v2.pkl'}")
    print(f"Final metrics: {FINAL_MODELS / 'ev_demand_model_v2_metrics.json'}")
    print(f"Final dashboard: http://localhost:8000/final_prototype/frontend/")


if __name__ == "__main__":
    main()
