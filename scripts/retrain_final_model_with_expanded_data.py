"""Retrain the final prototype EV demand model with expanded synthetic data.

The original root model is not changed. This script creates a larger final
prototype training set from the 20-zone master and masked grid constraints,
then trains a separate model under final_prototype/models.
"""

from __future__ import annotations

import json
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
DATA_DIR = FINAL / "data" / "processed"
DASHBOARD_DIR = FINAL / "data" / "dashboard"
MODEL_DIR = FINAL / "models"
REPORT_DIR = FINAL / "reports"

ZONES_PATH = DATA_DIR / "zones_master.csv"
GRID_PATH = DATA_DIR / "grid_capacity.csv"
ORIGINAL_HOURLY_PATH = DATA_DIR / "hourly_demand.csv"
TRAINING_PATH = DATA_DIR / "final_expanded_hourly_training_data.csv"
PREDICTIONS_PATH = DATA_DIR / "final_hourly_demand_with_predictions.csv"
MODEL_PATH = MODEL_DIR / "ev_demand_model_v2.pkl"
METRICS_PATH = MODEL_DIR / "ev_demand_model_v2_metrics.json"
REPORT_PATH = REPORT_DIR / "final_model_v2_training_report.md"
SUMMARY_JSON_PATH = DASHBOARD_DIR / "summary_metrics.json"
EXPLAIN_JSON_PATH = DASHBOARD_DIR / "model_explainability_summary.json"

RANDOM_SEED = 42
SIMULATED_DAYS = 60
TARGET_COLUMN = "synthetic_ev_load_kw"
FEATURE_COLUMNS = [
    "hour",
    "time_period",
    "is_peak_hour",
    "zone_type",
    "day_type",
    "is_weekend",
    "weather_condition",
    "temperature_c",
    "ev_count_estimate",
    "traffic_score",
    "traffic_score_hourly",
    "existing_chargers",
    "charger_utilization_proxy",
    "demand_growth_rate",
    "transformer_capacity_kw",
    "base_grid_load_kw",
]
CATEGORICAL_COLUMNS = ["time_period", "zone_type", "day_type", "weather_condition"]
NUMERIC_COLUMNS = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


def time_period(hour: int) -> str:
    if 0 <= hour <= 5:
        return "Night"
    if 6 <= hour <= 9:
        return "Morning"
    if 10 <= hour <= 16:
        return "Day"
    if 17 <= hour <= 22:
        return "Evening Peak"
    return "Late Night"


def base_multiplier(hour: int, rng: np.random.Generator) -> float:
    ranges = {
        "Night": (0.62, 0.84),
        "Morning": (0.84, 1.08),
        "Day": (0.90, 1.12),
        "Evening Peak": (1.05, 1.28),
        "Late Night": (0.72, 0.92),
    }
    low, high = ranges[time_period(hour)]
    return float(rng.uniform(low, high))


def ev_profile_multiplier(zone_type: str, hour: int, day_type: str, rng: np.random.Generator) -> float:
    period = time_period(hour)
    profiles = {
        "residential": {
            "Night": (0.55, 0.90),
            "Morning": (0.75, 1.12),
            "Day": (0.62, 0.98),
            "Evening Peak": (1.70, 2.55),
            "Late Night": (1.00, 1.42),
        },
        "commercial": {
            "Night": (0.25, 0.56),
            "Morning": (0.75, 1.12),
            "Day": (1.42, 2.20),
            "Evening Peak": (0.92, 1.38),
            "Late Night": (0.42, 0.78),
        },
        "mixed": {
            "Night": (0.42, 0.78),
            "Morning": (0.86, 1.20),
            "Day": (1.08, 1.62),
            "Evening Peak": (1.45, 2.28),
            "Late Night": (0.78, 1.15),
        },
        "industrial": {
            "Night": (0.42, 0.78),
            "Morning": (1.30, 2.00),
            "Day": (0.88, 1.24),
            "Evening Peak": (1.20, 1.82),
            "Late Night": (0.52, 0.88),
        },
    }
    low, high = profiles[zone_type][period]
    multiplier = float(rng.uniform(low, high))
    if day_type == "Weekend" and zone_type in {"commercial", "industrial"}:
        multiplier *= float(rng.uniform(0.76, 0.92))
    if day_type == "Weekend" and zone_type in {"residential", "mixed"} and period in {"Day", "Evening Peak"}:
        multiplier *= float(rng.uniform(1.04, 1.16))
    return multiplier


def risk_level(load_ratio: float) -> str:
    if load_ratio < 0.75:
        return "Low"
    if load_ratio < 0.90:
        return "Medium"
    if load_ratio <= 1.00:
        return "High"
    return "Critical"


def generate_training_data() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    zones = pd.read_csv(ZONES_PATH)
    grid = pd.read_csv(GRID_PATH)
    merged = zones.merge(
        grid[["zone_id", "zone_name", "base_load_kw", "transformer_capacity_kw"]],
        on=["zone_id", "zone_name"],
        how="inner",
        validate="one_to_one",
    )

    weather_options = ["Clear", "Hot", "Rain"]
    records = []
    for day in range(SIMULATED_DAYS):
        day_type = "Weekend" if day % 7 in {5, 6} else "Weekday"
        weather = str(rng.choice(weather_options, p=[0.62, 0.25, 0.13]))
        temp_base = {"Clear": 28, "Hot": 34, "Rain": 24}[weather]
        temperature = float(rng.normal(temp_base, 2.0))
        day_growth_factor = 1 + (day / 365) * 0.18

        for zone in merged.itertuples(index=False):
            for hour in range(24):
                period = time_period(hour)
                traffic_hour_factor = {
                    "Night": 0.55,
                    "Morning": 1.10,
                    "Day": 1.00,
                    "Evening Peak": 1.22,
                    "Late Night": 0.72,
                }[period]
                traffic_score_hourly = max(
                    10,
                    min(100, zone.traffic_score * traffic_hour_factor + rng.normal(0, 4)),
                )
                base_grid_load_kw = zone.base_load_kw * base_multiplier(hour, rng)
                ev_base_load = (
                    zone.ev_count_estimate * 0.035
                    + traffic_score_hourly * 1.12
                    + zone.existing_chargers * 6.5
                    + zone.demand_growth_rate * 270
                )
                charger_utilization_proxy = min(
                    1.0,
                    (zone.ev_count_estimate / max(zone.existing_chargers, 1)) / 7000,
                )
                weather_factor = {"Clear": 1.0, "Hot": 1.08, "Rain": 0.94}[weather]
                temp_factor = 1 + max(0, temperature - 30) * 0.008
                ev_load = (
                    ev_base_load
                    * ev_profile_multiplier(zone.zone_type, hour, day_type, rng)
                    * weather_factor
                    * temp_factor
                    * day_growth_factor
                    * float(rng.uniform(0.94, 1.07))
                )
                ev_load = max(0, ev_load)
                total_load = base_grid_load_kw + ev_load
                load_ratio = total_load / zone.transformer_capacity_kw
                records.append(
                    {
                        "zone_id": zone.zone_id,
                        "zone_name": zone.zone_name,
                        "day_index": day + 1,
                        "day_type": day_type,
                        "is_weekend": day_type == "Weekend",
                        "weather_condition": weather,
                        "temperature_c": round(temperature, 2),
                        "hour": hour,
                        "time_period": period,
                        "is_peak_hour": 17 <= hour <= 22,
                        "zone_type": zone.zone_type,
                        "ev_count_estimate": int(zone.ev_count_estimate),
                        "traffic_score": int(zone.traffic_score),
                        "traffic_score_hourly": round(float(traffic_score_hourly), 2),
                        "existing_chargers": int(zone.existing_chargers),
                        "charger_utilization_proxy": round(float(charger_utilization_proxy), 3),
                        "demand_growth_rate": round(float(zone.demand_growth_rate), 3),
                        "transformer_capacity_kw": int(zone.transformer_capacity_kw),
                        "base_grid_load_kw": round(float(base_grid_load_kw), 2),
                        "synthetic_ev_load_kw": round(float(ev_load), 2),
                        "synthetic_total_load_kw": round(float(total_load), 2),
                        "synthetic_load_ratio": round(float(load_ratio), 3),
                        "synthetic_risk_level": risk_level(load_ratio),
                    }
                )

    data = pd.DataFrame(records)
    data.to_csv(TRAINING_PATH, index=False)
    return data


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_SEED,
        max_depth=18,
        min_samples_split=3,
        min_samples_leaf=1,
        n_jobs=1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_model(data: pd.DataFrame) -> tuple[Pipeline, dict[str, object]]:
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    model = build_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline = np.full(len(y_test), float(y_train.mean()))
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    baseline_mae = float(mean_absolute_error(y_test, baseline))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline)))

    metrics = {
        "model_name": "RandomForestRegressor_v2_expanded_synthetic",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "row_count": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(float(r2_score(y_test, pred)), 6),
        "baseline_mae": round(baseline_mae, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "baseline_r2": round(float(r2_score(y_test, baseline)), 6),
        "improvement_over_baseline_mae_percent": round(((baseline_mae - mae) / baseline_mae) * 100, 2),
        "improvement_over_baseline_rmse_percent": round(((baseline_rmse - rmse) / baseline_rmse) * 100, 2),
        "training_data_note": (
            f"Expanded final prototype dataset generated across {SIMULATED_DAYS} simulated days, "
            "with weekday/weekend, weather, traffic, charger utilization, growth, and grid features."
        ),
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics


def predict_original_day(model: Pipeline) -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_HOURLY_PATH)
    original["day_type"] = "Weekday"
    original["is_weekend"] = False
    original["weather_condition"] = "Clear"
    original["temperature_c"] = 28.0
    original["traffic_score_hourly"] = original["traffic_score"]
    original["charger_utilization_proxy"] = (
        original["ev_count_estimate"] / original["existing_chargers"].clip(lower=1) / 7000
    ).clip(upper=1).round(3)
    original["final_model_predicted_ev_load_kw"] = model.predict(original[FEATURE_COLUMNS]).round(2)
    original["final_model_total_load_kw"] = (
        original["base_grid_load_kw"] + original["final_model_predicted_ev_load_kw"]
    ).round(2)
    original["final_model_load_ratio"] = (
        original["final_model_total_load_kw"] / original["transformer_capacity_kw"]
    ).round(3)
    original["final_model_risk_level"] = original["final_model_load_ratio"].apply(risk_level)
    original.to_csv(PREDICTIONS_PATH, index=False)
    return original


def update_dashboard_metrics(metrics: dict[str, object]) -> None:
    summary = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))
    summary["model_r2_score"] = round(float(metrics["r2_score"]), 2)
    summary["model_mae"] = round(float(metrics["mae"]), 2)
    summary["model_rmse"] = round(float(metrics["rmse"]), 2)
    summary["model_version"] = "v2 expanded synthetic"
    summary["training_rows"] = metrics["row_count"]
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if EXPLAIN_JSON_PATH.exists():
        explain = json.loads(EXPLAIN_JSON_PATH.read_text(encoding="utf-8"))
        explain["model_name"] = metrics["model_name"]
        explain["target_variable"] = metrics["target_column"]
        explain["input_features"] = metrics["feature_columns"]
        explain["model_r2_score"] = round(float(metrics["r2_score"]), 2)
        explain["mae"] = round(float(metrics["mae"]), 2)
        explain["rmse"] = round(float(metrics["rmse"]), 2)
        explain["baseline_comparison"] = {
            "baseline_mae": round(float(metrics["baseline_mae"]), 2),
            "baseline_rmse": round(float(metrics["baseline_rmse"]), 2),
            "mae_improvement_percent": metrics["improvement_over_baseline_mae_percent"],
            "rmse_improvement_percent": metrics["improvement_over_baseline_rmse_percent"],
        }
        EXPLAIN_JSON_PATH.write_text(json.dumps(explain, indent=2), encoding="utf-8")


def write_report(metrics: dict[str, object], predictions: pd.DataFrame) -> None:
    top_features = []
    model = joblib.load(MODEL_PATH)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    clean_names = [name.replace("categorical__", "").replace("numeric__", "") for name in feature_names]
    importances = model.named_steps["model"].feature_importances_
    for feature, importance in sorted(zip(clean_names, importances), key=lambda item: item[1], reverse=True)[:12]:
        top_features.append(f"| {feature} | {importance:.4f} |")

    report = f"""# Final Prototype Model v2 Training Report

## What Changed

The final prototype now trains a new model inside `final_prototype/models` using an expanded synthetic dataset instead of only the original 480 one-day rows.

## Expanded Data

- Simulated days: {SIMULATED_DAYS}
- Training rows: {metrics['row_count']}
- Added features: `day_type`, `is_weekend`, `weather_condition`, `temperature_c`, `traffic_score_hourly`, `charger_utilization_proxy`

## Metrics

- R2: {metrics['r2_score']}
- MAE: {metrics['mae']} kW
- RMSE: {metrics['rmse']} kW
- Baseline MAE: {metrics['baseline_mae']} kW
- Baseline RMSE: {metrics['baseline_rmse']} kW
- MAE improvement over baseline: {metrics['improvement_over_baseline_mae_percent']}%
- RMSE improvement over baseline: {metrics['improvement_over_baseline_rmse_percent']}%

## Top Feature Importances

| Feature | Importance |
|---|---:|
{chr(10).join(top_features)}

## Prediction Export

The script also creates `final_hourly_demand_with_predictions.csv` for the original 20-zone, 24-hour demo day using the new final model.

## Risk Mix On Demo Day

{predictions['final_model_risk_level'].value_counts().to_string()}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    data = generate_training_data()
    model, metrics = train_model(data)
    predictions = predict_original_day(model)
    update_dashboard_metrics(metrics)
    write_report(metrics, predictions)
    print("Retrained final prototype model with expanded data.")
    print(f"Training rows: {metrics['row_count']}")
    print(f"R2: {metrics['r2_score']}")
    print(f"MAE: {metrics['mae']} kW")
    print(f"RMSE: {metrics['rmse']} kW")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
