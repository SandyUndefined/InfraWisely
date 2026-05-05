"""Train the Step 5 EV charging demand prediction model."""

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


HOURLY_DEMAND_PATH = Path("data/processed/hourly_demand.csv")
PREDICTIONS_PATH = Path("data/processed/hourly_demand_with_predictions.csv")
MODEL_PATH = Path("models/ev_demand_model.pkl")
METRICS_PATH = Path("models/ev_demand_model_metrics.json")
REPORT_PATH = Path("reports/step5_model_training_report.md")
RANDOM_SEED = 42

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
REQUIRED_COLUMNS = {
    "zone_id",
    "zone_name",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
}


def ensure_directories() -> None:
    """Create folders used by the model training step."""
    Path("src/models").mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_hourly_demand(path: Path = HOURLY_DEMAND_PATH) -> pd.DataFrame:
    """Read and validate the hourly demand training dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}. Run Step 4 first.")

    hourly_demand = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(hourly_demand.columns)
    if missing_columns:
        raise ValueError(f"{path} missing required columns: {sorted(missing_columns)}")

    hourly_demand = hourly_demand.copy()
    hourly_demand["is_peak_hour"] = hourly_demand["is_peak_hour"].astype(bool)

    print(f"Loaded hourly demand data from: {path}")
    print(f"Dataset shape: {hourly_demand.shape}")
    return hourly_demand


def build_model_pipeline() -> Pipeline:
    """Build preprocessing and RandomForest model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("numeric", "passthrough", NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_SEED,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def percentage_improvement(baseline_value: float, model_value: float) -> float:
    """Calculate percentage improvement where lower metric is better."""
    if baseline_value == 0:
        return 0.0
    return round(((baseline_value - model_value) / baseline_value) * 100, 2)


def get_feature_importance(model_pipeline: Pipeline) -> pd.DataFrame:
    """Extract feature importances with transformed one-hot feature names."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [
        name.replace("categorical__", "").replace("numeric__", "")
        for name in feature_names
    ]

    importance = pd.DataFrame(
        {
            "feature": clean_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance.reset_index(drop=True)


def add_full_dataset_predictions(
    hourly_demand: pd.DataFrame, model_pipeline: Pipeline
) -> pd.DataFrame:
    """Generate predictions and prediction error columns for all rows."""
    predictions = model_pipeline.predict(hourly_demand[FEATURE_COLUMNS])
    result = hourly_demand.copy()
    result["model_predicted_ev_load_kw"] = np.round(predictions, 2)
    result["prediction_error_kw"] = np.round(
        result[TARGET_COLUMN] - result["model_predicted_ev_load_kw"], 2
    )
    actual = result[TARGET_COLUMN].to_numpy()
    absolute_error = np.abs(result["prediction_error_kw"].to_numpy())
    result["prediction_error_percent"] = np.round(
        np.divide(
            absolute_error,
            actual,
            out=np.zeros_like(absolute_error, dtype=float),
            where=actual != 0,
        )
        * 100,
        2,
    )
    return result


def save_metrics(metrics: dict[str, object]) -> None:
    """Save model metrics as JSON."""
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model metrics to: {METRICS_PATH}")


def create_report(
    metrics: dict[str, object],
    feature_importance: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    """Create Step 5 model training Markdown report."""
    top_features = feature_importance.head(10)
    sample_predictions = predictions_df[
        [
            "zone_name",
            "hour",
            TARGET_COLUMN,
            "model_predicted_ev_load_kw",
            "prediction_error_percent",
        ]
    ].head(10)

    feature_lines = [f"- `{feature}`" for feature in FEATURE_COLUMNS]
    importance_lines = [
        f"| {row.feature} | {row.importance:.4f} |"
        for row in top_features.itertuples(index=False)
    ]
    sample_lines = [
        (
            f"| {row.zone_name} | {row.hour} | {getattr(row, TARGET_COLUMN):.2f} | "
            f"{row.model_predicted_ev_load_kw:.2f} | {row.prediction_error_percent:.2f} |"
        )
        for row in sample_predictions.itertuples(index=False)
    ]

    report = f"""# Step 5 Model Training Report

## Step Name

Train EV charging demand prediction model.

## Input

- Input used: `data/processed/hourly_demand.csv`

## Target Variable

- `{TARGET_COLUMN}`

## Feature Columns

{chr(10).join(feature_lines)}

## Model

- Model used: RandomForestRegressor
- Train/test split: 80/20
- Train rows: {metrics["train_rows"]}
- Test rows: {metrics["test_rows"]}

## Evaluation Metrics

| Metric | Model | Baseline |
|---|---:|---:|
| MAE | {metrics["mae"]:.2f} | {metrics["baseline_mae"]:.2f} |
| RMSE | {metrics["rmse"]:.2f} | {metrics["baseline_rmse"]:.2f} |
| R2 score | {metrics["r2_score"]:.4f} | {metrics["baseline_r2"]:.4f} |

## Improvement Over Baseline

- MAE improvement: {metrics["improvement_over_baseline_mae_percent"]:.2f}%
- RMSE improvement: {metrics["improvement_over_baseline_rmse_percent"]:.2f}%

## Top 10 Feature Importances

| Feature | Importance |
|---|---:|
{chr(10).join(importance_lines)}

## Sample Predictions

| Zone | Hour | Actual EV load kW | Model predicted EV load kW | Error percent |
|---|---:|---:|---:|---:|
{chr(10).join(sample_lines)}

## Notes

- The model predicts hourly EV demand by zone.
- Output can be combined with transformer capacity for grid stress scoring.
- Predictions support smart charging recommendations and planning.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved Step 5 report to: {REPORT_PATH}")


def print_summary(
    hourly_demand: pd.DataFrame,
    metrics: dict[str, object],
    feature_importance: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    """Print model training outputs for inspection."""
    print(f"\nDataset shape: {hourly_demand.shape}")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2 score: {metrics['r2_score']:.4f}")
    print(f"Baseline MAE: {metrics['baseline_mae']:.2f}")
    print(f"Baseline RMSE: {metrics['baseline_rmse']:.2f}")
    print(
        "Improvement over baseline: "
        f"MAE {metrics['improvement_over_baseline_mae_percent']:.2f}%, "
        f"RMSE {metrics['improvement_over_baseline_rmse_percent']:.2f}%"
    )

    print("\nTop 10 feature importances:")
    print(feature_importance.head(10).to_string(index=False))

    print("\nFirst 10 prediction rows:")
    print(
        predictions_df[
            [
                "zone_name",
                "hour",
                TARGET_COLUMN,
                "model_predicted_ev_load_kw",
                "prediction_error_percent",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


def train_and_evaluate(hourly_demand: pd.DataFrame) -> tuple[Pipeline, dict[str, object], pd.DataFrame]:
    """Train the EV demand model and calculate model/baseline metrics."""
    X = hourly_demand[FEATURE_COLUMNS]
    y = hourly_demand[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    baseline_pred = np.full(shape=len(y_test), fill_value=float(y_train.mean()))

    model_mae = float(mean_absolute_error(y_test, y_pred))
    model_rmse = rmse(y_test, y_pred)
    model_r2 = float(r2_score(y_test, y_pred))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_rmse = rmse(y_test, baseline_pred)
    baseline_r2 = float(r2_score(y_test, baseline_pred))

    metrics: dict[str, object] = {
        "model_name": "RandomForestRegressor",
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "row_count": int(len(hourly_demand)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": round(model_mae, 4),
        "rmse": round(model_rmse, 4),
        "r2_score": round(model_r2, 6),
        "baseline_mae": round(baseline_mae, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "baseline_r2": round(baseline_r2, 6),
        "improvement_over_baseline_mae_percent": percentage_improvement(
            baseline_mae, model_mae
        ),
        "improvement_over_baseline_rmse_percent": percentage_improvement(
            baseline_rmse, model_rmse
        ),
    }

    predictions_df = add_full_dataset_predictions(hourly_demand, model_pipeline)
    return model_pipeline, metrics, predictions_df


def main() -> None:
    """Run Step 5 model training."""
    try:
        ensure_directories()
        hourly_demand = read_hourly_demand()
        model_pipeline, metrics, predictions_df = train_and_evaluate(hourly_demand)
        feature_importance = get_feature_importance(model_pipeline)

        joblib.dump(model_pipeline, MODEL_PATH)
        print(f"Saved trained model to: {MODEL_PATH}")

        save_metrics(metrics)
        predictions_df.to_csv(PREDICTIONS_PATH, index=False)
        print(f"Saved hourly demand predictions to: {PREDICTIONS_PATH}")
        create_report(metrics, feature_importance, predictions_df)
        print_summary(hourly_demand, metrics, feature_importance, predictions_df)
        print("\nStep 5 EV demand model training complete.")
    except Exception as exc:
        raise RuntimeError(f"Step 5 failed: {exc}") from exc


if __name__ == "__main__":
    main()
