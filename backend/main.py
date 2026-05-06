"""FastAPI backend for the InfraWisely EV charging optimization MVP."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "ev_demand_model_v2.pkl"
MODEL_METRICS_PATH = MODEL_DIR / "ev_demand_model_v2_metrics.json"

CANONICAL_MODEL_FEATURES = [
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


app = FastAPI(
    title="InfraWisely API",
    description=(
        "Decision-support API for EV demand forecasting, grid stress, charging "
        "schedules, and station planning."
    ),
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictDemandRequest(BaseModel):
    """Feature payload for the canonical v2 EV demand model."""

    hour: int = Field(..., ge=0, le=23)
    time_period: str
    is_peak_hour: bool
    zone_type: str
    day_type: str = Field(..., examples=["Weekday"])
    is_weekend: bool
    weather_condition: str = Field(..., examples=["Clear"])
    temperature_c: float
    ev_count_estimate: int = Field(..., ge=0)
    traffic_score: int = Field(..., ge=0, le=100)
    traffic_score_hourly: float = Field(..., ge=0, le=100)
    existing_chargers: int = Field(..., ge=0)
    charger_utilization_proxy: float = Field(..., ge=0, le=1)
    demand_growth_rate: float = Field(..., ge=0)
    transformer_capacity_kw: float = Field(..., gt=0)
    base_grid_load_kw: float = Field(..., ge=0)


class PredictDemandResponse(BaseModel):
    model_version: str
    model_predicted_ev_load_kw: float
    feature_columns: list[str]


def require_file(path: Path) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing file: {path.relative_to(ROOT_DIR)}")


def read_json_file(path: Path) -> Any:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="pandas is not installed. Install backend dependencies from requirements.txt.",
        ) from exc
    return pd


def to_python_value(value: Any) -> Any:
    pd = load_pandas()
    if pd.isna(value):
        return None

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return round(float(value), 4)
        if isinstance(value, np.bool_):
            return bool(value)

    if isinstance(value, float):
        return round(value, 4)
    return value


def dataframe_records(frame: Any) -> list[dict[str, Any]]:
    clean = frame.copy()
    return [
        {key: to_python_value(value) for key, value in row.items()}
        for row in clean.to_dict(orient="records")
    ]


def filter_records(
    records: list[dict[str, Any]],
    *,
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    hour: Optional[int] = None,
    risk_level: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    filtered = records
    if zone_id:
        filtered = [row for row in filtered if row.get("zone_id") == zone_id]
    if zone_name:
        filtered = [row for row in filtered if row.get("zone_name") == zone_name]
    if hour is not None:
        filtered = [row for row in filtered if row.get("hour") == hour or row.get("peak_hour") == hour]
    if risk_level:
        filtered = [
            row
            for row in filtered
            if row.get("model_risk_level") == risk_level
            or row.get("original_risk_level") == risk_level
            or row.get("before_risk_level") == risk_level
            or row.get("after_risk_level") == risk_level
        ]
    if limit is not None:
        return filtered[:limit]
    return filtered


@lru_cache(maxsize=1)
def load_model() -> Any:
    require_file(MODEL_PATH)
    try:
        import joblib
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="joblib is not installed. Install backend dependencies from requirements.txt.",
        ) from exc

    return joblib.load(MODEL_PATH)


def classify_risk(load_ratio: float) -> str:
    if load_ratio < 0.75:
        return "Low"
    if load_ratio < 0.90:
        return "Medium"
    if load_ratio <= 1.00:
        return "High"
    return "Critical"


def action_priority(risk_level: str) -> str:
    return {
        "Critical": "Immediate Action",
        "High": "Schedule Shift Recommended",
        "Medium": "Monitor",
        "Low": "Normal",
    }[risk_level]


def recommended_action(risk_level: str) -> str:
    return {
        "Critical": "Shift EV charging immediately and restrict fast charging during this hour.",
        "High": "Shift part of EV charging demand to off-peak hours.",
        "Medium": "Monitor load and encourage off-peak charging.",
        "Low": "No immediate action required.",
    }[risk_level]


def risk_reason(row: Any) -> str:
    load_percent = float(row["model_load_ratio"]) * 100
    hour_label = f"{int(row['hour']):02d}:00"
    if row["model_risk_level"] == "Critical":
        return (
            f"{row['zone_name']} at {hour_label} is Critical because predicted total "
            f"load reaches {load_percent:.1f}% of transformer capacity."
        )
    if row["model_risk_level"] == "High":
        return (
            f"{row['zone_name']} at {hour_label} is High risk because predicted total "
            f"load reaches {load_percent:.1f}% of transformer capacity during {row['time_period']}."
        )
    if row["model_risk_level"] == "Medium":
        return (
            f"{row['zone_name']} at {hour_label} needs monitoring because load reaches "
            f"{load_percent:.1f}% of transformer capacity."
        )
    return f"{row['zone_name']} at {hour_label} is Low risk because load remains within safe capacity limits."


def zone_priority(critical_hours: int, high_hours: int, overloaded_hours: int) -> str:
    if critical_hours >= 3 or overloaded_hours >= 3:
        return "Very High"
    if critical_hours >= 1 or high_hours >= 4:
        return "High"
    if high_hours >= 1:
        return "Medium"
    return "Low"


def create_zone_summary(grid_stress: Any) -> Any:
    pd = load_pandas()
    records = []
    for (zone_id, zone_name), group in grid_stress.groupby(["zone_id", "zone_name"]):
        worst = group.sort_values(["risk_score", "model_load_ratio"], ascending=False).iloc[0]
        peak = group[group["is_peak_hour"]]
        critical = int((group["model_risk_level"] == "Critical").sum())
        high = int((group["model_risk_level"] == "High").sum())
        overloaded = int(group["is_overloaded"].sum())
        records.append(
            {
                "zone_id": zone_id,
                "zone_name": zone_name,
                "max_risk_score": round(float(group["risk_score"].max()), 2),
                "avg_risk_score": round(float(group["risk_score"].mean()), 2),
                "max_model_load_ratio": round(float(group["model_load_ratio"].max()), 3),
                "critical_hours_count": critical,
                "high_hours_count": high,
                "overloaded_hours_count": overloaded,
                "peak_hour_avg_risk_score": round(float(peak["risk_score"].mean()), 2),
                "worst_hour": int(worst["hour"]),
                "worst_time_period": worst["time_period"],
                "worst_risk_level": worst["model_risk_level"],
                "zone_priority": zone_priority(critical, high, overloaded),
            }
        )
    priority_order = {"Very High": 0, "High": 1, "Medium": 2, "Low": 3}
    summary = pd.DataFrame(records)
    summary["_priority_order"] = summary["zone_priority"].map(priority_order)
    return summary.sort_values(
        ["_priority_order", "max_risk_score", "avg_risk_score"],
        ascending=[True, False, False],
    ).drop(columns=["_priority_order"]).reset_index(drop=True)


def build_feature_frame() -> Any:
    pd = load_pandas()
    path = PROCESSED_DIR / "hourly_demand.csv"
    require_file(path)

    frame = pd.read_csv(path)
    frame["is_peak_hour"] = frame["is_peak_hour"].astype(bool)
    frame["day_type"] = "Weekday"
    frame["is_weekend"] = False
    frame["weather_condition"] = "Clear"
    frame["temperature_c"] = 28.0
    frame["traffic_score_hourly"] = frame["traffic_score"]
    frame["charger_utilization_proxy"] = (
        frame["ev_count_estimate"] / frame["existing_chargers"].clip(lower=1) / 7000
    ).clip(upper=1).round(3)
    return frame


def create_demand_forecast() -> Any:
    frame = build_feature_frame()
    model = load_model()
    predictions = model.predict(frame[CANONICAL_MODEL_FEATURES])

    demand = frame.copy()
    demand["model_predicted_ev_load_kw"] = predictions.round(2)
    demand["model_total_load_kw"] = (
        demand["base_grid_load_kw"] + demand["model_predicted_ev_load_kw"]
    ).round(2)
    demand["model_load_ratio"] = (
        demand["model_total_load_kw"] / demand["transformer_capacity_kw"]
    ).round(3)
    demand["model_risk_level"] = demand["model_load_ratio"].apply(classify_risk)
    return demand


def create_grid_stress(demand: Any) -> Any:
    grid_stress = demand.copy()
    grid_stress["spare_capacity_after_ev_kw"] = (
        grid_stress["transformer_capacity_kw"] - grid_stress["model_total_load_kw"]
    ).round(2)
    grid_stress["risk_score"] = (grid_stress["model_load_ratio"] * 100).clip(upper=100).round(2)
    grid_stress["is_overloaded"] = grid_stress["model_load_ratio"] > 1.0
    grid_stress["action_priority"] = grid_stress["model_risk_level"].apply(action_priority)
    grid_stress["recommended_action"] = grid_stress["model_risk_level"].apply(recommended_action)
    grid_stress["risk_reason"] = grid_stress.apply(risk_reason, axis=1)

    columns = [
        "zone_id",
        "zone_name",
        "hour",
        "time_period",
        "is_peak_hour",
        "zone_type",
        "transformer_capacity_kw",
        "base_grid_load_kw",
        "model_predicted_ev_load_kw",
        "model_total_load_kw",
        "model_load_ratio",
        "spare_capacity_after_ev_kw",
        "model_risk_level",
        "risk_score",
        "is_overloaded",
        "action_priority",
        "recommended_action",
        "risk_reason",
    ]
    return grid_stress[columns]


def create_schedule_outputs(grid_stress: Any) -> tuple[Any, Any, Any]:
    try:
        from src.models.step7_smart_charging_optimizer import (
            create_impact_summary,
            create_recommendations_and_curve,
        )
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Could not import scheduling engine: {exc}") from exc

    recommendations, load_curve = create_recommendations_and_curve(grid_stress)
    impact = create_impact_summary(recommendations, load_curve)
    return recommendations, load_curve, impact


def create_station_outputs(zone_summary: Any) -> tuple[Any, Any]:
    pd = load_pandas()
    try:
        from src.models.step8_station_location_recommender import (
            create_planning_summary,
            create_recommendations,
        )
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Could not import station engine: {exc}") from exc

    zones_path = PROCESSED_DIR / "zones_master.csv"
    grid_path = PROCESSED_DIR / "grid_capacity.csv"
    require_file(zones_path)
    require_file(grid_path)
    zones = pd.read_csv(zones_path)
    grid = pd.read_csv(grid_path)
    recommendations = create_recommendations(
        zones.merge(
            grid[
                [
                    "zone_id",
                    "zone_name",
                    "transformer_capacity_kw",
                    "current_peak_load_kw",
                    "spare_capacity_kw",
                    "load_ratio",
                    "grid_health_score",
                    "grid_capacity_status",
                ]
            ],
            on=["zone_id", "zone_name"],
            how="inner",
            validate="one_to_one",
        ).merge(
            zone_summary[
                [
                    "zone_id",
                    "zone_name",
                    "max_risk_score",
                    "avg_risk_score",
                    "max_model_load_ratio",
                    "critical_hours_count",
                    "high_hours_count",
                    "overloaded_hours_count",
                    "peak_hour_avg_risk_score",
                    "zone_priority",
                ]
            ],
            on=["zone_id", "zone_name"],
            how="inner",
            validate="one_to_one",
        )
    )
    return recommendations, create_planning_summary(recommendations)


def create_map_zones(zone_summary: Any, station_recs: Any) -> Any:
    pd = load_pandas()
    zones_path = PROCESSED_DIR / "zones_master.csv"
    require_file(zones_path)
    zones = pd.read_csv(zones_path)
    map_df = zones.merge(
        zone_summary[
            [
                "zone_id",
                "zone_name",
                "max_risk_score",
                "avg_risk_score",
                "critical_hours_count",
                "overloaded_hours_count",
                "zone_priority",
            ]
        ],
        on=["zone_id", "zone_name"],
        how="inner",
        validate="one_to_one",
    ).merge(
        station_recs[
            [
                "zone_id",
                "zone_name",
                "station_priority_score",
                "recommended_station_type",
                "recommended_chargers",
                "capital_planning_flag",
            ]
        ],
        on=["zone_id", "zone_name"],
        how="inner",
        validate="one_to_one",
    )
    return map_df


def create_summary(
    demand: Any,
    zone_summary: Any,
    schedules: Any,
    impact: Any,
    stations: Any,
    station_planning: Any,
) -> dict[str, Any]:
    metrics = read_json_file(MODEL_METRICS_PATH)
    zones_count = int(demand["zone_id"].nunique())
    existing_chargers = int(demand.drop_duplicates("zone_id")["existing_chargers"].sum())
    impact_row = impact.iloc[0]
    station_row = station_planning.iloc[0]
    highest_risk = zone_summary.sort_values(["max_risk_score", "avg_risk_score"], ascending=False).iloc[0]

    summary = {
        "total_zones": zones_count,
        "total_existing_chargers": existing_chargers,
        "total_recommended_chargers": int(station_row["total_recommended_chargers"]),
        "model_r2_score": round(float(metrics.get("r2_score", 0)), 2),
        "model_mae": round(float(metrics.get("mae", 0)), 2),
        "model_rmse": round(float(metrics.get("rmse", 0)), 2),
        "peak_load_reduction_percent": round(float(impact_row["peak_load_reduction_percent"]), 2),
        "critical_hours_before": int(impact_row["before_critical_hours"]),
        "critical_hours_after": int(impact_row["after_critical_hours"]),
        "overloaded_hours_before": int(impact_row["before_overloaded_hours"]),
        "overloaded_hours_after": int(impact_row["after_overloaded_hours"]),
        "total_scheduling_recommendations": int(len(schedules)),
        "invest_now_zones": int(station_row["invest_now_zones"]),
        "fast_charging_hubs": int(station_row["fast_charging_hubs"]),
        "top_priority_station_zone": str(station_row["top_priority_zone"]),
        "highest_risk_zone": str(highest_risk["zone_name"]),
        "model_version": "v2 expanded synthetic",
        "training_rows": int(metrics.get("row_count", 0)),
    }
    summary["critical_risk_before"] = summary["critical_hours_before"]
    summary["critical_risk_after"] = summary["critical_hours_after"]
    summary["overload_risk_before"] = summary["overloaded_hours_before"]
    summary["overload_risk_after"] = summary["overloaded_hours_after"]
    return summary


def create_explainability(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = read_json_file(MODEL_METRICS_PATH)
    return {
        "model_name": metrics["model_name"],
        "target_variable": metrics["target_column"],
        "input_features": metrics["feature_columns"],
        "model_r2_score": round(float(metrics["r2_score"]), 2),
        "mae": round(float(metrics["mae"]), 2),
        "rmse": round(float(metrics["rmse"]), 2),
        "baseline_comparison": {
            "baseline_mae": round(float(metrics["baseline_mae"]), 2),
            "baseline_rmse": round(float(metrics["baseline_rmse"]), 2),
            "mae_improvement_percent": metrics["improvement_over_baseline_mae_percent"],
            "rmse_improvement_percent": metrics["improvement_over_baseline_rmse_percent"],
        },
        "why_model_is_useful": (
            "The API runs the saved v2 model against current feature rows to create "
            "the demand signal used for downstream decisions."
        ),
        "recommendation_logic": {
            "grid_stress": "Live model-predicted EV load is added to base grid load and compared with transformer capacity.",
            "scheduling": "High and Critical evening load is shifted to off-peak hours where post-shift load stays below 90% capacity.",
            "station_planning": "Zones are ranked using demand, growth, charger gap, traffic, grid spare capacity, grid health, stress, and overload counts.",
        },
        "explainability_examples": [
            {
                "type": "Grid risk explanation",
                "message": f"{summary['highest_risk_zone']} is highlighted by the dynamic API because its predicted load ratio is highest.",
            },
            {
                "type": "Scheduling explanation",
                "message": f"Critical risk reduces from {summary['critical_hours_before']} to {summary['critical_hours_after']} after shifting flexible charging.",
            },
        ],
    }


def create_storyline(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "title": "EV Demand Forecasting",
            "message": "The backend runs the v2 demand model and returns hourly EV load forecasts by zone.",
            "key_metric": f"Model R2: {summary['model_r2_score']}",
        },
        {
            "step": 2,
            "title": "Grid Stress Detection",
            "message": "Predicted demand is combined with transformer capacity to identify High and Critical risk periods.",
            "key_metric": f"Highest risk zone: {summary['highest_risk_zone']}",
        },
        {
            "step": 3,
            "title": "Smart Charging Optimization",
            "message": "The API derives off-peak charging recommendations from the latest predicted grid stress.",
            "key_metric": f"Critical risk reduced from {summary['critical_hours_before']} to {summary['critical_hours_after']}",
        },
        {
            "step": 4,
            "title": "Infrastructure Planning",
            "message": "Station planning is recalculated from dynamic demand, stress, and grid feasibility signals.",
            "key_metric": f"{summary['total_recommended_chargers']} chargers recommended",
        },
    ]


def dynamic_snapshot() -> dict[str, Any]:
    demand = create_demand_forecast()
    grid_stress = create_grid_stress(demand)
    zone_summary = create_zone_summary(grid_stress)
    schedules, load_curve, impact = create_schedule_outputs(grid_stress)
    stations, station_planning = create_station_outputs(zone_summary)
    map_zones = create_map_zones(zone_summary, stations)
    summary = create_summary(demand, zone_summary, schedules, impact, stations, station_planning)

    high_alerts = grid_stress[
        grid_stress["model_risk_level"].isin(["High", "Critical"])
    ].sort_values(["risk_score", "model_load_ratio"], ascending=False)

    return {
        "summary": summary,
        "zones": dataframe_records(map_zones),
        "demand": dataframe_records(demand),
        "loadCurve": dataframe_records(load_curve),
        "alerts": dataframe_records(high_alerts),
        "schedules": dataframe_records(schedules),
        "stations": dataframe_records(stations),
        "explainability": create_explainability(summary),
        "storyline": create_storyline(summary),
        "schedulingImpact": dataframe_records(impact)[0] if len(impact) else {},
        "stationPlanning": dataframe_records(station_planning)[0] if len(station_planning) else {},
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "infrawisely-api",
        "mode": "dynamic-model-derived",
        "canonical_model_available": MODEL_PATH.exists(),
        "source_features_available": (PROCESSED_DIR / "hourly_demand.csv").exists(),
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "InfraWisely API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/model")
def model_metadata() -> dict[str, Any]:
    metrics = read_json_file(MODEL_METRICS_PATH)
    return {
        "model_path": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "metrics_path": str(MODEL_METRICS_PATH.relative_to(ROOT_DIR)),
        "metrics": metrics,
        "canonical_feature_columns": CANONICAL_MODEL_FEATURES,
    }


@app.get("/api/dashboard-data")
def dashboard_data() -> dict[str, Any]:
    """Return one coherent dynamic snapshot for the frontend."""
    return dynamic_snapshot()


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    return dynamic_snapshot()["summary"]


@app.get("/api/zones")
def zones() -> list[dict[str, Any]]:
    pd = load_pandas()
    path = PROCESSED_DIR / "zones_master.csv"
    require_file(path)
    return dataframe_records(pd.read_csv(path))


@app.get("/api/demand-forecast")
def demand_forecast(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    hour: Optional[int] = Query(default=None, ge=0, le=23),
) -> list[dict[str, Any]]:
    records = dynamic_snapshot()["demand"]
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, hour=hour)


@app.get("/api/canonical-demand-forecast")
def canonical_demand_forecast(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    hour: Optional[int] = Query(default=None, ge=0, le=23),
) -> list[dict[str, Any]]:
    records = dynamic_snapshot()["demand"]
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, hour=hour)


@app.get("/api/grid-stress-alerts")
def grid_stress_alerts(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=480),
) -> list[dict[str, Any]]:
    records = dynamic_snapshot()["alerts"]
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, risk_level=risk_level, limit=limit)


@app.get("/api/scheduling-recommendations")
def scheduling_recommendations(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=500),
) -> list[dict[str, Any]]:
    records = dynamic_snapshot()["schedules"]
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, risk_level=risk_level, limit=limit)


@app.get("/api/scheduling-impact")
def scheduling_impact() -> dict[str, Any]:
    return dynamic_snapshot()["schedulingImpact"]


@app.get("/api/load-curve")
def load_curve(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    hour: Optional[int] = Query(default=None, ge=0, le=23),
) -> list[dict[str, Any]]:
    records = dynamic_snapshot()["loadCurve"]
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, hour=hour)


@app.get("/api/station-recommendations")
def station_recommendations(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    records = dynamic_snapshot()["stations"]
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, limit=limit)


@app.get("/api/station-planning-summary")
def station_planning_summary() -> dict[str, Any]:
    return dynamic_snapshot()["stationPlanning"]


@app.get("/api/map-zones")
def map_zones() -> list[dict[str, Any]]:
    return dynamic_snapshot()["zones"]


@app.get("/api/model-explainability")
def model_explainability() -> dict[str, Any]:
    return dynamic_snapshot()["explainability"]


@app.get("/api/demo-storyline")
def demo_storyline() -> list[dict[str, Any]]:
    return dynamic_snapshot()["storyline"]


@app.post("/api/predict-demand", response_model=PredictDemandResponse)
def predict_demand(payload: PredictDemandRequest) -> PredictDemandResponse:
    pd = load_pandas()
    model = load_model()
    feature_row = payload.dict()
    frame = pd.DataFrame([{feature: feature_row[feature] for feature in CANONICAL_MODEL_FEATURES}])
    prediction = float(model.predict(frame)[0])

    return PredictDemandResponse(
        model_version="v2 expanded synthetic",
        model_predicted_ev_load_kw=round(prediction, 2),
        feature_columns=CANONICAL_MODEL_FEATURES,
    )
