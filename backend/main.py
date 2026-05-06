"""FastAPI backend for the InfraWisely EV charging optimization MVP."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DASHBOARD_DIR = DATA_DIR / "dashboard"
PROCESSED_DIR = DATA_DIR / "processed"
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

DASHBOARD_FILES = {
    "summary": "summary_metrics.json",
    "demand_forecast": "demand_forecast.json",
    "grid_stress_alerts": "grid_stress_alerts.json",
    "scheduling_recommendations": "scheduling_recommendations.json",
    "load_curve": "load_curve_before_after.json",
    "station_recommendations": "station_recommendations.json",
    "map_zones": "map_zones.json",
    "model_explainability": "model_explainability_summary.json",
    "demo_storyline": "demo_storyline.json",
}


app = FastAPI(
    title="InfraWisely API",
    description="Decision-support API for EV demand forecasting, grid stress, charging schedules, and station planning.",
    version="0.1.0",
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


def parse_csv_value(value: str) -> Any:
    """Convert CSV strings into JSON-friendly primitive types."""
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." not in value:
            return int(value)
        return float(value)
    except ValueError:
        return value


def read_json_file(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing file: {path.relative_to(ROOT_DIR)}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing file: {path.relative_to(ROOT_DIR)}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: parse_csv_value(value) for key, value in row.items()}
            for row in reader
        ]


def dashboard_payload(name: str) -> Any:
    return read_json_file(DASHBOARD_DIR / DASHBOARD_FILES[name])


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
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model file not found: {MODEL_PATH.relative_to(ROOT_DIR)}")
    try:
        import joblib
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="joblib is not installed. Install backend dependencies from requirements.txt.",
        ) from exc

    return joblib.load(MODEL_PATH)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "infrawisely-api",
        "dashboard_data_available": DASHBOARD_DIR.exists(),
        "canonical_model_available": MODEL_PATH.exists(),
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "InfraWisely API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/model")
def model_metadata() -> dict[str, Any]:
    metrics = read_json_file(MODEL_METRICS_PATH)
    return {
        "model_path": str(MODEL_PATH.relative_to(ROOT_DIR)),
        "metrics_path": str(MODEL_METRICS_PATH.relative_to(ROOT_DIR)),
        "metrics": metrics,
        "canonical_feature_columns": CANONICAL_MODEL_FEATURES,
    }


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    return dashboard_payload("summary")


@app.get("/api/zones")
def zones() -> list[dict[str, Any]]:
    return read_csv_file(PROCESSED_DIR / "zones_master.csv")


@app.get("/api/demand-forecast")
def demand_forecast(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    hour: Optional[int] = Query(default=None, ge=0, le=23),
) -> list[dict[str, Any]]:
    records = dashboard_payload("demand_forecast")
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, hour=hour)


@app.get("/api/canonical-demand-forecast")
def canonical_demand_forecast(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    hour: Optional[int] = Query(default=None, ge=0, le=23),
) -> list[dict[str, Any]]:
    """Serve v2 demand predictions from the canonical final model output CSV."""
    records = read_csv_file(PROCESSED_DIR / "final_hourly_demand_with_predictions.csv")
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, hour=hour)


@app.get("/api/grid-stress-alerts")
def grid_stress_alerts(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=480),
) -> list[dict[str, Any]]:
    records = dashboard_payload("grid_stress_alerts")
    return filter_records(
        records,
        zone_id=zone_id,
        zone_name=zone_name,
        risk_level=risk_level,
        limit=limit,
    )


@app.get("/api/scheduling-recommendations")
def scheduling_recommendations(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=500),
) -> list[dict[str, Any]]:
    records = dashboard_payload("scheduling_recommendations")
    return filter_records(
        records,
        zone_id=zone_id,
        zone_name=zone_name,
        risk_level=risk_level,
        limit=limit,
    )


@app.get("/api/scheduling-impact")
def scheduling_impact() -> dict[str, Any]:
    records = read_csv_file(PROCESSED_DIR / "scheduling_impact_summary.csv")
    return records[0] if records else {}


@app.get("/api/load-curve")
def load_curve(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    hour: Optional[int] = Query(default=None, ge=0, le=23),
) -> list[dict[str, Any]]:
    records = dashboard_payload("load_curve")
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, hour=hour)


@app.get("/api/station-recommendations")
def station_recommendations(
    zone_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    records = dashboard_payload("station_recommendations")
    return filter_records(records, zone_id=zone_id, zone_name=zone_name, limit=limit)


@app.get("/api/station-planning-summary")
def station_planning_summary() -> dict[str, Any]:
    records = read_csv_file(PROCESSED_DIR / "station_planning_summary.csv")
    return records[0] if records else {}


@app.get("/api/map-zones")
def map_zones() -> list[dict[str, Any]]:
    return dashboard_payload("map_zones")


@app.get("/api/model-explainability")
def model_explainability() -> dict[str, Any]:
    return dashboard_payload("model_explainability")


@app.get("/api/demo-storyline")
def demo_storyline() -> list[dict[str, Any]]:
    return dashboard_payload("demo_storyline")


@app.post("/api/predict-demand", response_model=PredictDemandResponse)
def predict_demand(payload: PredictDemandRequest) -> PredictDemandResponse:
    try:
        import pandas as pd
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="pandas is not installed. Install backend dependencies from requirements.txt.",
        ) from exc

    model = load_model()
    feature_row = payload.dict()
    frame = pd.DataFrame([{feature: feature_row[feature] for feature in CANONICAL_MODEL_FEATURES}])
    prediction = float(model.predict(frame)[0])

    return PredictDemandResponse(
        model_version="v2 expanded synthetic",
        model_predicted_ev_load_kw=round(prediction, 2),
        feature_columns=CANONICAL_MODEL_FEATURES,
    )
