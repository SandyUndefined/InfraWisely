# GridCharge AI - Teammate Next Steps

## What Is Already Done By AI/ML

The AI/ML pipeline is complete for the prototype.

Completed:

1. Extracted BESCOM charger data from the official charger PDF.
2. Created zone-level Bengaluru master data.
3. Created synthetic masked grid capacity data.
4. Generated hourly EV demand data.
5. Trained the final EV demand prediction model.
6. Created grid stress scoring logic.
7. Created smart charging shift recommendations.
8. Created station location recommendations.
9. Exported dashboard/API-style sample JSON files.

Final model:

```text
models/ev_demand_model_v2.pkl
```

Final model metrics:

```text
models/ev_demand_model_v2_metrics.json
```

Main processed data:

```text
data/processed/zones_master.csv
data/processed/grid_capacity.csv
data/processed/final_expanded_hourly_training_data.csv
data/processed/final_hourly_demand_with_predictions.csv
data/processed/charging_schedule_recommendations.csv
data/processed/station_location_recommendations.csv
```

Sample dashboard/API JSON contracts:

```text
data/dashboard/summary_metrics.json
data/dashboard/demand_forecast.json
data/dashboard/scheduling_recommendations.json
data/dashboard/load_curve_before_after.json
data/dashboard/station_recommendations.json
data/dashboard/map_zones.json
data/dashboard/model_explainability_summary.json
data/dashboard/demo_storyline.json
```

Important note:

The JSON files in `data/dashboard/` are static sample outputs. They are not live API responses yet. Use them as the expected response format for the backend/frontend.

## What You Need To Build Next

Your role is frontend/backend integration.

Recommended next work:

1. Build backend API endpoints.
2. Load the AI/ML model from `models/ev_demand_model_v2.pkl`.
3. Serve processed CSV outputs through API endpoints.
4. Build frontend dashboard screens using these API responses.
5. Use the static JSON files as response examples while developing.

## Suggested Backend Endpoints

Use FastAPI or your preferred backend.

Recommended endpoints:

```text
GET /api/summary
GET /api/zones
GET /api/demand-forecast
GET /api/load-curve
GET /api/scheduling-recommendations
GET /api/station-recommendations
GET /api/map-zones
GET /api/model-explainability
```

Optional prediction endpoint:

```text
POST /api/predict-demand
```

This endpoint can load:

```text
models/ev_demand_model_v2.pkl
```

Input example:

```json
{
  "hour": 20,
  "time_period": "Evening Peak",
  "is_peak_hour": true,
  "zone_type": "mixed",
  "day_type": "Weekday",
  "is_weekend": false,
  "weather_condition": "Clear",
  "temperature_c": 28,
  "ev_count_estimate": 4770,
  "traffic_score": 90,
  "traffic_score_hourly": 95,
  "existing_chargers": 2,
  "charger_utilization_proxy": 0.34,
  "demand_growth_rate": 0.324,
  "transformer_capacity_kw": 1307,
  "base_grid_load_kw": 900
}
```

Output example:

```json
{
  "model_predicted_ev_load_kw": 642.25
}
```

## Suggested Frontend Screens

Build these screens/cards:

1. Executive KPI dashboard
2. Bengaluru zone map / heatmap
3. EV demand forecast chart
4. Before vs after load curve
5. Smart charging recommendations table
6. Station location recommendations table
7. Model explainability section

## Which Files Power Which Screen

KPI cards:

```text
data/dashboard/summary_metrics.json
```

Demand chart:

```text
data/dashboard/demand_forecast.json
```

Before/after load curve:

```text
data/dashboard/load_curve_before_after.json
```

Scheduling recommendations:

```text
data/dashboard/scheduling_recommendations.json
```

Station planning:

```text
data/dashboard/station_recommendations.json
```

Map markers:

```text
data/dashboard/map_zones.json
```

Explainability:

```text
data/dashboard/model_explainability_summary.json
data/dashboard/demo_storyline.json
```

## Important UI Wording

Avoid confusing terms like:

```text
zone-hours
critical alerts
overload alerts
```

Use simpler wording:

```text
Peak reduction
High-risk periods
Grid stress
Charging shift recommendations
Station planning
```

## Final Demo Story

Use this flow:

1. Show predicted EV charging demand by zone and hour.
2. Show how predicted demand creates grid stress when combined with transformer capacity.
3. Show before vs after load curve after smart charging optimization.
4. Show recommended zones for new chargers.
5. Explain that recommendations are grid-aware, so fast chargers are not blindly suggested in constrained areas.

## What Not To Rebuild

Do not retrain the model unless needed.

Do not recreate the AI/ML pipeline.

Use the model and outputs already provided:

```text
models/ev_demand_model_v2.pkl
data/processed/
data/dashboard/
```

## Quick Test

To preview the static dashboard included in this package:

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000/team_handoff_package/frontend/
```
