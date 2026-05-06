# InfraWisely Backend

FastAPI service for the hackathon MVP. It serves the existing dashboard JSON/CSV artifacts and exposes a canonical v2 demand prediction endpoint.

## Run

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Main Endpoints

- `GET /api/health`
- `GET /api/summary`
- `GET /api/zones`
- `GET /api/demand-forecast`
- `GET /api/canonical-demand-forecast`
- `GET /api/grid-stress-alerts`
- `GET /api/scheduling-recommendations`
- `GET /api/scheduling-impact`
- `GET /api/load-curve`
- `GET /api/station-recommendations`
- `GET /api/station-planning-summary`
- `GET /api/map-zones`
- `GET /api/model-explainability`
- `GET /api/demo-storyline`
- `POST /api/predict-demand`

Example prediction request:

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
  "charger_utilization_proxy": 0.341,
  "demand_growth_rate": 0.324,
  "transformer_capacity_kw": 1307,
  "base_grid_load_kw": 900
}
```
