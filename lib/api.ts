export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export type Summary = {
  total_zones: number;
  total_existing_chargers: number;
  total_recommended_chargers: number;
  model_r2_score: number;
  model_mae: number;
  model_rmse: number;
  peak_load_reduction_percent: number;
  critical_hours_before: number;
  critical_hours_after: number;
  overloaded_hours_before: number;
  overloaded_hours_after: number;
  total_scheduling_recommendations: number;
  invest_now_zones: number;
  fast_charging_hubs: number;
  top_priority_station_zone: string;
  highest_risk_zone: string;
  model_version?: string;
  training_rows?: number;
};

export type Zone = {
  zone_id: string;
  zone_name: string;
  latitude: number;
  longitude: number;
  zone_type: string;
  existing_chargers: number;
  ev_count_estimate: number;
  max_risk_score?: number;
  avg_risk_score?: number;
  critical_hours_count?: number;
  overloaded_hours_count?: number;
  zone_priority?: string;
  station_priority_score?: number;
  recommended_station_type?: string;
  recommended_chargers?: number;
  capital_planning_flag?: string;
};

export type DemandRow = {
  zone_id: string;
  zone_name: string;
  hour: number;
  time_period: string;
  is_peak_hour: boolean;
  zone_type: string;
  ev_count_estimate: number;
  traffic_score: number;
  existing_chargers: number;
  demand_growth_rate: number;
  transformer_capacity_kw: number;
  base_grid_load_kw: number;
  model_predicted_ev_load_kw?: number;
  final_model_predicted_ev_load_kw?: number;
  model_total_load_kw?: number;
  final_model_total_load_kw?: number;
  final_model_load_ratio?: number;
  final_model_risk_level?: string;
};

export type LoadCurveRow = {
  zone_id: string;
  zone_name: string;
  hour: number;
  time_period: string;
  transformer_capacity_kw: number;
  before_ev_load_kw: number;
  after_ev_load_kw: number;
  before_total_load_kw: number;
  after_total_load_kw: number;
  before_load_ratio: number;
  after_load_ratio: number;
  before_risk_level: string;
  after_risk_level: string;
};

export type StressAlert = {
  zone_id: string;
  zone_name: string;
  hour: number;
  time_period: string;
  model_total_load_kw: number;
  model_load_ratio: number;
  model_risk_level: string;
  risk_score: number;
  action_priority: string;
  risk_reason: string;
};

export type ScheduleRecommendation = {
  zone_id: string;
  zone_name: string;
  peak_hour: number;
  original_risk_level: string;
  recommended_shift_percent: number;
  shifted_load_kw: number;
  allocated_shift_kw: number;
  unallocated_shift_kw: number;
  recommended_offpeak_hours: string;
  feasibility_status: string;
  expected_peak_load_reduction_percent: number;
  explanation: string;
};

export type StationRecommendation = {
  rank: number;
  zone_id: string;
  zone_name: string;
  zone_type: string;
  ev_count_estimate: number;
  existing_chargers: number;
  spare_capacity_kw: number;
  avg_risk_score: number;
  overloaded_hours_count: number;
  station_priority_score: number;
  grid_feasibility_label: string;
  recommended_station_type: string;
  recommended_chargers: number;
  planning_priority: string;
  capital_planning_flag: string;
};

export type Explainability = {
  model_name: string;
  target_variable: string;
  input_features: string[];
  model_r2_score: number;
  mae: number;
  rmse: number;
  baseline_comparison: {
    mae_improvement_percent: number;
    rmse_improvement_percent: number;
  };
  recommendation_logic: {
    grid_stress: string;
    scheduling: string;
    station_planning: string;
  };
};

export type DashboardData = {
  summary: Summary;
  zones: Zone[];
  demand: DemandRow[];
  loadCurve: LoadCurveRow[];
  alerts: StressAlert[];
  schedules: ScheduleRecommendation[];
  stations: StationRecommendation[];
  explainability: Explainability;
};

export type PredictDemandRequest = {
  hour: number;
  time_period: string;
  is_peak_hour: boolean;
  zone_type: string;
  day_type: string;
  is_weekend: boolean;
  weather_condition: string;
  temperature_c: number;
  ev_count_estimate: number;
  traffic_score: number;
  traffic_score_hourly: number;
  existing_chargers: number;
  charger_utilization_proxy: number;
  demand_growth_rate: number;
  transformer_capacity_kw: number;
  base_grid_load_kw: number;
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function getDashboardData(): Promise<DashboardData> {
  const [
    summary,
    zones,
    demand,
    loadCurve,
    alerts,
    schedules,
    stations,
    explainability
  ] = await Promise.all([
    getJson<Summary>("/api/summary"),
    getJson<Zone[]>("/api/map-zones"),
    getJson<DemandRow[]>("/api/canonical-demand-forecast"),
    getJson<LoadCurveRow[]>("/api/load-curve"),
    getJson<StressAlert[]>("/api/grid-stress-alerts?limit=50"),
    getJson<ScheduleRecommendation[]>("/api/scheduling-recommendations?limit=200"),
    getJson<StationRecommendation[]>("/api/station-recommendations?limit=50"),
    getJson<Explainability>("/api/model-explainability")
  ]);

  return { summary, zones, demand, loadCurve, alerts, schedules, stations, explainability };
}

export async function predictDemand(payload: PredictDemandRequest) {
  const response = await fetch(`${API_BASE_URL}/api/predict-demand`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail || `/api/predict-demand returned ${response.status}`);
  }
  return response.json() as Promise<{
    model_version: string;
    model_predicted_ev_load_kw: number;
    feature_columns: string[];
  }>;
}

