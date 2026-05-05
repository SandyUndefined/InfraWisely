const DATA_DIR = "../data/dashboard";

const state = {
  data: {},
  selectedZone: "All Zones",
};

const files = {
  summary: "summary_metrics.json",
  demand: "demand_forecast.json",
  schedules: "scheduling_recommendations.json",
  loadCurve: "load_curve_before_after.json",
  stations: "station_recommendations.json",
  mapZones: "map_zones.json",
  storyline: "demo_storyline.json",
  explainability: "model_explainability_summary.json",
};

function fmt(value, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (typeof value === "number") {
    return `${Number.isInteger(value) ? value : value.toFixed(2)}${suffix}`;
  }
  return `${value}${suffix}`;
}

function formatHour(hour) {
  const value = Number(hour);
  const suffix = value < 12 ? "AM" : "PM";
  let display = value % 12;
  if (display === 0) display = 12;
  return `${display}:00 ${suffix}`;
}

function riskClass(risk) {
  const key = String(risk || "").toLowerCase();
  if (key.includes("critical")) return "risk-critical";
  if (key.includes("high")) return "risk-high";
  if (key.includes("medium")) return "risk-medium";
  return "risk-low";
}

function riskColor(score) {
  if (score >= 90) return "#c9493d";
  if (score >= 75) return "#d89119";
  return "#2e9d62";
}

async function loadJson(name) {
  const response = await fetch(`${DATA_DIR}/${files[name]}`);
  if (!response.ok) {
    throw new Error(`Failed to load ${files[name]} (${response.status})`);
  }
  return response.json();
}

async function bootstrap() {
  try {
    const entries = await Promise.all(
      Object.keys(files).map(async (name) => [name, await loadJson(name)])
    );
    state.data = Object.fromEntries(entries);
    initializeControls();
    renderAll();
  } catch (error) {
    document.body.innerHTML = `
      <main class="main-content">
        <section class="panel">
          <h2>Could not load dashboard JSON</h2>
          <p>${error.message}</p>
          <p>Run this from the project root with <code>python -m http.server 8000</code>, then open <code>http://localhost:8000/frontend/</code>.</p>
        </section>
      </main>
    `;
  }
}

function initializeControls() {
  const zoneFilter = document.getElementById("zoneFilter");
  const zones = ["All Zones", ...state.data.mapZones.map((zone) => zone.zone_name).sort()];
  zoneFilter.innerHTML = zones
    .map((zone) => `<option value="${zone}">${zone}</option>`)
    .join("");
  zoneFilter.addEventListener("change", (event) => {
    state.selectedZone = event.target.value;
    renderAll();
  });

  document.getElementById("resetView").addEventListener("click", () => {
    state.selectedZone = "All Zones";
    zoneFilter.value = state.selectedZone;
    renderAll();
  });

  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
      link.classList.add("active");
    });
  });
}

function zoneScoped(records, zoneKey = "zone_name") {
  if (state.selectedZone === "All Zones") return records;
  return records.filter((record) => record[zoneKey] === state.selectedZone);
}

function renderAll() {
  renderKpis();
  renderMap();
  renderStoryline();
  renderDemandChart();
  renderLoadCurveChart();
  renderScheduleTable();
  renderStationsTable();
  renderExplainability();
}

function renderKpis() {
  const s = state.data.summary;
  let cards;

  if (state.selectedZone === "All Zones") {
    cards = [
      ["Zones Monitored", s.total_zones, "Bengaluru operating areas"],
      ["Model R2", s.model_r2_score, `${s.model_version || "model"} / MAE ${s.model_mae} kW`],
      ["Peak Reduction", `${s.peak_load_reduction_percent}%`, "After smart charging"],
      ["Training Rows", s.training_rows || 480, "Expanded final model data"],
      ["Recommended Chargers", s.total_recommended_chargers, `${s.fast_charging_hubs} fast hubs`],
      ["Top Station Zone", s.top_priority_station_zone, "Best grid-aware location"],
      ["Highest Risk Zone", s.highest_risk_zone, "Worst predicted stress"],
      ["Scheduling Plans", s.total_scheduling_recommendations, "Charging shifts generated"],
    ];
  } else {
    const zone = state.data.mapZones.find((item) => item.zone_name === state.selectedZone);
    const station = state.data.stations.find((item) => item.zone_name === state.selectedZone);
    const demandRows = zoneScoped(state.data.demand);
    const curveRows = zoneScoped(state.data.loadCurve);
    const scheduleRows = zoneScoped(state.data.schedules);
    const peakDemand = demandRows.reduce((best, row) => row.model_predicted_ev_load_kw > best.model_predicted_ev_load_kw ? row : best, demandRows[0]);
    const beforePeak = curveRows.filter((row) => row.hour >= 17 && row.hour <= 22).reduce((sum, row) => sum + row.before_total_load_kw, 0);
    const afterPeak = curveRows.filter((row) => row.hour >= 17 && row.hour <= 22).reduce((sum, row) => sum + row.after_total_load_kw, 0);
    const reduction = beforePeak ? ((beforePeak - afterPeak) / beforePeak) * 100 : 0;
    cards = [
      ["Selected Zone", state.selectedZone, zone ? `${zone.zone_type} zone` : "Bengaluru zone"],
      ["Peak EV Demand", fmt(peakDemand?.model_predicted_ev_load_kw || 0, " kW"), peakDemand ? `at ${peakDemand.hour_label || formatHour(peakDemand.hour)}` : "No demand data"],
      ["Peak Reduction", `${reduction.toFixed(2)}%`, "Selected zone only"],
      ["Station Plan", station?.recommended_station_type || "Monitor Only", station ? `${station.station_priority_score} score` : "No station data"],
      ["Recommended Chargers", station?.recommended_chargers ?? 0, `${zone?.existing_chargers ?? 0} existing chargers`],
      ["Schedule Rows", scheduleRows.length, "Shift recommendations for this zone"],
      ["Model R2", s.model_r2_score, `${s.model_version || "model"}`],
    ];
  }

  document.getElementById("kpiGrid").innerHTML = cards
    .map(
      ([label, value, note]) => `
        <div class="kpi-card">
          <div class="kpi-label">${label}</div>
          <div class="kpi-value">${value}</div>
          <div class="kpi-note">${note}</div>
        </div>
      `
    )
    .join("");
}

function renderMap() {
  const map = document.getElementById("zoneMap");
  const zones = state.data.mapZones;
  const minLat = Math.min(...zones.map((zone) => zone.latitude));
  const maxLat = Math.max(...zones.map((zone) => zone.latitude));
  const minLng = Math.min(...zones.map((zone) => zone.longitude));
  const maxLng = Math.max(...zones.map((zone) => zone.longitude));

  const points = zones
    .map((zone) => {
      const x = 8 + ((zone.longitude - minLng) / (maxLng - minLng)) * 84;
      const y = 92 - ((zone.latitude - minLat) / (maxLat - minLat)) * 84;
      const selected = zone.zone_name === state.selectedZone;
      return `
        <button class="map-point ${selected ? "selected" : ""}" style="left:${x}%;top:${y}%;background:${riskColor(zone.max_risk_score)}" title="${zone.zone_name}: risk ${zone.max_risk_score}" data-zone="${zone.zone_name}" aria-label="${zone.zone_name}"></button>
        ${selected || zone.rank <= 5 ? `<span class="map-label" style="left:${x}%;top:${y}%">${zone.zone_name}</span>` : ""}
      `;
    })
    .join("");

  map.innerHTML = `
    <span class="map-axis" style="left:12px;top:10px">North Bengaluru</span>
    <span class="map-axis" style="right:12px;bottom:10px">East corridor</span>
    ${points}
  `;

  map.querySelectorAll(".map-point").forEach((point) => {
    point.addEventListener("click", () => {
      state.selectedZone = point.dataset.zone;
      document.getElementById("zoneFilter").value = state.selectedZone;
      renderAll();
    });
  });
}

function renderStoryline() {
  document.getElementById("storyline").innerHTML = state.data.storyline
    .map(
      (item) => `
        <article class="story-card">
          <div class="story-top">
            <h5>${item.step}. ${item.title}</h5>
            <span class="pill">${item.key_metric}</span>
          </div>
          <p>${item.message}</p>
        </article>
      `
    )
    .join("");
}

function renderDemandChart() {
  const records = zoneScoped(state.data.demand);
  const grouped = aggregateByHour(records, "model_predicted_ev_load_kw");
  const title = state.selectedZone === "All Zones" ? "System EV demand forecast" : `${state.selectedZone} EV demand forecast`;
  document.getElementById("demandChartTitle").textContent = title;
  const peak = grouped.reduce((best, row) => (row.value > best.value ? row : best), grouped[0]);
  document.getElementById("demandPeakLabel").textContent = `Peak ${formatHour(peak.hour)} / ${fmt(peak.value, " kW")}`;
  drawLineChart("demandChart", [
    { name: "Predicted EV load", color: "#0f8b8d", values: grouped },
  ], "kW");
}

function renderLoadCurveChart() {
  const records = zoneScoped(state.data.loadCurve);
  const before = aggregateByHour(records, "before_total_load_kw");
  const after = aggregateByHour(records, "after_total_load_kw");
  const title = state.selectedZone === "All Zones" ? "System load before vs after" : `${state.selectedZone} load before vs after`;
  document.getElementById("loadCurveTitle").textContent = title;
  drawLineChart("loadCurveChart", [
    { name: "Before", color: "#c9493d", values: before },
    { name: "After", color: "#2e9d62", values: after },
  ], "kW");
}

function aggregateByHour(records, key) {
  const groups = new Map();
  records.forEach((record) => {
    const hour = Number(record.hour);
    groups.set(hour, (groups.get(hour) || 0) + Number(record[key] || 0));
  });
  return [...groups.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([hour, value]) => ({ hour, value: Number(value.toFixed(2)) }));
}

function drawLineChart(containerId, series, unit) {
  const container = document.getElementById(containerId);
  const width = 900;
  const height = 310;
  const pad = { top: 18, right: 26, bottom: 38, left: 58 };
  const allValues = series.flatMap((line) => line.values.map((row) => row.value));
  const max = Math.max(...allValues) * 1.08;
  const min = 0;
  const x = (hour) => pad.left + (hour / 23) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - ((value - min) / (max - min)) * (height - pad.top - pad.bottom);

  const gridLines = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const gy = pad.top + ratio * (height - pad.top - pad.bottom);
    const label = max - ratio * (max - min);
    return `<line x1="${pad.left}" y1="${gy}" x2="${width - pad.right}" y2="${gy}" stroke="#dce5ee"/><text x="10" y="${gy + 4}" class="axis-text">${Math.round(label)}</text>`;
  }).join("");

  const paths = series.map((line) => {
    const d = line.values.map((row, index) => `${index === 0 ? "M" : "L"} ${x(row.hour)} ${y(row.value)}`).join(" ");
    const points = line.values.map((row) => `<circle cx="${x(row.hour)}" cy="${y(row.value)}" r="3" fill="${line.color}"><title>${line.name} ${formatHour(row.hour)} ${row.value} ${unit}</title></circle>`).join("");
    return `<path d="${d}" fill="none" stroke="${line.color}" stroke-width="3" stroke-linecap="round"/>${points}`;
  }).join("");

  const xLabels = [0, 4, 8, 12, 16, 20, 23].map((hour) => `<text x="${x(hour)}" y="${height - 12}" text-anchor="middle" class="axis-text">${formatHour(hour)}</text>`).join("");
  const legend = series.map((line, i) => `<g transform="translate(${pad.left + i * 120}, 8)"><rect width="12" height="12" rx="2" fill="${line.color}"/><text x="18" y="11" class="axis-text">${line.name}</text></g>`).join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${containerId}">
      ${gridLines}
      ${xLabels}
      ${paths}
      ${legend}
    </svg>
  `;
}

function renderScheduleTable() {
  const rows = zoneScoped(state.data.schedules).slice(0, 12);
  document.getElementById("scheduleTable").innerHTML = rows.length ? rows.map((row) => `
    <tr>
      <td><strong>${row.zone_name}</strong></td>
      <td>${row.peak_hour_label || formatHour(row.peak_hour)}</td>
      <td>${Math.round(row.recommended_shift_percent * 100)}% / ${fmt(row.allocated_shift_kw, " kW")}</td>
      <td>${row.recommended_offpeak_hours_label || row.recommended_offpeak_hours}</td>
      <td>${row.feasibility_status}</td>
      <td>${row.explanation}</td>
    </tr>
  `).join("") : `<tr><td colspan="6">No charging shift recommendation is needed for this selected zone.</td></tr>`;
}

function renderStationsTable() {
  const rows = zoneScoped(state.data.stations).slice(0, 14);
  document.getElementById("stationsTable").innerHTML = rows.length ? rows.map((row) => `
    <tr>
      <td>${row.rank}</td>
      <td><strong>${row.zone_name}</strong></td>
      <td>${fmt(row.station_priority_score)}</td>
      <td>${row.recommended_station_type}</td>
      <td>${row.recommended_chargers}</td>
      <td>${row.grid_feasibility_label}</td>
      <td>${row.capital_planning_flag}</td>
    </tr>
  `).join("") : `<tr><td colspan="7">No station planning record found for this selected zone.</td></tr>`;
}

function renderExplainability() {
  const explain = state.data.explainability;
  const examples = explain.explainability_examples.map((item) => `<li><strong>${item.type}:</strong> ${item.message}</li>`).join("");
  document.getElementById("explainabilityGrid").innerHTML = `
    <article class="explain-card">
      <h4>Model Performance</h4>
      <p><strong>${explain.model_name}</strong> predicts <code>${explain.target_variable}</code>.</p>
      <p>R2 ${fmt(explain.model_r2_score)}, MAE ${fmt(explain.mae, " kW")}, RMSE ${fmt(explain.rmse, " kW")}.</p>
      <p>MAE improved ${explain.baseline_comparison.mae_improvement_percent}% over baseline.</p>
    </article>
    <article class="explain-card">
      <h4>Decision Logic</h4>
      <p>${explain.why_model_is_useful}</p>
      <ul>
        <li>${explain.recommendation_logic.grid_stress}</li>
        <li>${explain.recommendation_logic.scheduling}</li>
        <li>${explain.recommendation_logic.station_planning}</li>
      </ul>
    </article>
    <article class="explain-card">
      <h4>Input Features</h4>
      <p>${explain.input_features.map((feature) => `<code>${feature}</code>`).join(", ")}</p>
    </article>
    <article class="explain-card">
      <h4>Explainability Examples</h4>
      <ul>${examples}</ul>
    </article>
  `;
}

bootstrap();
