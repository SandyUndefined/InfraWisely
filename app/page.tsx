"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Building2,
  Loader2,
  MapPin,
  RefreshCw,
  Zap
} from "lucide-react";
import {
  DashboardData,
  DemandRow,
  ExistingCharger,
  LoadCurveRow,
  PredictDemandRequest,
  ScheduleRecommendation,
  StationRecommendation,
  StressAlert,
  getDashboardData,
  predictDemand
} from "@/lib/api";
import { cn, formatHour, formatNumber, formatTimeText } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type Point = {
  hour: number;
  value: number;
};

type Series = {
  name: string;
  color: string;
  points: Point[];
};

type BarDatum = {
  label: string;
  value: number;
  note?: string;
  color?: string;
};

type ActionPlan = {
  alert: StressAlert;
  schedule?: ScheduleRecommendation;
};

const ALL_ZONES = "All Zones";

function riskVariant(risk?: string) {
  const value = String(risk || "").toLowerCase();
  if (value.includes("critical")) return "danger" as const;
  if (value.includes("high")) return "warning" as const;
  if (value.includes("medium")) return "muted" as const;
  return "success" as const;
}

function riskColor(score = 0) {
  if (score >= 90) return "#c9493d";
  if (score >= 75) return "#d89119";
  return "#2e9d62";
}

function evLoad(row: DemandRow) {
  return row.final_model_predicted_ev_load_kw ?? row.model_predicted_ev_load_kw ?? 0;
}

function totalLoad(row: DemandRow) {
  return row.final_model_total_load_kw ?? row.model_total_load_kw ?? row.base_grid_load_kw + evLoad(row);
}

function loadRatio(row: DemandRow) {
  return row.final_model_load_ratio ?? row.model_load_ratio ?? totalLoad(row) / row.transformer_capacity_kw;
}

function scoped<T extends { zone_name: string }>(records: T[], zone: string) {
  return zone === ALL_ZONES ? records : records.filter((record) => record.zone_name === zone);
}

function aggregateByHour<T extends { hour: number }>(
  records: T[],
  getValue: (record: T) => number
): Point[] {
  const groups = new Map<number, number>();
  for (const record of records) {
    groups.set(record.hour, (groups.get(record.hour) || 0) + getValue(record));
  }
  return Array.from(groups.entries())
    .sort(([a], [b]) => a - b)
    .map(([hour, value]) => ({ hour, value: Number(value.toFixed(2)) }));
}

function averageByHour<T extends { hour: number }>(
  records: T[],
  getValue: (record: T) => number
): Point[] {
  const groups = new Map<number, { total: number; count: number }>();
  for (const record of records) {
    const current = groups.get(record.hour) || { total: 0, count: 0 };
    current.total += getValue(record);
    current.count += 1;
    groups.set(record.hour, current);
  }
  return Array.from(groups.entries())
    .sort(([a], [b]) => a - b)
    .map(([hour, group]) => ({ hour, value: Number((group.total / Math.max(group.count, 1)).toFixed(3)) }));
}

function peakPoint(points: Point[]) {
  return points.reduce((best, row) => (row.value > best.value ? row : best), { hour: 0, value: 0 });
}

function actionKey(row: StressAlert) {
  return `${row.zone_id}-${row.hour}`;
}

function scheduleForAlert(alert: StressAlert, schedules: ScheduleRecommendation[]) {
  return schedules.find((schedule) => schedule.zone_id === alert.zone_id && schedule.peak_hour === alert.hour);
}

function formatOffPeakWindow(schedule?: ScheduleRecommendation) {
  const raw = schedule?.recommended_offpeak_hours_label || schedule?.recommended_offpeak_hours;
  return formatTimeText(raw) || "No feasible off-peak window found";
}

function actionPlanText(plan: ActionPlan) {
  const schedule = plan.schedule;
  return [
    "InfraWisely Operator Action Plan",
    "",
    `Zone: ${plan.alert.zone_name}`,
    `Peak hour: ${formatHour(plan.alert.hour)}`,
    `Risk ratio: ${formatNumber(plan.alert.model_load_ratio)}`,
    `Why it is critical: ${formatTimeText(plan.alert.risk_reason)}`,
    `Recommended load shift: ${schedule ? formatNumber(schedule.allocated_shift_kw || schedule.shifted_load_kw, " kW") : "No matching schedule recommendation"}`,
    `Off-peak charging window: ${formatOffPeakWindow(schedule)}`,
    `Expected peak reduction: ${schedule ? `${formatNumber(schedule.expected_peak_load_reduction_kw, " kW")} / ${formatNumber(schedule.expected_peak_load_reduction_percent, "%")}` : "-"}`,
    "",
    "Operator note: Decision-support only. No grid system modification."
  ].join("\n");
}

function LineChart({ series, unit }: { series: Series[]; unit: string }) {
  const width = 900;
  const height = 280;
  const pad = { top: 24, right: 24, bottom: 36, left: 58 };
  const values = series.flatMap((line) => line.points.map((point) => point.value));
  const hours = Array.from(
    new Set(series.flatMap((line) => line.points.map((point) => point.hour)))
  ).sort((a, b) => a - b);
  const max = Math.max(1, ...values) * 1.08;
  const minHour = hours[0] ?? 0;
  const maxHour = hours[hours.length - 1] ?? 23;
  const hourSpan = Math.max(1, maxHour - minHour);
  const x = (hour: number) => pad.left + ((hour - minHour) / hourSpan) * (width - pad.left - pad.right);
  const y = (value: number) => height - pad.bottom - (value / max) * (height - pad.top - pad.bottom);
  const grid = [0, 0.25, 0.5, 0.75, 1];
  const tickHours = hours.filter((hour, index) => index === 0 || index === hours.length - 1 || index % 4 === 0);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-[280px] w-full">
      {grid.map((ratio) => {
        const gy = pad.top + ratio * (height - pad.top - pad.bottom);
        const label = max - ratio * max;
        const labelText = max < 10
          ? label.toFixed(2)
          : Math.round(label).toLocaleString("en-IN");
        return (
          <g key={ratio}>
            <line x1={pad.left} y1={gy} x2={width - pad.right} y2={gy} stroke="#d7dee7" />
            <text x="10" y={gy + 4} className="fill-slate-500 text-xs">
              {labelText}
            </text>
          </g>
        );
      })}
      {tickHours.map((hour) => (
        <text key={hour} x={x(hour)} y={height - 10} textAnchor="middle" className="fill-slate-500 text-xs">
          {formatHour(hour)}
        </text>
      ))}
      {series.map((line, index) => {
        const d = line.points
          .map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${x(point.hour)} ${y(point.value)}`)
          .join(" ");
        return (
          <g key={line.name}>
            <path d={d} fill="none" stroke={line.color} strokeWidth="3" strokeLinecap="round" />
            {line.points.map((point) => (
              <circle key={`${line.name}-${point.hour}`} cx={x(point.hour)} cy={y(point.value)} r="3" fill={line.color}>
                <title>{`${line.name} ${formatHour(point.hour)} ${point.value} ${unit}`}</title>
              </circle>
            ))}
            <g transform={`translate(${pad.left + index * 160}, 8)`}>
              <rect width="12" height="12" rx="3" fill={line.color} />
              <text x="18" y="11" className="fill-slate-600 text-xs">
                {line.name}
              </text>
            </g>
          </g>
        );
      })}
    </svg>
  );
}

function BarChart({ data, unit }: { data: BarDatum[]; unit: string }) {
  const max = Math.max(1, ...data.map((item) => item.value));

  return (
    <div className="space-y-3">
      {data.map((item) => {
        const widthPercent = Math.max(3, (item.value / max) * 100);
        return (
          <div key={item.label} className="grid grid-cols-[minmax(120px,180px)_1fr] items-center gap-3 text-sm">
            <div className="min-w-0">
              <p className="truncate font-medium">{item.label}</p>
              {item.note ? <p className="truncate text-xs text-muted-foreground">{item.note}</p> : null}
            </div>
            <div className="min-w-0">
              <div className="grid grid-cols-[1fr_auto] items-center gap-2">
                <div className="h-8 rounded-md bg-muted">
                  <div
                    className="h-8 rounded-md"
                    style={{ width: `${widthPercent}%`, backgroundColor: item.color || "#0f8b8d" }}
                  />
                </div>
                <span className="w-20 text-right text-xs font-semibold">{formatNumber(item.value, unit)}</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ZoneMap({
  zones,
  selectedZone,
  onSelect
}: {
  zones: DashboardData["zones"];
  selectedZone: string;
  onSelect: (zone: string) => void;
}) {
  const mapElementRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<import("leaflet").Map | null>(null);
  const markerLayerRef = useRef<import("leaflet").LayerGroup | null>(null);
  const leafletRef = useRef<typeof import("leaflet") | null>(null);
  const onSelectRef = useRef(onSelect);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    onSelectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    let cancelled = false;

    async function initMap() {
      if (!mapElementRef.current || mapRef.current) return;
      const L = await import("leaflet");
      if (cancelled || !mapElementRef.current) return;

      leafletRef.current = L;
      const map = L.map(mapElementRef.current, {
        attributionControl: true,
        scrollWheelZoom: false,
        zoomControl: true
      }).setView([12.9716, 77.5946], 11);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
      }).addTo(map);

      mapRef.current = map;
      markerLayerRef.current = L.layerGroup().addTo(map);
      setReady(true);
      window.setTimeout(() => map.invalidateSize(), 0);
    }

    void initMap();

    return () => {
      cancelled = true;
      mapRef.current?.remove();
      mapRef.current = null;
      markerLayerRef.current = null;
      leafletRef.current = null;
    };
  }, []);

  useEffect(() => {
    const L = leafletRef.current;
    const map = mapRef.current;
    const markerLayer = markerLayerRef.current;
    if (!ready || !L || !map || !markerLayer) return;

    markerLayer.clearLayers();
    const bounds: [number, number][] = [];
    let selectedCenter: [number, number] | null = null;

    zones.forEach((zone) => {
      if (!Number.isFinite(zone.latitude) || !Number.isFinite(zone.longitude)) return;

      const selected = selectedZone === zone.zone_name;
      const score = zone.max_risk_score ?? 0;
      const color = riskColor(score);
      const center: [number, number] = [zone.latitude, zone.longitude];
      bounds.push(center);
      if (selected) selectedCenter = center;

      const icon = L.divIcon({
        className: "",
        iconSize: selected ? [24, 24] : [18, 18],
        iconAnchor: selected ? [12, 12] : [9, 9],
        html: `<span class="zone-marker${selected ? " zone-marker-selected" : ""}" style="background:${color}"></span>`
      });

      const marker = L.marker(center, { icon })
        .addTo(markerLayer)
        .bindPopup(
          `<strong>${zone.zone_name}</strong><br/>Risk score: ${score}<br/>Station: ${
            zone.recommended_station_type || "Monitor"
          }<br/>Existing chargers: ${zone.existing_chargers}<br/>Recommended chargers: ${zone.recommended_chargers ?? 0}`
        )
        .bindTooltip(zone.zone_name, {
          direction: "top",
          offset: [0, -12],
          opacity: 0.95,
          permanent: selected || (zone.station_priority_score ?? 0) >= 70,
          className: "zone-tooltip"
        });

      marker.on("click", () => onSelectRef.current(zone.zone_name));
    });

    if (selectedCenter) {
      map.setView(selectedCenter, 13, { animate: true });
    } else if (bounds.length) {
      map.fitBounds(bounds, { padding: [28, 28], maxZoom: 12 });
    }

    window.setTimeout(() => map.invalidateSize(), 0);
  }, [ready, selectedZone, zones]);

  return (
    <div className="relative h-[380px] overflow-hidden rounded-lg border border-border bg-muted">
      <div ref={mapElementRef} className="h-full w-full" />
      {!ready ? (
        <div className="absolute inset-0 flex items-center justify-center bg-card/80">
          <div className="flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm shadow-sm">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            Loading map
          </div>
        </div>
      ) : null}
      <div className="absolute bottom-3 left-3 z-[500] flex gap-2 text-xs">
        <Badge variant="success">Low</Badge>
        <Badge variant="warning">High</Badge>
        <Badge variant="danger">Critical</Badge>
      </div>
    </div>
  );
}

function KpiCard({
  label,
  value,
  note,
  icon: Icon,
  onClick
}: {
  label: string;
  value: string | number;
  note: string;
  icon: typeof Activity;
  onClick?: () => void;
}) {
  return (
    <Card
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={(event) => {
        if (!onClick) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick();
        }
      }}
      className={cn(onClick && "cursor-pointer transition hover:border-primary hover:shadow-md focus:outline-none focus:ring-2 focus:ring-ring")}
    >
      <CardContent className="flex min-h-[112px] items-start justify-between gap-3 p-4">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase text-muted-foreground">{label}</p>
          <p className="mt-2 break-words text-2xl font-semibold">{value}</p>
          <p className="mt-1 text-sm text-muted-foreground">{note}</p>
        </div>
        <span className="rounded-md bg-accent p-2 text-accent-foreground">
          <Icon className="h-4 w-4" />
        </span>
      </CardContent>
    </Card>
  );
}

function ChargerDetailsModal({
  zoneLabel,
  rows,
  onClose
}: {
  zoneLabel: string;
  rows: ExistingCharger[];
  onClose: () => void;
}) {
  const displayedRows = rows.slice(0, 80);

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-slate-950/45 p-4">
      <div className="flex max-h-[86vh] w-full max-w-4xl flex-col rounded-lg border border-border bg-card shadow-xl">
        <div className="flex items-start justify-between gap-4 border-b border-border p-5">
          <div>
            <div className="flex items-center gap-2">
              <MapPin className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Existing Chargers</h2>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              {zoneLabel} / {rows.length} charger{rows.length === 1 ? "" : "s"}
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>

        <div className="overflow-auto p-5">
          {displayedRows.length ? (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Location</TableHead>
                  <TableHead>Station</TableHead>
                  <TableHead>Code</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Source</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {displayedRows.map((row) => (
                  <TableRow key={`${row.sl_no}-${row.charger_code}`}>
                    <TableCell className="font-medium">{row.area}</TableCell>
                    <TableCell className="min-w-[220px]">{row.station_name}</TableCell>
                    <TableCell>{row.charger_code || "-"}</TableCell>
                    <TableCell className="min-w-[170px]">{row.charger_model || "-"}</TableCell>
                    <TableCell className="text-right">Page {row.source_page}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="rounded-md border border-border bg-muted p-4 text-sm text-muted-foreground">
              No mapped BESCOM charger records found for this selected zone.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StressTable({
  rows,
  schedules,
  reviewedKeys,
  onOpenActionPlan
}: {
  rows: StressAlert[];
  schedules: ScheduleRecommendation[];
  reviewedKeys: Set<string>;
  onOpenActionPlan: (plan: ActionPlan) => void;
}) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Zone</TableHead>
            <TableHead>Hour</TableHead>
            <TableHead>Risk</TableHead>
            <TableHead className="text-right">Ratio</TableHead>
            <TableHead>Action</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.slice(0, 10).map((row) => {
            const key = actionKey(row);
            const reviewed = reviewedKeys.has(key);
            return (
              <TableRow key={key}>
                <TableCell className="font-medium">{row.zone_name}</TableCell>
                <TableCell>{formatHour(row.hour)}</TableCell>
                <TableCell>
                  <Badge variant={riskVariant(row.model_risk_level)}>{row.model_risk_level}</Badge>
                </TableCell>
                <TableCell className="text-right">{formatNumber(row.model_load_ratio)}</TableCell>
                <TableCell className="min-w-[210px]">
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => onOpenActionPlan({ alert: row, schedule: scheduleForAlert(row, schedules) })}
                    >
                      View Action Plan
                    </Button>
                    {reviewed ? <Badge variant="success">Reviewed</Badge> : null}
                  </div>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

function ActionPlanModal({
  plan,
  reviewed,
  onClose,
  onMarkReviewed
}: {
  plan: ActionPlan;
  reviewed: boolean;
  onClose: () => void;
  onMarkReviewed: () => void;
}) {
  const schedule = plan.schedule;

  function exportReport() {
    const blob = new Blob([actionPlanText(plan)], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `action-plan-${plan.alert.zone_id}-${plan.alert.hour}.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-slate-950/45 p-4">
      <div className="w-full max-w-2xl rounded-lg border border-border bg-card shadow-xl">
        <div className="flex items-start justify-between gap-4 border-b border-border p-5">
          <div>
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-600" />
              <h2 className="text-lg font-semibold">Operator Action Plan</h2>
            </div>
            <p className="mt-1 text-sm text-muted-foreground">
              {plan.alert.zone_name} / {formatHour(plan.alert.hour)}
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>

        <div className="grid gap-4 p-5 sm:grid-cols-2">
          <div className="rounded-md border border-border p-3">
            <p className="text-xs font-semibold uppercase text-muted-foreground">Zone name</p>
            <p className="mt-1 font-semibold">{plan.alert.zone_name}</p>
          </div>
          <div className="rounded-md border border-border p-3">
            <p className="text-xs font-semibold uppercase text-muted-foreground">Peak hour</p>
            <p className="mt-1 font-semibold">{formatHour(plan.alert.hour)}</p>
          </div>
          <div className="rounded-md border border-border p-3">
            <p className="text-xs font-semibold uppercase text-muted-foreground">Risk ratio</p>
            <p className="mt-1 font-semibold">{formatNumber(plan.alert.model_load_ratio)}</p>
          </div>
          <div className="rounded-md border border-border p-3">
            <p className="text-xs font-semibold uppercase text-muted-foreground">Recommended load shift</p>
            <p className="mt-1 font-semibold">
              {schedule ? formatNumber(schedule.allocated_shift_kw || schedule.shifted_load_kw, " kW") : "-"}
            </p>
          </div>
          <div className="rounded-md border border-border p-3">
            <p className="text-xs font-semibold uppercase text-muted-foreground">Off-peak charging window</p>
            <p className="mt-1 font-semibold">{formatOffPeakWindow(schedule)}</p>
          </div>
          <div className="rounded-md border border-border p-3">
            <p className="text-xs font-semibold uppercase text-muted-foreground">Expected peak reduction</p>
            <p className="mt-1 font-semibold">
              {schedule
                ? `${formatNumber(schedule.expected_peak_load_reduction_kw, " kW")} / ${formatNumber(schedule.expected_peak_load_reduction_percent, "%")}`
                : "-"}
            </p>
          </div>
          <div className="rounded-md border border-border p-3 sm:col-span-2">
            <p className="text-xs font-semibold uppercase text-muted-foreground">Why it is critical</p>
            <p className="mt-2 text-sm leading-6">{formatTimeText(plan.alert.risk_reason)}</p>
          </div>
          <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900 sm:col-span-2">
            Operator note: Decision-support only. No grid system modification.
          </div>
        </div>

        <div className="flex flex-col-reverse gap-2 border-t border-border p-5 sm:flex-row sm:justify-end">
          <Button variant="outline" onClick={exportReport}>
            Export Report
          </Button>
          <Button onClick={onMarkReviewed} disabled={reviewed}>
            Mark Reviewed
          </Button>
        </div>
      </div>
    </div>
  );
}

function ScheduleTable({ rows }: { rows: ScheduleRecommendation[] }) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Zone</TableHead>
            <TableHead>Peak</TableHead>
            <TableHead>Risk</TableHead>
            <TableHead className="text-right">Shift</TableHead>
            <TableHead>Off-Peak Window</TableHead>
            <TableHead>Feasibility</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.slice(0, 12).map((row) => (
            <TableRow key={`${row.zone_id}-${row.peak_hour}`}>
              <TableCell className="font-medium">{row.zone_name}</TableCell>
              <TableCell>{formatHour(row.peak_hour)}</TableCell>
              <TableCell>
                <Badge variant={riskVariant(row.original_risk_level)}>{row.original_risk_level}</Badge>
              </TableCell>
              <TableCell className="text-right">{formatNumber(row.allocated_shift_kw, " kW")}</TableCell>
              <TableCell className="min-w-[170px]">{formatOffPeakWindow(row)}</TableCell>
              <TableCell>{row.feasibility_status}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function StationTable({ rows }: { rows: StationRecommendation[] }) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Rank</TableHead>
            <TableHead>Zone</TableHead>
            <TableHead className="text-right">Score</TableHead>
            <TableHead>Station Type</TableHead>
            <TableHead className="text-right">Chargers</TableHead>
            <TableHead>Grid</TableHead>
            <TableHead>Flag</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.slice(0, 12).map((row) => (
            <TableRow key={row.zone_id}>
              <TableCell>{row.rank}</TableCell>
              <TableCell className="font-medium">{row.zone_name}</TableCell>
              <TableCell className="text-right">{formatNumber(row.station_priority_score)}</TableCell>
              <TableCell className="min-w-[190px]">{row.recommended_station_type}</TableCell>
              <TableCell className="text-right">{row.recommended_chargers}</TableCell>
              <TableCell>{row.grid_feasibility_label}</TableCell>
              <TableCell>{row.capital_planning_flag}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function predictionPayloadFrom(row: DemandRow | undefined): PredictDemandRequest | null {
  if (!row) return null;
  return {
    hour: row.hour,
    time_period: row.time_period,
    is_peak_hour: row.is_peak_hour,
    zone_type: row.zone_type,
    day_type: "Weekday",
    is_weekend: false,
    weather_condition: "Clear",
    temperature_c: 28,
    ev_count_estimate: row.ev_count_estimate,
    traffic_score: row.traffic_score,
    traffic_score_hourly: row.traffic_score,
    existing_chargers: row.existing_chargers,
    charger_utilization_proxy: Math.min(row.ev_count_estimate / Math.max(row.existing_chargers, 1) / 7000, 1),
    demand_growth_rate: row.demand_growth_rate,
    transformer_capacity_kw: row.transformer_capacity_kw,
    base_grid_load_kw: row.base_grid_load_kw
  };
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [selectedZone, setSelectedZone] = useState(ALL_ZONES);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [activeActionPlan, setActiveActionPlan] = useState<ActionPlan | null>(null);
  const [showChargerDetails, setShowChargerDetails] = useState(false);
  const [reviewedActionKeys, setReviewedActionKeys] = useState<Set<string>>(() => new Set());

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setData(await getDashboardData());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load dashboard data");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const zoneOptions = useMemo(() => {
    if (!data) return [ALL_ZONES];
    return [ALL_ZONES, ...data.zones.map((zone) => zone.zone_name).sort()];
  }, [data]);

  const scopedDemand = useMemo(() => scoped(data?.demand || [], selectedZone), [data, selectedZone]);
  const scopedCurve = useMemo(() => scoped(data?.loadCurve || [], selectedZone), [data, selectedZone]);
  const scopedAlerts = useMemo(() => scoped(data?.alerts || [], selectedZone), [data, selectedZone]);
  const scopedSchedules = useMemo(() => scoped(data?.schedules || [], selectedZone), [data, selectedZone]);
  const scopedStations = useMemo(() => scoped(data?.stations || [], selectedZone), [data, selectedZone]);
  const scopedChargers = useMemo(() => {
    const chargers = data?.chargers || [];
    if (selectedZone === ALL_ZONES) {
      return chargers.filter((charger) => charger.area !== "Unknown");
    }
    return chargers.filter((charger) => charger.area === selectedZone);
  }, [data, selectedZone]);

  const demandPoints = useMemo(() => aggregateByHour(scopedDemand, evLoad), [scopedDemand]);
  const riskRatioPoints = useMemo(() => averageByHour(scopedDemand, loadRatio), [scopedDemand]);
  const loadBefore = useMemo(() => aggregateByHour(scopedCurve, (row: LoadCurveRow) => row.before_total_load_kw), [scopedCurve]);
  const loadAfter = useMemo(() => aggregateByHour(scopedCurve, (row: LoadCurveRow) => row.after_total_load_kw), [scopedCurve]);
  const peakDemand = peakPoint(demandPoints);
  const peakRiskRatio = peakPoint(riskRatioPoints);
  const peakBefore = peakPoint(loadBefore);
  const peakAfter = peakPoint(loadAfter);
  const selectedDemandRow = scopedDemand.find((row) => row.hour === peakDemand.hour) || scopedDemand[0];

  async function runPrediction() {
    const payload = predictionPayloadFrom(selectedDemandRow);
    if (!payload) return;
    setPredicting(true);
    setPredictionError(null);
    try {
      const result = await predictDemand(payload);
      setPrediction(result.model_predicted_ev_load_kw);
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  }

  if (loading) {
    return (
      <main className="flex min-h-screen items-center justify-center">
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-3 shadow-sm">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <span className="text-sm font-medium">Loading InfraWisely dashboard</span>
        </div>
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="mx-auto flex min-h-screen max-w-3xl items-center justify-center p-6">
        <Card className="w-full">
          <CardHeader>
            <CardTitle>API connection failed</CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => void load()}>
              <RefreshCw className="h-4 w-4" />
              Retry
            </Button>
          </CardContent>
        </Card>
      </main>
    );
  }

  const summary = data.summary;
  const selectedStation = scopedStations[0];
  const topStation = data.stations[0];
  const selectedZoneRecord = data.zones.find((zone) => zone.zone_name === selectedZone);
  const zoneLabel = selectedZone === ALL_ZONES ? `${summary.total_zones} zones` : selectedZone;
  const chartScopeLabel = selectedZone === ALL_ZONES ? "System" : selectedZone;
  const coverageNote = selectedZone === ALL_ZONES
    ? `${summary.total_existing_chargers} mapped existing chargers`
    : `${selectedZoneRecord?.existing_chargers ?? scopedChargers.length} existing chargers`;
  const peakShiftKw = Math.max(0, peakBefore.value - peakAfter.value);
  const criticalDelta = `${summary.critical_hours_before} -> ${summary.critical_hours_after}`;
  const overloadDelta = `${summary.overloaded_hours_before} -> ${summary.overloaded_hours_after}`;
  const stationKpiValue = selectedZone === ALL_ZONES
    ? summary.top_priority_station_zone
    : selectedStation?.zone_name || selectedZone;
  const stationKpiNote = selectedZone === ALL_ZONES
    ? `${topStation?.recommended_station_type || "Top priority"} / ${topStation?.recommended_chargers ?? summary.total_recommended_chargers} chargers`
    : `${selectedStation?.recommended_station_type || "Monitor Only"} / ${selectedStation?.recommended_chargers ?? 0} chargers`;
  const chargerCoverageBars: BarDatum[] = (selectedZone === ALL_ZONES ? data.zones : data.zones.filter((zone) => zone.zone_name === selectedZone))
    .slice()
    .sort((a, b) => b.existing_chargers - a.existing_chargers)
    .slice(0, selectedZone === ALL_ZONES ? 10 : 1)
    .map((zone) => ({
      label: zone.zone_name,
      value: zone.existing_chargers,
      note: `${zone.zone_type} / ${formatNumber(zone.ev_count_estimate)} EVs`,
      color: riskColor(zone.max_risk_score)
    }));
  const stationPriorityBars: BarDatum[] = (selectedZone === ALL_ZONES ? data.stations.slice(0, 8) : scopedStations.slice(0, 1)).map((station) => ({
    label: station.zone_name,
    value: station.station_priority_score,
    note: `${station.recommended_station_type} / ${station.recommended_chargers} chargers`,
    color: station.station_priority_score >= 70 ? "#c9493d" : station.station_priority_score >= 55 ? "#d89119" : "#2e9d62"
  }));

  return (
    <main className="min-h-screen">
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="rounded-md bg-primary p-2 text-primary-foreground">
                <Zap className="h-5 w-5" />
              </span>
              <div>
                <h1 className="text-xl font-semibold">InfraWisely</h1>
                <p className="text-sm text-muted-foreground">BESCOM EV charging demand and infrastructure planning</p>
              </div>
            </div>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <Select value={selectedZone} onChange={(event) => setSelectedZone(event.target.value)} className="min-w-[220px]">
              {zoneOptions.map((zone) => (
                <option key={zone} value={zone}>
                  {zone}
                </option>
              ))}
            </Select>
            <Button variant="outline" onClick={() => void load()}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <KpiCard label="Coverage" value={zoneLabel} note={coverageNote} icon={MapPin} onClick={() => setShowChargerDetails(true)} />
          <KpiCard label="Model R2" value={formatNumber(summary.model_r2_score)} note={`${summary.model_version || "v2"} / MAE ${summary.model_mae} kW`} icon={BarChart3} />
          <KpiCard label="Peak Reduction" value={`${summary.peak_load_reduction_percent}%`} note={`Critical ${criticalDelta}`} icon={Activity} />
          <KpiCard label="Station Plan" value={stationKpiValue} note={stationKpiNote} icon={Building2} />
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle>Zone Grid Stress Map</CardTitle>
                  <CardDescription>{summary.highest_risk_zone} is the highest-risk zone in current demo outputs</CardDescription>
                </div>
                <Badge variant="muted">Overload {overloadDelta}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <ZoneMap zones={data.zones} selectedZone={selectedZone} onSelect={setSelectedZone} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Operator Queue</CardTitle>
              <CardDescription>{scopedAlerts.length} high-risk records in the selected view</CardDescription>
            </CardHeader>
            <CardContent>
              <StressTable
                rows={scopedAlerts}
                schedules={data.schedules}
                reviewedKeys={reviewedActionKeys}
                onOpenActionPlan={setActiveActionPlan}
              />
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle>Hourly EV Demand Prediction</CardTitle>
                  <CardDescription>{chartScopeLabel} EV demand forecast from live model output</CardDescription>
                </div>
                <Badge variant="muted">Peak {formatHour(peakDemand.hour)} / {formatNumber(peakDemand.value, " kW")}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <LineChart
                unit="kW"
                series={[{ name: "Predicted EV load", color: "#0f8b8d", points: demandPoints }]}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle>Before vs After Smart Charging</CardTitle>
                  <CardDescription>{chartScopeLabel} load curve from optimizer output</CardDescription>
                </div>
                <Badge variant="success">Peak shift {formatNumber(peakShiftKw, " kW")}</Badge>
              </div>
            </CardHeader>
            <CardContent>
              <LineChart
                unit="kW"
                series={[
                  { name: "Before", color: "#c9493d", points: loadBefore },
                  { name: "After", color: "#2e9d62", points: loadAfter }
                ]}
              />
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle>Hourly Grid Risk Ratio</CardTitle>
                  <CardDescription>{chartScopeLabel} average load-to-capacity ratio from model output</CardDescription>
                </div>
                <Badge variant={riskVariant(peakRiskRatio.value >= 1 ? "Critical" : peakRiskRatio.value >= 0.9 ? "High" : "Medium")}>
                  Peak {formatHour(peakRiskRatio.hour)} / {formatNumber(peakRiskRatio.value)}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <LineChart
                unit=" ratio"
                series={[{ name: "Load ratio", color: "#4f46e5", points: riskRatioPoints }]}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Existing Charger Coverage</CardTitle>
              <CardDescription>{selectedZone === ALL_ZONES ? "Top mapped zones by BESCOM charger count" : "Selected-zone mapped BESCOM charger count"}</CardDescription>
            </CardHeader>
            <CardContent>
              <BarChart data={chargerCoverageBars} unit=" chargers" />
            </CardContent>
          </Card>

          <Card className="lg:col-span-3">
            <CardHeader>
              <CardTitle>Station Priority Ranking</CardTitle>
              <CardDescription>{selectedZone === ALL_ZONES ? "Top zones from dynamic station recommender" : "Selected-zone station recommender score"}</CardDescription>
            </CardHeader>
            <CardContent>
              <BarChart data={stationPriorityBars} unit="" />
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1fr_360px]">
          <Card>
            <CardHeader>
              <CardTitle>Smart Charging Schedule</CardTitle>
              <CardDescription>{scopedSchedules.length} recommendations in the selected view</CardDescription>
            </CardHeader>
            <CardContent>
              <ScheduleTable rows={scopedSchedules} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Live Model Probe</CardTitle>
              <CardDescription>{selectedDemandRow ? `${selectedDemandRow.zone_name}, ${formatHour(selectedDemandRow.hour)}` : "No demand row selected"}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-md border border-border p-3">
                  <p className="text-muted-foreground">Base grid</p>
                  <p className="mt-1 font-semibold">{formatNumber(selectedDemandRow?.base_grid_load_kw, " kW")}</p>
                </div>
                <div className="rounded-md border border-border p-3">
                  <p className="text-muted-foreground">CSV forecast</p>
                  <p className="mt-1 font-semibold">{formatNumber(selectedDemandRow ? evLoad(selectedDemandRow) : undefined, " kW")}</p>
                </div>
                <div className="rounded-md border border-border p-3">
                  <p className="text-muted-foreground">Total load</p>
                  <p className="mt-1 font-semibold">{formatNumber(selectedDemandRow ? totalLoad(selectedDemandRow) : undefined, " kW")}</p>
                </div>
                <div className="rounded-md border border-border p-3">
                  <p className="text-muted-foreground">API prediction</p>
                  <p className="mt-1 font-semibold">{formatNumber(prediction, " kW")}</p>
                </div>
              </div>
              {predictionError ? (
                <p className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{predictionError}</p>
              ) : null}
              <Button className="w-full" onClick={() => void runPrediction()} disabled={predicting || !selectedDemandRow}>
                {predicting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
                Predict Demand
              </Button>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1fr_0.8fr]">
          <Card>
            <CardHeader>
              <CardTitle>Station Location Planning</CardTitle>
              <CardDescription>{summary.total_recommended_chargers} chargers recommended across priority zones</CardDescription>
            </CardHeader>
            <CardContent>
              <StationTable rows={selectedZone === ALL_ZONES ? data.stations : scopedStations} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model Explainability</CardTitle>
              <CardDescription>{data.explainability.model_name}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div className="rounded-md bg-muted p-3">
                  <p className="text-muted-foreground">R2</p>
                  <p className="mt-1 font-semibold">{formatNumber(data.explainability.model_r2_score)}</p>
                </div>
                <div className="rounded-md bg-muted p-3">
                  <p className="text-muted-foreground">MAE</p>
                  <p className="mt-1 font-semibold">{formatNumber(data.explainability.mae, " kW")}</p>
                </div>
                <div className="rounded-md bg-muted p-3">
                  <p className="text-muted-foreground">RMSE</p>
                  <p className="mt-1 font-semibold">{formatNumber(data.explainability.rmse, " kW")}</p>
                </div>
              </div>
              <div className="space-y-3 text-sm leading-6">
                <p>{data.explainability.recommendation_logic.grid_stress}</p>
                <p>{data.explainability.recommendation_logic.scheduling}</p>
                <p>{data.explainability.recommendation_logic.station_planning}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {data.explainability.input_features.slice(0, 8).map((feature) => (
                  <Badge key={feature} variant="muted">
                    {feature}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </section>

        <footer className="pb-4 text-xs text-muted-foreground">
          API base: {process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000"}
        </footer>
      </div>
      {showChargerDetails ? (
        <ChargerDetailsModal
          zoneLabel={selectedZone === ALL_ZONES ? "Mapped dashboard zones" : selectedZone}
          rows={scopedChargers}
          onClose={() => setShowChargerDetails(false)}
        />
      ) : null}

      {activeActionPlan ? (
        <ActionPlanModal
          plan={activeActionPlan}
          reviewed={reviewedActionKeys.has(actionKey(activeActionPlan.alert))}
          onClose={() => setActiveActionPlan(null)}
          onMarkReviewed={() => {
            const key = actionKey(activeActionPlan.alert);
            setReviewedActionKeys((current) => new Set(current).add(key));
            setActiveActionPlan(null);
          }}
        />
      ) : null}
    </main>
  );
}
