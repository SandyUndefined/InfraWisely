import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(value: number | null | undefined, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const formatted = Number.isInteger(value)
    ? value.toLocaleString("en-IN")
    : value.toLocaleString("en-IN", { maximumFractionDigits: 2 });
  return `${formatted}${suffix}`;
}

export function formatHour(hour: number | string) {
  const value = Number(hour);
  if (!Number.isFinite(value)) {
    return "-";
  }
  const normalizedHour = ((Math.trunc(value) % 24) + 24) % 24;
  const suffix = normalizedHour < 12 ? "AM" : "PM";
  const display = normalizedHour % 12 || 12;
  return `${display}:00 ${suffix}`;
}

export function formatTimeText(text: string | null | undefined) {
  if (!text) {
    return "";
  }

  return text.replace(/\b([01]?\d|2[0-3]):00\b/g, (match) => {
    const hour = Number.parseInt(match, 10);
    return formatHour(hour);
  });
}
