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

export function formatHour(hour: number) {
  const suffix = hour < 12 ? "AM" : "PM";
  const display = hour % 12 || 12;
  return `${display}:00 ${suffix}`;
}

