import * as React from "react";
import { cn } from "@/lib/utils";

export function Select({
  className,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "h-10 rounded-md border border-input bg-card px-3 text-sm outline-none transition-colors focus:ring-2 focus:ring-ring",
        className
      )}
      {...props}
    />
  );
}

