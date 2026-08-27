"use client";

import { cn } from "@/lib/utils";

type Props = {
  total: number;
  positions: number[];
  highlightPositions?: number[];
  colorClass?: string;
  highlightClass?: string;
  height?: number;
};

export default function DensityBar({
  total,
  positions,
  highlightPositions,
  colorClass = "bg-stone-400",
  highlightClass = "bg-amber-500",
  height = 6,
}: Props) {
  if (total === 0) return null;
  const hi = new Set(highlightPositions ?? []);
  const sorted = [...positions].sort((a, b) => a - b);

  return (
    <div
      className="relative w-full rounded-full bg-stone-100 overflow-hidden"
      style={{ height }}
      aria-label={`Trace coverage: ${positions.length} of ${total}`}
    >
      {sorted.map((p, i) => {
        const left = (p / total) * 100;
        const w = Math.max(0.4, (1 / total) * 100);
        const isHi = hi.has(p);
        return (
          <span
            key={`${p}-${i}`}
            className={cn("absolute top-0 bottom-0", isHi ? highlightClass : colorClass)}
            style={{
              left: `${left}%`,
              width: `${w}%`,
              opacity: isHi ? 1 : 0.7,
            }}
          />
        );
      })}
    </div>
  );
}
