export const THREAD_PALETTE: Record<string, { dot: string; text: string; soft: string; ring: string; bar: string }> = {
  C1: { dot: "bg-indigo-500", text: "text-indigo-700", soft: "bg-indigo-50", ring: "ring-indigo-200", bar: "bg-indigo-400" },
  C2: { dot: "bg-emerald-500", text: "text-emerald-700", soft: "bg-emerald-50", ring: "ring-emerald-200", bar: "bg-emerald-400" },
  C3: { dot: "bg-rose-500", text: "text-rose-700", soft: "bg-rose-50", ring: "ring-rose-200", bar: "bg-rose-400" },
  C4: { dot: "bg-amber-500", text: "text-amber-700", soft: "bg-amber-50", ring: "ring-amber-200", bar: "bg-amber-400" },
  C5: { dot: "bg-sky-500", text: "text-sky-700", soft: "bg-sky-50", ring: "ring-sky-200", bar: "bg-sky-400" },
  C6: { dot: "bg-fuchsia-500", text: "text-fuchsia-700", soft: "bg-fuchsia-50", ring: "ring-fuchsia-200", bar: "bg-fuchsia-400" },
  C7: { dot: "bg-violet-500", text: "text-violet-700", soft: "bg-violet-50", ring: "ring-violet-200", bar: "bg-violet-400" },
  C8: { dot: "bg-teal-500", text: "text-teal-700", soft: "bg-teal-50", ring: "ring-teal-200", bar: "bg-teal-400" },
  C9: { dot: "bg-orange-500", text: "text-orange-700", soft: "bg-orange-50", ring: "ring-orange-200", bar: "bg-orange-400" },
};

export function threadColor(id: string) {
  return (
    THREAD_PALETTE[id] ?? {
      dot: "bg-stone-400",
      text: "text-stone-700",
      soft: "bg-stone-50",
      ring: "ring-stone-200",
      bar: "bg-stone-400",
    }
  );
}
