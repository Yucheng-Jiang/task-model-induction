const ACTIVITY_RE = /^activity_(\d+)$/;
const ACTIVITY_RANGE_RE = /^activity_(\d+)-activity_(\d+)$/;

export function padActivityId(n: number): string {
  return `activity_${String(n).padStart(4, "0")}`;
}

export function parseActivityNumber(id: string): number | null {
  const m = ACTIVITY_RE.exec(id);
  return m ? parseInt(m[1], 10) : null;
}

export function expandSegmentRef(ref: string): string[] {
  const range = ACTIVITY_RANGE_RE.exec(ref);
  if (range) {
    const start = parseInt(range[1], 10);
    const end = parseInt(range[2], 10);
    const out: string[] = [];
    for (let i = start; i <= end; i++) out.push(padActivityId(i));
    return out;
  }
  const single = ACTIVITY_RE.exec(ref);
  if (single) return [ref];
  return [];
}

export function expandSegmentRefs(refs: string[] | undefined | null): string[] {
  if (!refs) return [];
  const set = new Set<string>();
  for (const r of refs) {
    for (const id of expandSegmentRef(r)) set.add(id);
  }
  return Array.from(set);
}

export function summarizeRefs(refs: string[] | undefined | null): { count: number; ranges: string[] } {
  if (!refs || refs.length === 0) return { count: 0, ranges: [] };
  return {
    count: expandSegmentRefs(refs).length,
    ranges: refs,
  };
}
