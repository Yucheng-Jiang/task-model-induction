import Explorer from "@/components/Explorer";
import DirInput from "@/components/DirInput";
import { DEFAULT_SESSION_DIR, loadSession } from "@/lib/session";

type SearchParams = Promise<{ [key: string]: string | string[] | undefined }>;

export default async function Page({ searchParams }: { searchParams: SearchParams }) {
  const sp = await searchParams;
  const rawDir = sp.dir;
  const dir =
    typeof rawDir === "string" && rawDir.trim().length > 0 ? rawDir : DEFAULT_SESSION_DIR;

  const result = await loadSession(dir);

  if (!result.ok) {
    return (
      <main className="min-h-screen bg-[var(--background)] text-stone-900">
        <div className="max-w-3xl mx-auto px-6 py-16">
          <h1 className="text-2xl font-semibold tracking-tight">Task Trace Explorer</h1>
          <p className="text-sm text-stone-500 mt-1">
            Couldn’t load this session. Point the explorer to a different directory.
          </p>
          <div className="mt-8 rounded-2xl border border-stone-200 bg-white p-6 shadow-sm">
            <DirInput initialDir={dir} />
            <p className="mt-6 text-xs text-stone-500 font-mono leading-relaxed break-all">
              Resolved path: {result.resolvedDir}
            </p>
            <p className="mt-2 text-sm text-rose-700 leading-relaxed">{result.error}</p>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-[var(--background)] text-stone-900">
      <Explorer data={result.data} />
    </main>
  );
}
