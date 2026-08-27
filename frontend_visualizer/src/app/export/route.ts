import { exportedHtmlFilename, renderStandaloneReport } from "@/lib/exportReport";
import { renderShareExplorer } from "@/lib/shareExplorerReport";
import { loadSession } from "@/lib/session";

function htmlResponse(html: string, filename: string) {
  return new Response(html, {
    status: 200,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Content-Disposition": `attachment; filename="${filename}"`,
      "Cache-Control": "no-store",
    },
  });
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const dir = searchParams.get("dir");

  if (!dir?.trim()) {
    return new Response("Missing required `dir` query parameter.", { status: 400 });
  }

  const result = await loadSession(dir);
  if (!result.ok) {
    return new Response(result.error, { status: 400 });
  }

  return htmlResponse(
    renderStandaloneReport(result.data),
    exportedHtmlFilename(result.data),
  );
}

export async function POST(request: Request) {
  let body: { dir?: string; activeThreadId?: string };
  try {
    const parsed: unknown = await request.json();
    if (typeof parsed !== "object" || parsed === null) {
      return new Response("Request body must be a JSON object.", { status: 400 });
    }
    body = parsed as { dir?: string; activeThreadId?: string };
  } catch {
    return new Response("Request body must be valid JSON.", { status: 400 });
  }

  if (typeof body.dir !== "string" || !body.dir.trim()) {
    return new Response("Missing required `dir`.", { status: 400 });
  }
  if (body.activeThreadId !== undefined && typeof body.activeThreadId !== "string") {
    return new Response("`activeThreadId` must be a string when provided.", { status: 400 });
  }

  const result = await loadSession(body.dir);
  if (!result.ok) {
    return new Response(result.error, { status: 400 });
  }

  const initialThreadId = body.activeThreadId?.trim() || undefined;
  if (
    initialThreadId &&
    !result.data.threads.some((candidate) => candidate.id === initialThreadId)
  ) {
    return new Response(`Unknown task thread: ${initialThreadId}`, { status: 400 });
  }

  return htmlResponse(
    renderShareExplorer(result.data, initialThreadId),
    "task-results.html",
  );
}
