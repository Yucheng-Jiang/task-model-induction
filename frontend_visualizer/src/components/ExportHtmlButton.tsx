"use client";

import { useState } from "react";
import { Download } from "lucide-react";

export default function ExportHtmlButton({
  dir,
  activeThreadId,
}: {
  dir: string;
  activeThreadId: string;
}) {
  const [pending, setPending] = useState(false);

  async function exportHtml() {
    setPending(true);
    try {
      const response = await fetch("/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dir, activeThreadId }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "task-results.html";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      window.alert(`Could not prepare the HTML report: ${message}`);
    } finally {
      setPending(false);
    }
  }

  return (
    <button
      type="button"
      onClick={exportHtml}
      disabled={pending}
      title="Download a self-contained HTML explorer with every task tab"
      className="inline-flex h-9 shrink-0 items-center gap-2 rounded-lg bg-stone-900 px-3 text-xs font-medium text-white shadow-sm transition-colors hover:bg-stone-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-stone-500 focus-visible:ring-offset-2 disabled:cursor-wait disabled:opacity-60"
    >
      <Download className="h-3.5 w-3.5" />
      {pending ? "Preparing…" : "Share HTML"}
    </button>
  );
}
