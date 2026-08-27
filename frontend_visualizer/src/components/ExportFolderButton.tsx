"use client";

import { useState } from "react";

type ExportUiState = {
  activeThreadId: string;
  filterTrace: boolean;
  viewMode: "pre-reconciliation" | "unified";
  selectedObjectiveId: string | null;
  selectedObjModelNodeId: string | null;
  selectedProcNodeId: string | null;
};

type PortableFile = {
  path: string;
  content: string;
};

type DirectoryPickerWindow = Window & {
  showDirectoryPicker?: (options?: { mode?: "read" | "readwrite" }) => Promise<FileSystemDirectoryHandle>;
};

function collectStyles(): string {
  const chunks: string[] = [];

  for (const sheet of Array.from(document.styleSheets)) {
    try {
      const rules = Array.from(sheet.cssRules);
      chunks.push(rules.map((rule) => rule.cssText).join("\n"));
    } catch {
      // Ignore inaccessible stylesheets.
    }
  }

  return chunks.join("\n");
}

async function writeFileRecursive(
  root: FileSystemDirectoryHandle,
  relativePath: string,
  content: string,
) {
  const parts = relativePath.split("/").filter(Boolean);
  const fileName = parts.pop();
  if (!fileName) return;

  let dir = root;
  for (const part of parts) {
    dir = await dir.getDirectoryHandle(part, { create: true });
  }

  const file = await dir.getFileHandle(fileName, { create: true });
  const writable = await file.createWritable();
  await writable.write(content);
  await writable.close();
}

export default function ExportFolderButton({
  dir,
  exportState,
}: {
  dir: string;
  exportState: ExportUiState;
}) {
  const [pending, setPending] = useState(false);

  async function exportFolder() {
    const pickerWindow = window as DirectoryPickerWindow;

    if (!pickerWindow.showDirectoryPicker) {
      window.alert("This browser does not support folder export. Use Chromium.");
      return;
    }

    setPending(true);
    try {
      const response = await fetch("/export-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dir, exportState }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const payload = (await response.json()) as { files: PortableFile[] };
      const root = await pickerWindow.showDirectoryPicker({ mode: "readwrite" });

      for (const file of payload.files) {
        await writeFileRecursive(root, file.path, file.content);
      }

      await writeFileRecursive(root, "app.css", collectStyles());
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      window.alert(`Folder export failed: ${message}`);
    } finally {
      setPending(false);
    }
  }

  return (
    <button
      type="button"
      onClick={exportFolder}
      disabled={pending}
      className="shrink-0 inline-flex items-center rounded-lg border border-stone-200 bg-white px-3 py-2 text-[12px] font-medium text-stone-700 transition-colors hover:bg-stone-50 disabled:opacity-60"
    >
      {pending ? "Exporting..." : "Export Folder"}
    </button>
  );
}
