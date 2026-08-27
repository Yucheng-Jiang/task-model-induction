import { promises as fs } from "fs";
import path from "path";
import ts from "typescript";
import { loadSession } from "@/lib/session";

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

const PROJECT_ROOT = process.cwd();
const SRC_ROOT = path.join(PROJECT_ROOT, "src");
const ENTRY_FILE = path.join(SRC_ROOT, "portable", "entry.tsx");
const IMPORT_RE =
  /((?:import|export)\s+(?:type\s+)?(?:[^"'()]*?\s+from\s+)?|import\s*\()\s*["']([^"']+)["'](\s*\)?)/g;

function toPosix(value: string): string {
  return value.replaceAll(path.sep, "/");
}

async function fileExists(file: string) {
  try {
    await fs.access(file);
    return true;
  } catch {
    return false;
  }
}

async function resolveLocalImport(specifier: string, importer: string): Promise<string | null> {
  const base = specifier.startsWith("@/")
    ? path.join(SRC_ROOT, specifier.slice(2))
    : specifier.startsWith(".")
    ? path.resolve(path.dirname(importer), specifier)
    : null;

  if (!base) return null;

  for (const ext of [".ts", ".tsx", ".js"]) {
    if (await fileExists(base + ext)) return base + ext;
  }
  for (const suffix of ["/index.ts", "/index.tsx", "/index.js"]) {
    if (await fileExists(base + suffix)) return base + suffix;
  }
  return null;
}

function emittedPathForSource(sourceFile: string): string {
  return toPosix(path.relative(SRC_ROOT, sourceFile)).replace(/\.(ts|tsx)$/, ".js");
}

function browserSpecifier(fromSource: string, toSource: string): string {
  const fromOut = emittedPathForSource(fromSource);
  const toOut = emittedPathForSource(toSource);
  let relative = path.relative(path.dirname(fromOut), toOut);
  if (!relative.startsWith(".")) relative = `./${relative}`;
  return toPosix(relative);
}

async function transpilePortableModules(entryFile: string): Promise<PortableFile[]> {
  const queue = [entryFile];
  const seen = new Set<string>();
  const files: PortableFile[] = [];

  while (queue.length > 0) {
    const file = queue.pop();
    if (!file || seen.has(file)) continue;
    seen.add(file);

    const source = await fs.readFile(file, "utf8");
    const rewrites = new Map<string, string>();
    let match: RegExpExecArray | null;

    while ((match = IMPORT_RE.exec(source)) !== null) {
      const specifier = match[2];
      const resolved = await resolveLocalImport(specifier, file);
      if (resolved) {
        rewrites.set(specifier, browserSpecifier(file, resolved));
        queue.push(resolved);
      }
    }
    IMPORT_RE.lastIndex = 0;

    const transpiled = ts.transpileModule(source, {
      compilerOptions: {
        target: ts.ScriptTarget.ES2020,
        module: ts.ModuleKind.ES2020,
        jsx: ts.JsxEmit.ReactJSX,
      },
      fileName: file,
    }).outputText;

    const rewritten = transpiled.replace(IMPORT_RE, (full, prefix, specifier, suffix) => {
      return `${prefix}"${rewrites.get(specifier) ?? specifier}"${suffix}`;
    });

    files.push({
      path: emittedPathForSource(file),
      content: rewritten,
    });
  }

  return files;
}

function buildIndexHtml(exportState: ExportUiState): string {
  return `<!doctype html>
<html lang="en" class="h-full antialiased">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Task Trace Explorer</title>
    <link rel="stylesheet" href="./app.css" />
    <script type="importmap">
      {
        "imports": {
          "react": "https://esm.sh/react@19.2.4",
          "react/jsx-runtime": "https://esm.sh/react@19.2.4/jsx-runtime",
          "react-dom/client": "https://esm.sh/react-dom@19.2.4/client",
          "lucide-react": "https://esm.sh/lucide-react@1.16.0",
          "clsx": "https://esm.sh/clsx@2.1.1",
          "tailwind-merge": "https://esm.sh/tailwind-merge@3.6.0",
          "next/navigation": "./portable/shims/next-navigation.js"
        }
      }
    </script>
  </head>
  <body class="min-h-full">
    <div id="app"></div>
    <script>window.__EXPLORER_EXPORT_STATE__ = ${JSON.stringify(exportState)};</script>
    <script type="module" src="./portable/entry.js"></script>
  </body>
</html>`;
}

export async function POST(request: Request) {
  const body = (await request.json()) as {
    dir?: string;
    exportState?: ExportUiState;
  };

  if (!body.dir?.trim()) {
    return new Response("Missing required `dir`.", { status: 400 });
  }
  if (!body.exportState) {
    return new Response("Missing required `exportState`.", { status: 400 });
  }

  const session = await loadSession(body.dir);
  if (!session.ok) {
    return new Response(session.error, { status: 400 });
  }

  const files = await transpilePortableModules(ENTRY_FILE);
  files.push({
    path: "portable/session-data.js",
    content: `export default ${JSON.stringify(session.data, null, 2)};`,
  });
  files.push({
    path: "portable/shims/next-navigation.js",
    content: `export function useRouter() {
  return {
    push(href) {
      if (typeof window !== "undefined") {
        window.location.href = href;
      }
    }
  };
}
`,
  });
  files.push({
    path: "index.html",
    content: buildIndexHtml(body.exportState),
  });

  return Response.json({ files });
}
