"use client";

import { createRoot } from "react-dom/client";
import Explorer from "@/components/Explorer";
// @ts-ignore Generated into the export folder at runtime.
import data from "./session-data.js";

const rootEl = document.getElementById("app");

if (!rootEl) {
  throw new Error("Portable export could not find #app.");
}

createRoot(rootEl).render(<Explorer data={data} />);
