"use client";

import { exportAllData, importData } from "@/lib/db";
import { ExportData } from "@/lib/types";
import { useRef, useState } from "react";

export default function ExportImport({ onImport }: { onImport: () => void }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string | null>(null);

  const handleExport = async () => {
    try {
      const data = await exportAllData();
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `1bit-llm-progress-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      setStatus("Exported successfully!");
      setTimeout(() => setStatus(null), 3000);
    } catch {
      setStatus("Export failed");
    }
  };

  const handleImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data: ExportData = JSON.parse(text);
      await importData(data);
      setStatus("Imported successfully! Refreshing...");
      onImport();
      setTimeout(() => setStatus(null), 3000);
    } catch {
      setStatus("Import failed — invalid file format");
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={handleExport}
        className="px-3 py-1.5 text-sm bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 transition-colors cursor-pointer"
      >
        Export Data
      </button>
      <label className="px-3 py-1.5 text-sm bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 transition-colors cursor-pointer">
        Import Data
        <input
          ref={fileRef}
          type="file"
          accept=".json"
          className="hidden"
          onChange={handleImport}
        />
      </label>
      {status && (
        <span className="text-sm text-emerald-400">{status}</span>
      )}
    </div>
  );
}
