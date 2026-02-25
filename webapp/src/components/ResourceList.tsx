"use client";

import { Resource } from "@/lib/types";

const TYPE_ICONS: Record<Resource["type"], string> = {
  text: "T",
  pdf: "P",
  video: "V",
  audio: "A",
  link: "L",
};

const TYPE_COLORS: Record<Resource["type"], string> = {
  text: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  pdf: "bg-red-500/20 text-red-400 border-red-500/30",
  video: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  audio: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  link: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
};

const TYPE_LABELS: Record<Resource["type"], string> = {
  text: "Text",
  pdf: "PDF",
  video: "Video",
  audio: "Audio",
  link: "Link",
};

export default function ResourceList({ resources }: { resources: Resource[] }) {
  if (resources.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-lg font-semibold text-white">Resources</h3>
      {resources.map((r) => (
        <a
          key={r.id}
          href={r.url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 p-3 rounded-lg border border-zinc-700 hover:border-zinc-500 transition-colors group"
        >
          <span
            className={`w-8 h-8 flex items-center justify-center rounded text-xs font-bold border ${TYPE_COLORS[r.type]}`}
          >
            {TYPE_ICONS[r.type]}
          </span>
          <div className="flex-1 min-w-0">
            <p className="text-zinc-200 text-sm font-medium truncate group-hover:text-white transition-colors">
              {r.title}
            </p>
            <p className="text-zinc-500 text-xs">{TYPE_LABELS[r.type]}</p>
          </div>
          <svg
            className="w-4 h-4 text-zinc-500 group-hover:text-zinc-300 transition-colors flex-shrink-0"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
            />
          </svg>
        </a>
      ))}
    </div>
  );
}
