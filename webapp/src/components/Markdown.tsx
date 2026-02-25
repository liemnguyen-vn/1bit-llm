"use client";

import React from "react";

function parseInlineCode(text: string): React.ReactNode[] {
  const parts = text.split(/(`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={i}
          className="bg-zinc-800 text-emerald-400 px-1.5 py-0.5 rounded text-sm font-mono"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    // bold
    const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
    return boldParts.map((bp, j) => {
      if (bp.startsWith("**") && bp.endsWith("**")) {
        return <strong key={`${i}-${j}`}>{bp.slice(2, -2)}</strong>;
      }
      return <React.Fragment key={`${i}-${j}`}>{bp}</React.Fragment>;
    });
  });
}

export default function Markdown({ content }: { content: string }) {
  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Code blocks
    if (line.trim().startsWith("```")) {
      const lang = line.trim().slice(3);
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      elements.push(
        <div key={elements.length} className="my-4">
          {lang && (
            <div className="bg-zinc-900 text-zinc-500 text-xs px-4 py-1 rounded-t border border-b-0 border-zinc-700 font-mono">
              {lang}
            </div>
          )}
          <pre
            className={`bg-zinc-900 text-zinc-200 p-4 overflow-x-auto text-sm font-mono border border-zinc-700 ${
              lang ? "rounded-b" : "rounded"
            }`}
          >
            <code>{codeLines.join("\n")}</code>
          </pre>
        </div>
      );
      continue;
    }

    // Headers
    if (line.startsWith("## ")) {
      elements.push(
        <h2
          key={elements.length}
          className="text-2xl font-bold mt-8 mb-4 text-white"
        >
          {line.slice(3)}
        </h2>
      );
      i++;
      continue;
    }
    if (line.startsWith("### ")) {
      elements.push(
        <h3
          key={elements.length}
          className="text-xl font-semibold mt-6 mb-3 text-zinc-100"
        >
          {line.slice(4)}
        </h3>
      );
      i++;
      continue;
    }

    // Tables
    if (line.includes("|") && line.trim().startsWith("|")) {
      const tableLines: string[] = [];
      while (
        i < lines.length &&
        lines[i].includes("|") &&
        lines[i].trim().startsWith("|")
      ) {
        tableLines.push(lines[i]);
        i++;
      }
      if (tableLines.length >= 2) {
        const header = tableLines[0]
          .split("|")
          .filter((c) => c.trim())
          .map((c) => c.trim());
        const rows = tableLines.slice(2).map((r) =>
          r
            .split("|")
            .filter((c) => c.trim())
            .map((c) => c.trim())
        );
        elements.push(
          <div key={elements.length} className="my-4 overflow-x-auto">
            <table className="w-full text-sm border border-zinc-700">
              <thead>
                <tr className="bg-zinc-800">
                  {header.map((h, hi) => (
                    <th
                      key={hi}
                      className="px-3 py-2 text-left font-semibold text-zinc-200 border-b border-zinc-700"
                    >
                      {parseInlineCode(h)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, ri) => (
                  <tr
                    key={ri}
                    className={ri % 2 === 0 ? "bg-zinc-900/50" : "bg-zinc-900"}
                  >
                    {row.map((cell, ci) => (
                      <td
                        key={ci}
                        className="px-3 py-2 border-b border-zinc-800 text-zinc-300"
                      >
                        {parseInlineCode(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }
    }

    // Bullet lists
    if (line.trim().startsWith("- ")) {
      const items: string[] = [];
      while (
        i < lines.length &&
        lines[i].trim().startsWith("- ")
      ) {
        items.push(lines[i].trim().slice(2));
        i++;
      }
      elements.push(
        <ul key={elements.length} className="my-3 space-y-1 pl-5 list-disc">
          {items.map((item, idx) => (
            <li key={idx} className="text-zinc-300">
              {parseInlineCode(item)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Empty lines
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Paragraphs
    elements.push(
      <p key={elements.length} className="my-3 text-zinc-300 leading-relaxed">
        {parseInlineCode(line)}
      </p>
    );
    i++;
  }

  return <div className="prose-custom">{elements}</div>;
}
