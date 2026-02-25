"use client";

import { useProgress } from "@/lib/hooks";
import { COURSE } from "@/lib/course-data";
import ExportImport from "@/components/ExportImport";
import Link from "next/link";

export default function Home() {
  const {
    progress,
    loading,
    enrolled,
    enroll,
    completedModules,
    totalModules,
  } = useProgress();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse text-zinc-500">Loading...</div>
      </div>
    );
  }

  const pct = totalModules > 0 ? (completedModules / totalModules) * 100 : 0;

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-sm font-bold">
              1b
            </div>
            <span className="font-semibold text-zinc-200">
              1-Bit LLM Course
            </span>
          </Link>
          {enrolled && (
            <ExportImport onImport={() => window.location.reload()} />
          )}
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-12">
        {/* Hero */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold mb-4">{COURSE.title}</h1>
          <p className="text-zinc-400 text-lg max-w-2xl">
            {COURSE.description}
          </p>

          {!enrolled ? (
            <button
              onClick={enroll}
              className="mt-6 px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium text-lg transition-colors cursor-pointer"
            >
              Enroll in Course
            </button>
          ) : (
            <div className="mt-6 space-y-2">
              <div className="flex items-center gap-3">
                <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden max-w-xs">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-sm text-zinc-400">
                  {completedModules}/{totalModules} modules completed
                </span>
              </div>
              <p className="text-sm text-zinc-500">
                Enrolled{" "}
                {new Date(progress!.enrolledAt).toLocaleDateString()}
              </p>
            </div>
          )}
        </div>

        {/* Module Grid */}
        <div className="space-y-3">
          {COURSE.modules.map((mod) => {
            const mp = progress?.moduleProgress[mod.id];
            const isCompleted = mp?.completed ?? false;
            const isStarted = mp?.started ?? false;
            const exerciseCount = mod.exercises.length;
            const exerciseDone = mp
              ? Object.values(mp.exerciseResults).filter((r) => r.correct)
                  .length
              : 0;

            return (
              <Link
                key={mod.id}
                href={enrolled ? `/course/${mod.id}` : "#"}
                onClick={(e) => !enrolled && e.preventDefault()}
                className={`block p-5 rounded-xl border transition-all ${
                  enrolled
                    ? "hover:border-zinc-600 hover:bg-zinc-900/50 cursor-pointer"
                    : "opacity-50 cursor-not-allowed"
                } ${
                  isCompleted
                    ? "border-emerald-500/30 bg-emerald-500/5"
                    : "border-zinc-800"
                }`}
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold flex-shrink-0 ${
                      isCompleted
                        ? "bg-emerald-600 text-white"
                        : isStarted
                          ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                          : "bg-zinc-800 text-zinc-400"
                    }`}
                  >
                    {isCompleted ? (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      mod.number
                    )}
                  </div>

                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-zinc-100">
                      Module {mod.number}: {mod.title}
                    </h3>
                    <p className="text-sm text-zinc-400 mt-1">
                      {mod.description}
                    </p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-zinc-500">
                      <span>
                        {exerciseDone}/{exerciseCount} exercises
                      </span>
                      <span>{mod.resources.length} resources</span>
                      {isCompleted && mp?.completedAt && (
                        <span className="text-emerald-500">
                          Completed{" "}
                          {new Date(mp.completedAt).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>

                  {enrolled && (
                    <svg
                      className="w-5 h-5 text-zinc-600 flex-shrink-0 mt-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  )}
                </div>
              </Link>
            );
          })}
        </div>

        {/* Key Papers */}
        {enrolled && (
          <div className="mt-16 pt-8 border-t border-zinc-800">
            <h2 className="text-xl font-bold mb-4">Key Papers</h2>
            <div className="grid gap-3 sm:grid-cols-2">
              {[
                {
                  title: "The Era of 1-bit LLMs (BitNet b1.58)",
                  url: "https://arxiv.org/abs/2402.17764",
                  year: 2024,
                },
                {
                  title: "BitNet: Scaling 1-bit Transformers",
                  url: "https://arxiv.org/abs/2310.11453",
                  year: 2023,
                },
                {
                  title: "BitNet b1.58 2B4T Technical Report",
                  url: "https://arxiv.org/abs/2504.12285",
                  year: 2025,
                },
                {
                  title: "HuggingFace: Fine-tuning to 1.58-bit",
                  url: "https://huggingface.co/blog/1_58_llm_extreme_quantization",
                  year: 2024,
                },
              ].map((paper) => (
                <a
                  key={paper.url}
                  href={paper.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-3 rounded-lg border border-zinc-800 hover:border-zinc-600 transition-colors"
                >
                  <p className="text-sm text-zinc-200">{paper.title}</p>
                  <p className="text-xs text-zinc-500 mt-1">{paper.year}</p>
                </a>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
