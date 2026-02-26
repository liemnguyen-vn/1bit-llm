"use client";

import { useEffect, useState } from "react";
import { useProgress } from "@/lib/hooks";
import { COURSE } from "@/lib/course-data";
import Markdown from "@/components/Markdown";
import ExerciseRunner from "@/components/ExerciseRunner";
import ResourceList from "@/components/ResourceList";
import ExportImport from "@/components/ExportImport";
import AnimationSection from "@/components/AnimationSection";
import Link from "next/link";

type Tab = "learn" | "exercises" | "resources" | "notes";

export default function ModulePageClient({
  moduleId,
}: {
  moduleId: string;
}) {
  const [tab, setTab] = useState<Tab>("learn");
  const {
    progress,
    loading,
    enrolled,
    markModuleStarted,
    markModuleCompleted,
    markModuleIncomplete,
    submitExercise,
    updateNotes,
  } = useProgress();

  const mod = COURSE.modules.find((m) => m.id === moduleId);
  const modIndex = COURSE.modules.findIndex((m) => m.id === moduleId);
  const prevMod = modIndex > 0 ? COURSE.modules[modIndex - 1] : null;
  const nextMod =
    modIndex < COURSE.modules.length - 1
      ? COURSE.modules[modIndex + 1]
      : null;

  const mp = progress?.moduleProgress[moduleId];
  const isCompleted = mp?.completed ?? false;

  useEffect(() => {
    if (enrolled && mod) {
      markModuleStarted(mod.id);
    }
  }, [enrolled, mod, markModuleStarted]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse text-zinc-500">Loading...</div>
      </div>
    );
  }

  if (!mod) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-zinc-500 mb-4">Module not found</p>
          <Link href="/" className="text-emerald-400 hover:underline">
            Back to course
          </Link>
        </div>
      </div>
    );
  }

  if (!enrolled) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-zinc-500 mb-4">Please enroll first</p>
          <Link href="/" className="text-emerald-400 hover:underline">
            Go to course page
          </Link>
        </div>
      </div>
    );
  }

  const tabs: { key: Tab; label: string; count?: number }[] = [
    { key: "learn", label: "Learn" },
    { key: "exercises", label: "Exercises", count: mod.exercises.length },
    { key: "resources", label: "Resources", count: mod.resources.length },
    { key: "notes", label: "Notes" },
  ];

  const exerciseDone = mp
    ? Object.values(mp.exerciseResults).filter((r) => r.correct).length
    : 0;

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
          <ExportImport onImport={() => window.location.reload()} />
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-zinc-500 mb-6">
          <Link href="/" className="hover:text-zinc-300 transition-colors">
            Course
          </Link>
          <span>/</span>
          <span className="text-zinc-300">
            Module {mod.number}: {mod.title}
          </span>
        </div>

        {/* Module Header */}
        <div className="mb-8">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold mb-2">
                Module {mod.number}: {mod.title}
              </h1>
              <p className="text-zinc-400">{mod.description}</p>
            </div>

            <button
              onClick={() =>
                isCompleted
                  ? markModuleIncomplete(mod.id)
                  : markModuleCompleted(mod.id)
              }
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex-shrink-0 cursor-pointer ${
                isCompleted
                  ? "bg-emerald-600 text-white hover:bg-emerald-700"
                  : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700 border border-zinc-700"
              }`}
            >
              {isCompleted ? "Completed" : "Mark Complete"}
            </button>
          </div>

          <div className="flex items-center gap-3 mt-4">
            <span className="text-xs text-zinc-500">
              {exerciseDone}/{mod.exercises.length} exercises correct
            </span>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 border-b border-zinc-800 mb-8">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`px-4 py-2.5 text-sm font-medium transition-colors relative cursor-pointer ${
                tab === t.key
                  ? "text-white"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t.label}
              {t.count !== undefined && (
                <span className="ml-1.5 text-xs text-zinc-600">
                  ({t.count})
                </span>
              )}
              {tab === t.key && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-500" />
              )}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {tab === "learn" && (
          <div className="max-w-3xl">
            <AnimationSection moduleId={moduleId} />
            <Markdown content={mod.content} />
          </div>
        )}

        {tab === "exercises" && (
          <div className="max-w-3xl space-y-6">
            {mod.exercises.length === 0 ? (
              <p className="text-zinc-500">No exercises for this module.</p>
            ) : (
              mod.exercises.map((ex) => (
                <ExerciseRunner
                  key={ex.id}
                  exercise={ex}
                  previousResult={mp?.exerciseResults[ex.id]}
                  onSubmit={(answer, correct) =>
                    submitExercise(mod.id, ex.id, answer, correct)
                  }
                />
              ))
            )}
          </div>
        )}

        {tab === "resources" && (
          <div className="max-w-3xl">
            <ResourceList resources={mod.resources} />
            {mod.resources.length === 0 && (
              <p className="text-zinc-500">No resources for this module.</p>
            )}
          </div>
        )}

        {tab === "notes" && (
          <div className="max-w-3xl">
            <p className="text-sm text-zinc-500 mb-3">
              Your personal notes for this module. Saved automatically to
              IndexedDB.
            </p>
            <textarea
              className="w-full h-64 bg-zinc-900 border border-zinc-700 rounded-lg p-4 text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-emerald-500 resize-y font-mono text-sm"
              placeholder="Write your notes here..."
              value={mp?.notes ?? ""}
              onChange={(e) => updateNotes(mod.id, e.target.value)}
            />
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-between mt-12 pt-8 border-t border-zinc-800">
          {prevMod ? (
            <Link
              href={`/course/${prevMod.id}`}
              className="flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 19l-7-7 7-7"
                />
              </svg>
              Module {prevMod.number}: {prevMod.title}
            </Link>
          ) : (
            <div />
          )}
          {nextMod ? (
            <Link
              href={`/course/${nextMod.id}`}
              className="flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              Module {nextMod.number}: {nextMod.title}
              <svg
                className="w-4 h-4"
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
            </Link>
          ) : (
            <Link
              href="/"
              className="flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
            >
              Back to Course Overview
            </Link>
          )}
        </div>
      </main>
    </div>
  );
}
