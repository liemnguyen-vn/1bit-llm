"use client";

import { useState, useEffect, useCallback } from "react";
import { UserProgress } from "./types";
import { getProgress, saveProgress, createEmptyProgress } from "./db";
import { COURSE } from "./course-data";

export function useProgress() {
  const [progress, setProgress] = useState<UserProgress | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getProgress(COURSE.id).then((p) => {
      setProgress(p);
      setLoading(false);
    });
  }, []);

  const enroll = useCallback(async () => {
    const p = createEmptyProgress(COURSE.id);
    await saveProgress(p);
    setProgress(p);
  }, []);

  const update = useCallback(
    async (updater: (prev: UserProgress) => UserProgress) => {
      setProgress((prev) => {
        if (!prev) return prev;
        const next = updater(prev);
        saveProgress(next);
        return next;
      });
    },
    []
  );

  const markModuleStarted = useCallback(
    (moduleId: string) => {
      update((prev) => ({
        ...prev,
        moduleProgress: {
          ...prev.moduleProgress,
          [moduleId]: {
            ...prev.moduleProgress[moduleId],
            started: true,
            completed: prev.moduleProgress[moduleId]?.completed ?? false,
            exerciseResults:
              prev.moduleProgress[moduleId]?.exerciseResults ?? {},
            notes: prev.moduleProgress[moduleId]?.notes ?? "",
          },
        },
      }));
    },
    [update]
  );

  const markModuleCompleted = useCallback(
    (moduleId: string) => {
      update((prev) => ({
        ...prev,
        moduleProgress: {
          ...prev.moduleProgress,
          [moduleId]: {
            ...prev.moduleProgress[moduleId],
            started: true,
            completed: true,
            completedAt: new Date().toISOString(),
            exerciseResults:
              prev.moduleProgress[moduleId]?.exerciseResults ?? {},
            notes: prev.moduleProgress[moduleId]?.notes ?? "",
          },
        },
      }));
    },
    [update]
  );

  const markModuleIncomplete = useCallback(
    (moduleId: string) => {
      update((prev) => ({
        ...prev,
        moduleProgress: {
          ...prev.moduleProgress,
          [moduleId]: {
            ...prev.moduleProgress[moduleId],
            started: true,
            completed: false,
            completedAt: undefined,
            exerciseResults:
              prev.moduleProgress[moduleId]?.exerciseResults ?? {},
            notes: prev.moduleProgress[moduleId]?.notes ?? "",
          },
        },
      }));
    },
    [update]
  );

  const submitExercise = useCallback(
    (moduleId: string, exerciseId: string, answer: string, correct: boolean) => {
      update((prev) => ({
        ...prev,
        moduleProgress: {
          ...prev.moduleProgress,
          [moduleId]: {
            ...prev.moduleProgress[moduleId],
            started: true,
            completed: prev.moduleProgress[moduleId]?.completed ?? false,
            exerciseResults: {
              ...(prev.moduleProgress[moduleId]?.exerciseResults ?? {}),
              [exerciseId]: {
                answer,
                correct,
                attemptedAt: new Date().toISOString(),
              },
            },
            notes: prev.moduleProgress[moduleId]?.notes ?? "",
          },
        },
      }));
    },
    [update]
  );

  const updateNotes = useCallback(
    (moduleId: string, notes: string) => {
      update((prev) => ({
        ...prev,
        moduleProgress: {
          ...prev.moduleProgress,
          [moduleId]: {
            ...prev.moduleProgress[moduleId],
            started: prev.moduleProgress[moduleId]?.started ?? true,
            completed: prev.moduleProgress[moduleId]?.completed ?? false,
            exerciseResults:
              prev.moduleProgress[moduleId]?.exerciseResults ?? {},
            notes,
          },
        },
      }));
    },
    [update]
  );

  const enrolled = progress !== null;
  const completedModules = progress
    ? Object.values(progress.moduleProgress).filter((m) => m.completed).length
    : 0;
  const totalModules = COURSE.modules.length;

  return {
    progress,
    loading,
    enrolled,
    enroll,
    markModuleStarted,
    markModuleCompleted,
    markModuleIncomplete,
    submitExercise,
    updateNotes,
    completedModules,
    totalModules,
  };
}
