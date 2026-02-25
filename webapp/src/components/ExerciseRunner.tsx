"use client";

import { Exercise } from "@/lib/types";
import { useState } from "react";

interface Props {
  exercise: Exercise;
  previousResult?: { answer: string; correct: boolean };
  onSubmit: (answer: string, correct: boolean) => void;
}

export default function ExerciseRunner({
  exercise,
  previousResult,
  onSubmit,
}: Props) {
  const [selected, setSelected] = useState<string>(
    previousResult?.answer ?? ""
  );
  const [submitted, setSubmitted] = useState(!!previousResult);
  const [isCorrect, setIsCorrect] = useState(previousResult?.correct ?? false);

  const handleSubmit = () => {
    if (!selected) return;
    const correct =
      selected.trim().toLowerCase() ===
      exercise.correctAnswer.trim().toLowerCase();
    setIsCorrect(correct);
    setSubmitted(true);
    onSubmit(selected, correct);
  };

  const handleReset = () => {
    setSelected("");
    setSubmitted(false);
    setIsCorrect(false);
  };

  return (
    <div className="border border-zinc-700 rounded-lg overflow-hidden">
      <div className="bg-zinc-800 px-4 py-3 border-b border-zinc-700">
        <h4 className="font-semibold text-white">{exercise.title}</h4>
        <p className="text-sm text-zinc-400">{exercise.description}</p>
      </div>

      <div className="p-4 space-y-4">
        <p className="text-zinc-200">{exercise.question}</p>

        {exercise.type === "multiple-choice" && exercise.options && (
          <div className="space-y-2">
            {exercise.options.map((option) => (
              <label
                key={option}
                className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                  submitted && option === exercise.correctAnswer
                    ? "border-emerald-500 bg-emerald-500/10"
                    : submitted &&
                        selected === option &&
                        option !== exercise.correctAnswer
                      ? "border-red-500 bg-red-500/10"
                      : selected === option
                        ? "border-blue-500 bg-blue-500/10"
                        : "border-zinc-700 hover:border-zinc-500"
                }`}
              >
                <input
                  type="radio"
                  name={exercise.id}
                  value={option}
                  checked={selected === option}
                  onChange={() => !submitted && setSelected(option)}
                  disabled={submitted}
                  className="mt-1 accent-blue-500"
                />
                <span className="text-zinc-300 text-sm">{option}</span>
              </label>
            ))}
          </div>
        )}

        {exercise.type === "short-answer" && (
          <input
            type="text"
            value={selected}
            onChange={(e) => !submitted && setSelected(e.target.value)}
            disabled={submitted}
            placeholder="Type your answer..."
            className={`w-full px-4 py-2 bg-zinc-900 border rounded-lg text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 ${
              submitted
                ? isCorrect
                  ? "border-emerald-500 focus:ring-emerald-500"
                  : "border-red-500 focus:ring-red-500"
                : "border-zinc-700 focus:ring-blue-500"
            }`}
          />
        )}

        <div className="flex items-center gap-3">
          {!submitted ? (
            <button
              onClick={handleSubmit}
              disabled={!selected}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-lg text-sm font-medium transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              Submit Answer
            </button>
          ) : (
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 text-zinc-200 rounded-lg text-sm font-medium transition-colors cursor-pointer"
            >
              Try Again
            </button>
          )}

          {submitted && (
            <span
              className={`text-sm font-medium ${
                isCorrect ? "text-emerald-400" : "text-red-400"
              }`}
            >
              {isCorrect ? "Correct!" : "Incorrect"}
            </span>
          )}
        </div>

        {submitted && (
          <div
            className={`p-3 rounded-lg text-sm ${
              isCorrect
                ? "bg-emerald-500/10 border border-emerald-500/30 text-emerald-300"
                : "bg-amber-500/10 border border-amber-500/30 text-amber-300"
            }`}
          >
            <strong>Explanation:</strong> {exercise.explanation}
          </div>
        )}
      </div>
    </div>
  );
}
