export interface Module {
  id: string;
  number: number;
  title: string;
  description: string;
  content: string; // markdown content
  exercises: Exercise[];
  resources: Resource[];
}

export interface Exercise {
  id: string;
  title: string;
  description: string;
  type: "multiple-choice" | "code" | "short-answer";
  question: string;
  options?: string[];
  correctAnswer: string;
  explanation: string;
}

export interface Resource {
  id: string;
  title: string;
  type: "text" | "pdf" | "video" | "audio" | "link";
  url: string;
}

export interface Course {
  id: string;
  title: string;
  description: string;
  modules: Module[];
}

export interface UserProgress {
  courseId: string;
  enrolledAt: string;
  moduleProgress: Record<
    string,
    {
      started: boolean;
      completed: boolean;
      completedAt?: string;
      exerciseResults: Record<
        string,
        { answer: string; correct: boolean; attemptedAt: string }
      >;
      notes: string;
    }
  >;
}

export interface ExportData {
  version: 1;
  exportedAt: string;
  progress: UserProgress[];
}
