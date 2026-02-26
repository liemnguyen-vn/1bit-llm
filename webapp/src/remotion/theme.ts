export const COLORS = {
  bg: "#09090b",
  surface: "#18181b",
  surfaceLight: "#27272a",
  border: "#3f3f46",
  text: "#fafafa",
  textMuted: "#a1a1aa",
  textDim: "#71717a",
  accent: "#10b981",
  accentLight: "#34d399",
  accentDim: "#059669",
  blue: "#3b82f6",
  blueLight: "#60a5fa",
  red: "#ef4444",
  amber: "#f59e0b",
  purple: "#a855f7",
  orange: "#f97316",
  cyan: "#06b6d4",
  pink: "#ec4899",
} as const;

export const FONTS = {
  sans: "var(--font-geist-sans), system-ui, sans-serif",
  mono: "var(--font-geist-mono), monospace",
} as const;

export const DIMENSIONS = {
  width: 960,
  height: 540,
} as const;

export const FPS = 30;
