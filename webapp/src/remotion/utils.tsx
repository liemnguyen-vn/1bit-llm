import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { CSSProperties, ReactNode } from "react";
import { COLORS, FONTS } from "./theme";

export function FadeIn({
  children,
  delay = 0,
  duration = 15,
  style,
}: {
  children: ReactNode;
  delay?: number;
  duration?: number;
  style?: CSSProperties;
}) {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame - delay, [0, duration], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return <div style={{ opacity, ...style }}>{children}</div>;
}

export function SlideIn({
  children,
  delay = 0,
  from = "left",
  distance = 60,
  style,
}: {
  children: ReactNode;
  delay?: number;
  from?: "left" | "right" | "top" | "bottom";
  distance?: number;
  style?: CSSProperties;
}) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({ frame: frame - delay, fps, config: { damping: 15 } });
  const axis = from === "left" || from === "right" ? "X" : "Y";
  const sign = from === "left" || from === "top" ? -1 : 1;
  const translate = interpolate(progress, [0, 1], [sign * distance, 0]);
  const opacity = interpolate(progress, [0, 1], [0, 1]);
  return (
    <div style={{ transform: `translate${axis}(${translate}px)`, opacity, ...style }}>
      {children}
    </div>
  );
}

export function AnimatedBar({
  value,
  maxValue,
  label,
  color,
  delay = 0,
  width = 400,
  height = 36,
  showValue,
}: {
  value: number;
  maxValue: number;
  label: string;
  color: string;
  delay?: number;
  width?: number;
  height?: number;
  showValue?: string;
}) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({ frame: frame - delay, fps, config: { damping: 15, mass: 0.8 } });
  const barWidth = interpolate(progress, [0, 1], [0, (value / maxValue) * width]);
  const opacity = interpolate(frame - delay, [0, 10], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ opacity, display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
      <div
        style={{
          width: 120,
          textAlign: "right",
          fontSize: 14,
          color: COLORS.textMuted,
          fontFamily: FONTS.sans,
          flexShrink: 0,
        }}
      >
        {label}
      </div>
      <div
        style={{
          width,
          height,
          background: COLORS.surface,
          borderRadius: 6,
          overflow: "hidden",
          position: "relative",
        }}
      >
        <div
          style={{
            width: barWidth,
            height: "100%",
            background: color,
            borderRadius: 6,
          }}
        />
      </div>
      <div style={{ fontSize: 14, color: COLORS.text, fontFamily: FONTS.mono, minWidth: 60 }}>
        {showValue ?? `${value}`}
      </div>
    </div>
  );
}

export function MatrixCell({
  value,
  highlight = false,
  color,
  size = 48,
  delay = 0,
}: {
  value: string | number;
  highlight?: boolean;
  color?: string;
  size?: number;
  delay?: number;
}) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({ frame: frame - delay, fps, config: { damping: 12 } });
  const scale = interpolate(progress, [0, 1], [0.5, 1]);
  const opacity = interpolate(progress, [0, 1], [0, 1]);

  return (
    <div
      style={{
        width: size,
        height: size,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: size * 0.35,
        fontFamily: FONTS.mono,
        fontWeight: 700,
        color: highlight ? COLORS.bg : COLORS.text,
        background: highlight ? (color ?? COLORS.accent) : COLORS.surface,
        border: `1px solid ${highlight ? "transparent" : COLORS.border}`,
        borderRadius: 6,
        transform: `scale(${scale})`,
        opacity,
      }}
    >
      {value}
    </div>
  );
}

export function AnimatedNumber({
  from,
  to,
  delay = 0,
  decimals = 0,
  suffix = "",
  style,
}: {
  from: number;
  to: number;
  delay?: number;
  decimals?: number;
  suffix?: string;
  style?: CSSProperties;
}) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({ frame: frame - delay, fps, config: { damping: 20 } });
  const val = interpolate(progress, [0, 1], [from, to]);
  return (
    <span
      style={{
        fontFamily: FONTS.mono,
        fontWeight: 700,
        color: COLORS.accent,
        ...style,
      }}
    >
      {val.toFixed(decimals)}
      {suffix}
    </span>
  );
}

export function Title({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        fontSize: 28,
        fontWeight: 700,
        color: COLORS.text,
        fontFamily: FONTS.sans,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Subtitle({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div
      style={{
        fontSize: 16,
        color: COLORS.textMuted,
        fontFamily: FONTS.sans,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Label({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <span
      style={{
        fontSize: 13,
        color: COLORS.textDim,
        fontFamily: FONTS.sans,
        ...style,
      }}
    >
      {children}
    </span>
  );
}

export function Box({
  children,
  style,
}: {
  children: ReactNode;
  style?: CSSProperties;
}) {
  return (
    <div
      style={{
        background: COLORS.surface,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 12,
        padding: 20,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export function Arrow({
  x1,
  y1,
  x2,
  y2,
  color = COLORS.accent,
  delay = 0,
  strokeWidth = 2,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color?: string;
  delay?: number;
  strokeWidth?: number;
}) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const progress = spring({ frame: frame - delay, fps, config: { damping: 15 } });
  const cx = interpolate(progress, [0, 1], [x1, x2]);
  const cy = interpolate(progress, [0, 1], [y1, y2]);

  return (
    <svg
      style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    >
      <line x1={x1} y1={y1} x2={cx} y2={cy} stroke={color} strokeWidth={strokeWidth} />
      {progress > 0.8 && (
        <circle cx={cx} cy={cy} r={4} fill={color} />
      )}
    </svg>
  );
}
