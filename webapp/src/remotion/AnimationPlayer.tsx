"use client";

import { Player } from "@remotion/player";
import type { AnimationMeta } from "./types";
import { COLORS, FONTS } from "./theme";

export default function AnimationPlayer({ animation }: { animation: AnimationMeta }) {
  return (
    <div
      style={{
        marginBottom: 24,
        borderRadius: 12,
        overflow: "hidden",
        border: `1px solid ${COLORS.border}`,
        background: COLORS.surface,
      }}
    >
      <div
        style={{
          padding: "10px 16px",
          borderBottom: `1px solid ${COLORS.border}`,
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: COLORS.accent,
          }}
        />
        <span
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: COLORS.textMuted,
            fontFamily: FONTS.sans,
          }}
        >
          {animation.title}
        </span>
      </div>
      <Player
        component={animation.component}
        compositionWidth={animation.width}
        compositionHeight={animation.height}
        durationInFrames={animation.durationInFrames}
        fps={animation.fps}
        autoPlay
        loop
        style={{ width: "100%" }}
        controls
      />
    </div>
  );
}
