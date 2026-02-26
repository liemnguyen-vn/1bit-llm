import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle } from "../../utils";

const DATA = [
  { label: "FP32 MUL", energy: 3.7, unit: "pJ", color: COLORS.red, maxBar: 3.7 },
  { label: "FP32 ADD", energy: 0.9, unit: "pJ", color: COLORS.amber, maxBar: 3.7 },
  { label: "FP16 MUL", energy: 1.1, unit: "pJ", color: COLORS.blue, maxBar: 3.7 },
  { label: "FP16 ADD", energy: 0.4, unit: "pJ", color: COLORS.blue, maxBar: 3.7 },
  { label: "INT8 ADD", energy: 0.03, unit: "pJ", color: COLORS.purple, maxBar: 3.7 },
  { label: "Ternary", energy: 0.052, unit: "pJ", color: COLORS.accent, maxBar: 3.7 },
];

export default function EnergyComparison() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const barMaxWidth = 480;

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: COLORS.bg,
        padding: 40,
        display: "flex",
        flexDirection: "column",
        fontFamily: FONTS.sans,
      }}
    >
      <FadeIn>
        <Title>Energy per Operation</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Ternary operations consume up to 71x less energy than FP32 multiply
        </Subtitle>
      </FadeIn>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 10,
          marginTop: 16,
        }}
      >
        {DATA.map((d, i) => {
          const barDelay = 25 + i * 18;
          const p = spring({ frame: frame - barDelay, fps, config: { damping: 15, mass: 0.8 } });
          const barWidth = interpolate(p, [0, 1], [0, (d.energy / d.maxBar) * barMaxWidth]);
          const opacity = interpolate(frame - barDelay, [0, 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          // Ensure minimum visible width for tiny bars
          const minBarWidth = Math.max(barWidth, d.energy < 0.1 ? 4 * p : barWidth);

          return (
            <div
              key={d.label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                opacity,
              }}
            >
              <div
                style={{
                  width: 100,
                  textAlign: "right",
                  fontSize: 13,
                  color: COLORS.textMuted,
                  fontFamily: FONTS.sans,
                  flexShrink: 0,
                }}
              >
                {d.label}
              </div>
              <div
                style={{
                  width: barMaxWidth,
                  height: 32,
                  background: COLORS.surface,
                  borderRadius: 6,
                  overflow: "hidden",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    width: minBarWidth,
                    height: "100%",
                    background: d.color,
                    borderRadius: 6,
                    minWidth: d.energy < 0.1 && p > 0.5 ? 6 : 0,
                  }}
                />
              </div>
              <div
                style={{
                  fontSize: 13,
                  color: COLORS.text,
                  fontFamily: FONTS.mono,
                  minWidth: 80,
                  fontWeight: d.label === "Ternary" ? 700 : 400,
                }}
              >
                {d.energy} {d.unit}
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary cards */}
      {frame > 160 && (
        <FadeIn delay={160}>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: 32,
              marginTop: 8,
            }}
          >
            {[
              { label: "vs FP32 MUL", ratio: "71×", desc: "less energy", color: COLORS.accent },
              { label: "vs FP16 MUL", ratio: "21×", desc: "less energy", color: COLORS.accent },
              { label: "Memory access", ratio: "~10×", desc: "less data moved", color: COLORS.accentLight },
            ].map((item) => (
              <div
                key={item.label}
                style={{
                  background: COLORS.surface,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 8,
                  padding: "10px 20px",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: 11, color: COLORS.textDim }}>{item.label}</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: item.color, fontFamily: FONTS.mono }}>
                  {item.ratio}
                </div>
                <div style={{ fontSize: 11, color: COLORS.textDim }}>{item.desc}</div>
              </div>
            ))}
          </div>
        </FadeIn>
      )}
    </div>
  );
}
