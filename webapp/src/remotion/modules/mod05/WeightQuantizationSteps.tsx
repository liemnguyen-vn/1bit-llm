import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle } from "../../utils";

const STEPS = [
  {
    title: "1. Compute Scale",
    formula: "α = mean(|W|)",
    example: "α = mean(|[0.73, -1.12, 0.05, -0.88]|) = 0.695",
    color: COLORS.blue,
  },
  {
    title: "2. Scale Weights",
    formula: "W' = W / α",
    example: "W' = [1.05, -1.61, 0.07, -1.27]",
    color: COLORS.purple,
  },
  {
    title: "3. Round & Clamp",
    formula: "Ŵ = clamp(round(W'), -1, 1)",
    example: "Ŵ = [+1, -1, 0, -1]",
    color: COLORS.amber,
  },
  {
    title: "4. Dequantize",
    formula: "W̃ = Ŵ × α",
    example: "W̃ = [0.695, -0.695, 0, -0.695]",
    color: COLORS.accent,
  },
];

export default function WeightQuantizationSteps() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

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
        <Title>Weight Quantization Steps</Title>
        <Subtitle style={{ marginTop: 4 }}>
          How full-precision weights become ternary values
        </Subtitle>
      </FadeIn>

      <div style={{ display: "flex", gap: 16, marginTop: 28, flex: 1 }}>
        {STEPS.map((step, i) => {
          const stepDelay = 30 + i * 40;
          const p = spring({ frame: frame - stepDelay, fps, config: { damping: 15 } });

          return (
            <SlideIn key={i} delay={stepDelay} from="bottom" style={{ flex: 1 }}>
              <div
                style={{
                  background: COLORS.surface,
                  border: `1px solid ${step.color}40`,
                  borderRadius: 12,
                  padding: 16,
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: step.color,
                    marginBottom: 12,
                  }}
                >
                  {step.title}
                </div>

                <div
                  style={{
                    background: `${step.color}15`,
                    borderRadius: 8,
                    padding: "10px 12px",
                    marginBottom: 12,
                    fontFamily: FONTS.mono,
                    fontSize: 13,
                    color: step.color,
                    fontWeight: 600,
                    textAlign: "center",
                  }}
                >
                  {step.formula}
                </div>

                {frame > stepDelay + 20 && (
                  <FadeIn delay={stepDelay + 20}>
                    <div
                      style={{
                        fontSize: 11,
                        color: COLORS.textMuted,
                        fontFamily: FONTS.mono,
                        lineHeight: 1.6,
                        wordBreak: "break-all",
                      }}
                    >
                      {step.example}
                    </div>
                  </FadeIn>
                )}

                {/* Arrow */}
                {i < STEPS.length - 1 && (
                  <div
                    style={{
                      position: "absolute",
                      right: -12,
                      top: "50%",
                      color: COLORS.textDim,
                      fontSize: 16,
                    }}
                  >
                    →
                  </div>
                )}
              </div>
            </SlideIn>
          );
        })}
      </div>

      {frame > 220 && (
        <FadeIn delay={220}>
          <div
            style={{
              textAlign: "center",
              marginTop: 16,
              fontSize: 14,
              color: COLORS.accent,
              fontWeight: 600,
            }}
          >
            Scale factor α preserves the magnitude information lost during quantization
          </div>
        </FadeIn>
      )}
    </div>
  );
}
