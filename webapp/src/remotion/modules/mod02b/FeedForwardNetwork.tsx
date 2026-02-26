import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle, Box } from "../../utils";

/**
 * Feed-Forward Network pipeline: expand (W1) → activate → project (W2).
 * Shows the expand/compress shape change and activation evolution.
 */
export default function FeedForwardNetwork() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const phTitle = 0;
  const phPipeline = 30; // Show W1 → activate → W2
  const phActivation = 120; // Activation evolution
  const phSqReLU = 190; // Why Squared ReLU for BitNet

  const stages = [
    {
      label: "Input",
      dim: "d",
      sub: "2560",
      color: COLORS.blue,
      width: 60,
    },
    {
      label: "W1\n(expand)",
      dim: "4d",
      sub: "6912",
      color: COLORS.purple,
      width: 100,
    },
    {
      label: "Squared\nReLU",
      dim: "4d",
      sub: "6912",
      color: COLORS.amber,
      width: 80,
    },
    {
      label: "W2\n(project)",
      dim: "d",
      sub: "2560",
      color: COLORS.accent,
      width: 60,
    },
  ];

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: COLORS.bg,
        padding: 36,
        display: "flex",
        flexDirection: "column",
        fontFamily: FONTS.sans,
      }}
    >
      <FadeIn>
        <Title>Feed-Forward Network</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Expand {"\u2192"} Activate {"\u2192"} Project: the other half of each
          Transformer block
        </Subtitle>
      </FadeIn>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 28,
        }}
      >
        {/* Pipeline */}
        {frame > phPipeline && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 0,
            }}
          >
            {stages.map((stage, i) => {
              const h = stage.width + 40; // bar height varies to show expansion
              return (
                <div key={i} style={{ display: "flex", alignItems: "center" }}>
                  <SlideIn delay={phPipeline + i * 15} from="bottom">
                    <div style={{ textAlign: "center" }}>
                      <div
                        style={{
                          fontSize: 12,
                          fontWeight: 600,
                          color: stage.color,
                          marginBottom: 4,
                          whiteSpace: "pre-line",
                          lineHeight: 1.3,
                        }}
                      >
                        {stage.label}
                      </div>
                      <div
                        style={{
                          width: 90,
                          height: h,
                          borderRadius: 8,
                          background: `${stage.color}20`,
                          border: `2px solid ${stage.color}60`,
                          display: "flex",
                          flexDirection: "column",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 16,
                            fontWeight: 700,
                            color: stage.color,
                            fontFamily: FONTS.mono,
                          }}
                        >
                          {stage.dim}
                        </div>
                        <div
                          style={{
                            fontSize: 10,
                            color: COLORS.textDim,
                            fontFamily: FONTS.mono,
                          }}
                        >
                          {stage.sub}
                        </div>
                      </div>
                    </div>
                  </SlideIn>
                  {i < stages.length - 1 && (
                    <div
                      style={{
                        width: 32,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <svg width={28} height={20}>
                        <line
                          x1={0}
                          y1={10}
                          x2={20}
                          y2={10}
                          stroke={COLORS.border}
                          strokeWidth={2}
                        />
                        <polygon
                          points="16,5 24,10 16,15"
                          fill={COLORS.border}
                        />
                      </svg>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Activation evolution */}
        {frame > phActivation && (
          <FadeIn delay={phActivation}>
            <div style={{ textAlign: "center" }}>
              <div
                style={{
                  fontSize: 14,
                  fontWeight: 600,
                  color: COLORS.text,
                  marginBottom: 10,
                }}
              >
                Activation Function Evolution
              </div>
              <div
                style={{
                  display: "flex",
                  gap: 16,
                  justifyContent: "center",
                }}
              >
                {[
                  {
                    name: "ReLU",
                    desc: "max(0, x)",
                    era: "Original",
                    active: false,
                  },
                  {
                    name: "GELU",
                    desc: "x\u00B7\u03A6(x)",
                    era: "GPT/BERT",
                    active: false,
                  },
                  {
                    name: "SqReLU",
                    desc: "ReLU(x)\u00B2",
                    era: "BitNet",
                    active: true,
                  },
                ].map((act, i) => {
                  const d = phActivation + 10 + i * 12;
                  const p = spring({
                    frame: frame - d,
                    fps,
                    config: { damping: 12 },
                  });
                  return (
                    <div
                      key={act.name}
                      style={{
                        opacity: interpolate(p, [0, 1], [0, 1]),
                        transform: `scale(${interpolate(p, [0, 1], [0.8, 1])})`,
                      }}
                    >
                      <Box
                        style={{
                          width: 160,
                          padding: "10px 14px",
                          textAlign: "center",
                          borderColor: act.active
                            ? `${COLORS.accent}60`
                            : COLORS.border,
                          background: act.active
                            ? `${COLORS.accent}10`
                            : COLORS.surface,
                        }}
                      >
                        <div
                          style={{
                            fontSize: 15,
                            fontWeight: 700,
                            color: act.active ? COLORS.accent : COLORS.text,
                          }}
                        >
                          {act.name}
                        </div>
                        <div
                          style={{
                            fontSize: 12,
                            fontFamily: FONTS.mono,
                            color: COLORS.textMuted,
                            marginTop: 2,
                          }}
                        >
                          {act.desc}
                        </div>
                        <div
                          style={{
                            fontSize: 10,
                            color: COLORS.textDim,
                            marginTop: 4,
                          }}
                        >
                          {act.era}
                        </div>
                      </Box>
                    </div>
                  );
                })}
              </div>
            </div>
          </FadeIn>
        )}

        {/* Squared ReLU callout */}
        {frame > phSqReLU && (
          <FadeIn delay={phSqReLU}>
            <div
              style={{
                textAlign: "center",
                fontSize: 13,
                color: COLORS.accent,
                fontWeight: 600,
              }}
            >
              Squared ReLU creates sparser activations — complements ternary
              weights
            </div>
          </FadeIn>
        )}
      </div>
    </div>
  );
}
