import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle, Box } from "../../utils";

export default function STEVisualization() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const phase1 = 0;     // Title
  const phase2 = 30;    // Forward pass
  const phase3 = 150;   // Backward pass
  const phase4 = 240;   // Key insight

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
        <Title>Straight-Through Estimator (STE)</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Forward pass uses quantized weights; backward pass passes gradients through
        </Subtitle>
      </FadeIn>

      <div style={{ display: "flex", gap: 40, marginTop: 32, flex: 1 }}>
        {/* Forward pass */}
        <SlideIn delay={phase2} from="left" style={{ flex: 1 }}>
          <Box style={{ height: "100%", borderColor: `${COLORS.blue}40` }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: COLORS.blue, marginBottom: 16 }}>
              Forward Pass
            </div>

            {/* Flow diagram */}
            {[
              { label: "Full-precision weights", val: "W = 0.73", color: COLORS.text },
              { label: "Quantize", val: "Q(W) → +1", color: COLORS.amber, isOp: true },
              { label: "Use quantized weight", val: "y = x · (+1)", color: COLORS.blue },
            ].map((step, i) => {
              const stepDelay = phase2 + 20 + i * 25;
              const p = spring({ frame: frame - stepDelay, fps, config: { damping: 15 } });
              return (
                <div key={i}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 12,
                      opacity: interpolate(p, [0, 1], [0, 1]),
                      marginBottom: 8,
                    }}
                  >
                    {step.isOp ? (
                      <div
                        style={{
                          background: `${step.color}20`,
                          border: `1px solid ${step.color}40`,
                          borderRadius: 6,
                          padding: "6px 12px",
                          flex: 1,
                        }}
                      >
                        <div style={{ fontSize: 11, color: COLORS.textDim }}>{step.label}</div>
                        <div style={{ fontSize: 14, fontFamily: FONTS.mono, fontWeight: 600, color: step.color }}>
                          {step.val}
                        </div>
                      </div>
                    ) : (
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 11, color: COLORS.textDim }}>{step.label}</div>
                        <div style={{ fontSize: 14, fontFamily: FONTS.mono, fontWeight: 600, color: step.color }}>
                          {step.val}
                        </div>
                      </div>
                    )}
                  </div>
                  {i < 2 && (
                    <div
                      style={{
                        textAlign: "center",
                        color: COLORS.textDim,
                        fontSize: 16,
                        marginBottom: 8,
                        opacity: interpolate(p, [0, 1], [0, 1]),
                      }}
                    >
                      ↓
                    </div>
                  )}
                </div>
              );
            })}

            <div
              style={{
                marginTop: 12,
                padding: "6px 10px",
                background: `${COLORS.blue}15`,
                borderRadius: 6,
                fontSize: 12,
                color: COLORS.blue,
                textAlign: "center",
              }}
            >
              Efficient: only adds/negations at inference
            </div>
          </Box>
        </SlideIn>

        {/* Backward pass */}
        {frame > phase3 && (
          <SlideIn delay={phase3} from="right" style={{ flex: 1 }}>
            <Box style={{ height: "100%", borderColor: `${COLORS.accent}40` }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: COLORS.accent, marginBottom: 16 }}>
                Backward Pass (STE)
              </div>

              {[
                { label: "Gradient from loss", val: "∂L/∂y = -0.42", color: COLORS.text },
                { label: "STE trick", val: "∂Q/∂W ≈ 1", color: COLORS.amber, isOp: true },
                { label: "Update full-precision", val: "W -= lr · (-0.42)", color: COLORS.accent },
              ].map((step, i) => {
                const stepDelay = phase3 + 20 + i * 25;
                const p = spring({ frame: frame - stepDelay, fps, config: { damping: 15 } });
                return (
                  <div key={i}>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 12,
                        opacity: interpolate(p, [0, 1], [0, 1]),
                        marginBottom: 8,
                      }}
                    >
                      {step.isOp ? (
                        <div
                          style={{
                            background: `${step.color}20`,
                            border: `1px solid ${step.color}40`,
                            borderRadius: 6,
                            padding: "6px 12px",
                            flex: 1,
                          }}
                        >
                          <div style={{ fontSize: 11, color: COLORS.textDim }}>{step.label}</div>
                          <div style={{ fontSize: 14, fontFamily: FONTS.mono, fontWeight: 600, color: step.color }}>
                            {step.val}
                          </div>
                          <div style={{ fontSize: 10, color: COLORS.textDim, marginTop: 2 }}>
                            Pretend quantization is identity
                          </div>
                        </div>
                      ) : (
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 11, color: COLORS.textDim }}>{step.label}</div>
                          <div style={{ fontSize: 14, fontFamily: FONTS.mono, fontWeight: 600, color: step.color }}>
                            {step.val}
                          </div>
                        </div>
                      )}
                    </div>
                    {i < 2 && (
                      <div
                        style={{
                          textAlign: "center",
                          color: COLORS.textDim,
                          fontSize: 16,
                          marginBottom: 8,
                          opacity: interpolate(p, [0, 1], [0, 1]),
                        }}
                      >
                        ↓
                      </div>
                    )}
                  </div>
                );
              })}

              <div
                style={{
                  marginTop: 12,
                  padding: "6px 10px",
                  background: `${COLORS.accent}15`,
                  borderRadius: 6,
                  fontSize: 12,
                  color: COLORS.accent,
                  textAlign: "center",
                }}
              >
                Gradients flow as if quantization didn&apos;t happen
              </div>
            </Box>
          </SlideIn>
        )}
      </div>

      {frame > phase4 && (
        <FadeIn delay={phase4}>
          <div
            style={{
              textAlign: "center",
              marginTop: 16,
              fontSize: 14,
              color: COLORS.accent,
              fontWeight: 600,
            }}
          >
            STE enables training with discrete weights by bypassing the non-differentiable quantization step
          </div>
        </FadeIn>
      )}
    </div>
  );
}
