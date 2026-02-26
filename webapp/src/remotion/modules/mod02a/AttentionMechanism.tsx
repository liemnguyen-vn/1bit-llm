import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle, Box } from "../../utils";

const TOKENS = ["The", "cat", "sat"];

export default function AttentionMechanism() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const phase1 = 0;    // Title + tokens
  const phase2 = 45;   // Q, K, V projections
  const phase3 = 120;  // Attention scores
  const phase4 = 210;  // Weighted output
  const phase5 = 300;  // Final summary

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
        <Title>Self-Attention Mechanism</Title>
        <Subtitle style={{ marginTop: 4 }}>
          How tokens attend to each other via Q, K, V projections
        </Subtitle>
      </FadeIn>

      <div style={{ flex: 1, display: "flex", flexDirection: "column", marginTop: 24 }}>
        {/* Input tokens */}
        <div style={{ display: "flex", gap: 16, justifyContent: "center", marginBottom: 20 }}>
          {TOKENS.map((tok, i) => (
            <SlideIn key={tok} delay={10 + i * 8} from="top">
              <div
                style={{
                  background: COLORS.surface,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 8,
                  padding: "8px 20px",
                  fontSize: 16,
                  fontWeight: 600,
                  color: COLORS.text,
                  fontFamily: FONTS.mono,
                  textAlign: "center",
                }}
              >
                {tok}
              </div>
            </SlideIn>
          ))}
        </div>

        {/* Q, K, V projections */}
        {frame > phase2 && (
          <div style={{ display: "flex", gap: 24, justifyContent: "center", marginBottom: 20 }}>
            {[
              { label: "Q (Query)", color: COLORS.blue, desc: "What am I looking for?" },
              { label: "K (Key)", color: COLORS.amber, desc: "What do I contain?" },
              { label: "V (Value)", color: COLORS.accent, desc: "What do I output?" },
            ].map((proj, i) => (
              <SlideIn key={proj.label} delay={phase2 + i * 15} from="bottom">
                <Box
                  style={{
                    padding: "12px 18px",
                    borderColor: `${proj.color}40`,
                    textAlign: "center",
                    width: 200,
                  }}
                >
                  <div style={{ fontSize: 14, fontWeight: 700, color: proj.color, marginBottom: 4 }}>
                    {proj.label}
                  </div>
                  <div style={{ fontSize: 11, color: COLORS.textDim }}>{proj.desc}</div>
                  <div
                    style={{
                      display: "flex",
                      gap: 6,
                      justifyContent: "center",
                      marginTop: 8,
                    }}
                  >
                    {TOKENS.map((_, j) => {
                      const cellDelay = phase2 + i * 15 + 20 + j * 5;
                      const p = spring({ frame: frame - cellDelay, fps, config: { damping: 12 } });
                      return (
                        <div
                          key={j}
                          style={{
                            width: 36,
                            height: 24,
                            borderRadius: 4,
                            background: `${proj.color}30`,
                            border: `1px solid ${proj.color}50`,
                            opacity: interpolate(p, [0, 1], [0, 1]),
                            transform: `scale(${interpolate(p, [0, 1], [0.5, 1])})`,
                          }}
                        />
                      );
                    })}
                  </div>
                </Box>
              </SlideIn>
            ))}
          </div>
        )}

        {/* Attention scores */}
        {frame > phase3 && (
          <FadeIn delay={phase3}>
            <div style={{ textAlign: "center", marginBottom: 16 }}>
              <div style={{ fontSize: 13, color: COLORS.textDim, marginBottom: 8 }}>
                Attention Scores (Q x K^T / sqrt(d))
              </div>
              <div style={{ display: "flex", justifyContent: "center", gap: 4 }}>
                {[
                  [0.7, 0.2, 0.1],
                  [0.1, 0.6, 0.3],
                  [0.15, 0.25, 0.6],
                ].map((row, ri) => (
                  <div key={ri} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    {row.map((val, ci) => {
                      const cellDelay = phase3 + 15 + ri * 8 + ci * 5;
                      const p = spring({ frame: frame - cellDelay, fps, config: { damping: 12 } });
                      const opacity = interpolate(p, [0, 1], [0, 1]);
                      return (
                        <div
                          key={ci}
                          style={{
                            width: 52,
                            height: 32,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            borderRadius: 4,
                            fontSize: 12,
                            fontFamily: FONTS.mono,
                            fontWeight: 600,
                            opacity,
                            background: `rgba(16, 185, 129, ${val * 0.6})`,
                            color: val > 0.4 ? COLORS.bg : COLORS.text,
                            border: `1px solid ${COLORS.border}`,
                          }}
                        >
                          {val.toFixed(1)}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>
        )}

        {/* Weighted output */}
        {frame > phase4 && (
          <FadeIn delay={phase4}>
            <div style={{ display: "flex", justifyContent: "center", gap: 12, marginTop: 8 }}>
              {TOKENS.map((tok, i) => {
                const p = spring({ frame: frame - phase4 - 15 - i * 10, fps, config: { damping: 15 } });
                return (
                  <div
                    key={tok}
                    style={{
                      background: `${COLORS.accent}20`,
                      border: `1px solid ${COLORS.accent}50`,
                      borderRadius: 8,
                      padding: "8px 20px",
                      textAlign: "center",
                      opacity: interpolate(p, [0, 1], [0, 1]),
                      transform: `scale(${interpolate(p, [0, 1], [0.8, 1])})`,
                    }}
                  >
                    <div style={{ fontSize: 11, color: COLORS.textDim }}>Output</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: COLORS.accent, fontFamily: FONTS.mono }}>
                      {tok}&apos;
                    </div>
                  </div>
                );
              })}
            </div>
          </FadeIn>
        )}

        {frame > phase5 && (
          <FadeIn delay={phase5}>
            <div
              style={{
                textAlign: "center",
                marginTop: 16,
                fontSize: 14,
                color: COLORS.accent,
                fontWeight: 600,
              }}
            >
              Each output is a weighted combination of all value vectors
            </div>
          </FadeIn>
        )}
      </div>
    </div>
  );
}
