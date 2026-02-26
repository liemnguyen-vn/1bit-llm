import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle } from "../../utils";

const INPUT = [3, -2, 5, 1];
const WEIGHTS_FP = [0.73, -1.12, 0.05, -0.88];
const WEIGHTS_TERNARY = [1, -1, 0, -1];

export default function MatrixMultiplication() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const phase1End = 60;   // Show standard multiply
  const phase2Start = 90; // Show ternary
  const phase3Start = 180; // Show result comparison

  const fp32Result = INPUT.reduce((s, v, i) => s + v * WEIGHTS_FP[i], 0);
  const ternaryResult = INPUT.reduce((s, v, i) => s + v * WEIGHTS_TERNARY[i], 0);

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
        <Title>Matrix Multiplication with Ternary Weights</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Ternary weights eliminate multiplications entirely
        </Subtitle>
      </FadeIn>

      <div style={{ display: "flex", gap: 40, marginTop: 40, flex: 1 }}>
        {/* FP32 side */}
        <SlideIn delay={15} from="left" style={{ flex: 1 }}>
          <div
            style={{
              background: COLORS.surface,
              borderRadius: 12,
              padding: 20,
              border: `1px solid ${COLORS.border}`,
              height: "100%",
            }}
          >
            <div style={{ fontSize: 15, fontWeight: 600, color: COLORS.blue, marginBottom: 16 }}>
              Standard (FP32)
            </div>
            {INPUT.map((inp, i) => {
              const rowDelay = 30 + i * 15;
              const rowProgress = spring({ frame: frame - rowDelay, fps, config: { damping: 15 } });
              const opacity = interpolate(rowProgress, [0, 1], [0, 1]);
              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 10,
                    opacity,
                    fontFamily: FONTS.mono,
                    fontSize: 15,
                  }}
                >
                  <span style={{ color: COLORS.text, width: 30, textAlign: "right" }}>{inp}</span>
                  <span style={{ color: COLORS.textDim }}>x</span>
                  <span style={{ color: COLORS.blue, width: 50, textAlign: "right" }}>
                    {WEIGHTS_FP[i].toFixed(2)}
                  </span>
                  <span style={{ color: COLORS.textDim }}>=</span>
                  <span style={{ color: COLORS.textMuted }}>
                    {(inp * WEIGHTS_FP[i]).toFixed(2)}
                  </span>
                  <span
                    style={{
                      marginLeft: 8,
                      fontSize: 11,
                      color: COLORS.textDim,
                      background: COLORS.surfaceLight,
                      padding: "2px 6px",
                      borderRadius: 4,
                    }}
                  >
                    MUL
                  </span>
                </div>
              );
            })}
            {frame > phase1End && (
              <FadeIn delay={phase1End}>
                <div
                  style={{
                    marginTop: 16,
                    paddingTop: 12,
                    borderTop: `1px solid ${COLORS.border}`,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    fontFamily: FONTS.mono,
                    fontSize: 15,
                  }}
                >
                  <span style={{ color: COLORS.textDim }}>Sum =</span>
                  <span style={{ color: COLORS.blue, fontWeight: 700 }}>
                    {fp32Result.toFixed(2)}
                  </span>
                  <span style={{ fontSize: 11, color: COLORS.textDim }}>
                    (4 multiplications)
                  </span>
                </div>
              </FadeIn>
            )}
          </div>
        </SlideIn>

        {/* Ternary side */}
        <SlideIn delay={phase2Start} from="right" style={{ flex: 1 }}>
          <div
            style={{
              background: COLORS.surface,
              borderRadius: 12,
              padding: 20,
              border: `1px solid ${COLORS.accent}40`,
              height: "100%",
            }}
          >
            <div style={{ fontSize: 15, fontWeight: 600, color: COLORS.accent, marginBottom: 16 }}>
              Ternary ({"{-1, 0, +1}"})
            </div>
            {INPUT.map((inp, i) => {
              const rowDelay = phase2Start + 10 + i * 15;
              const rowProgress = spring({ frame: frame - rowDelay, fps, config: { damping: 15 } });
              const opacity = interpolate(rowProgress, [0, 1], [0, 1]);
              const w = WEIGHTS_TERNARY[i];
              const op = w === 1 ? "ADD" : w === -1 ? "NEG" : "SKIP";
              const opColor = w === 0 ? COLORS.textDim : COLORS.accent;
              const result = w === 1 ? `+${inp}` : w === -1 ? `${-inp}` : "0";
              return (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 10,
                    opacity,
                    fontFamily: FONTS.mono,
                    fontSize: 15,
                  }}
                >
                  <span style={{ color: COLORS.text, width: 30, textAlign: "right" }}>{inp}</span>
                  <span style={{ color: COLORS.textDim }}>x</span>
                  <span
                    style={{
                      color: w === 0 ? COLORS.textDim : COLORS.accent,
                      width: 50,
                      textAlign: "right",
                      fontWeight: 700,
                    }}
                  >
                    {w > 0 ? `+${w}` : w}
                  </span>
                  <span style={{ color: COLORS.textDim }}>=</span>
                  <span style={{ color: COLORS.textMuted }}>{result}</span>
                  <span
                    style={{
                      marginLeft: 8,
                      fontSize: 11,
                      color: opColor,
                      background: `${opColor}15`,
                      padding: "2px 6px",
                      borderRadius: 4,
                      fontWeight: 600,
                    }}
                  >
                    {op}
                  </span>
                </div>
              );
            })}
            {frame > phase3Start && (
              <FadeIn delay={phase3Start}>
                <div
                  style={{
                    marginTop: 16,
                    paddingTop: 12,
                    borderTop: `1px solid ${COLORS.border}`,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    fontFamily: FONTS.mono,
                    fontSize: 15,
                  }}
                >
                  <span style={{ color: COLORS.textDim }}>Sum =</span>
                  <span style={{ color: COLORS.accent, fontWeight: 700 }}>
                    {ternaryResult}
                  </span>
                  <span style={{ fontSize: 11, color: COLORS.accent }}>
                    (0 multiplications!)
                  </span>
                </div>
              </FadeIn>
            )}
          </div>
        </SlideIn>
      </div>

      {frame > phase3Start + 30 && (
        <FadeIn delay={phase3Start + 30}>
          <div
            style={{
              marginTop: 16,
              textAlign: "center",
              fontSize: 15,
              color: COLORS.accent,
              fontWeight: 600,
            }}
          >
            Only additions and sign flips needed — no floating-point multiplications
          </div>
        </FadeIn>
      )}
    </div>
  );
}
