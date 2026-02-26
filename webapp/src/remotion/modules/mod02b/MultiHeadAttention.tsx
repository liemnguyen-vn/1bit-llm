import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle, Box } from "../../utils";

/**
 * Multi-Head Attention with Grouped-Query Attention (GQA).
 * Shows head splitting, parallel attention, concatenation, and GQA sharing.
 */
export default function MultiHeadAttention() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const phTitle = 0;
  const phHeads = 30; // Show multiple heads
  const phParallel = 90; // Parallel attention computation
  const phConcat = 160; // Concatenate + W_o
  const phGQA = 220; // GQA: KV sharing

  const headColors = [COLORS.blue, COLORS.amber, COLORS.purple, COLORS.cyan];
  const headLabels = ["Syntax", "Coref", "Semantic", "Position"];

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
        <Title>Multi-Head Attention</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Multiple heads capture different relationship types
        </Subtitle>
      </FadeIn>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 20,
        }}
      >
        {/* Heads row */}
        {frame > phHeads && (
          <div
            style={{
              display: "flex",
              gap: 16,
              justifyContent: "center",
            }}
          >
            {headLabels.map((label, i) => (
              <SlideIn key={label} delay={phHeads + i * 10} from="top">
                <Box
                  style={{
                    width: 170,
                    padding: "10px 14px",
                    textAlign: "center",
                    borderColor: `${headColors[i]}40`,
                  }}
                >
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: 700,
                      color: headColors[i],
                      marginBottom: 4,
                    }}
                  >
                    Head {i + 1}
                  </div>
                  <div style={{ fontSize: 11, color: COLORS.textDim }}>
                    {label}
                  </div>
                  {/* Mini Q/K/V blocks */}
                  {frame > phParallel && (
                    <FadeIn delay={phParallel + i * 8}>
                      <div
                        style={{
                          display: "flex",
                          gap: 4,
                          justifyContent: "center",
                          marginTop: 8,
                        }}
                      >
                        {["Q", "K", "V"].map((m) => (
                          <div
                            key={m}
                            style={{
                              width: 28,
                              height: 22,
                              borderRadius: 4,
                              background: `${headColors[i]}25`,
                              border: `1px solid ${headColors[i]}50`,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              fontSize: 10,
                              fontFamily: FONTS.mono,
                              fontWeight: 600,
                              color: headColors[i],
                            }}
                          >
                            {m}
                          </div>
                        ))}
                      </div>
                    </FadeIn>
                  )}
                </Box>
              </SlideIn>
            ))}
          </div>
        )}

        {/* Concatenate + W_o */}
        {frame > phConcat && (
          <FadeIn delay={phConcat}>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 8,
              }}
            >
              <div style={{ fontSize: 18, color: COLORS.textDim }}>
                {"\u2193"} Concatenate
              </div>
              <Box
                style={{
                  padding: "10px 32px",
                  borderColor: `${COLORS.accent}40`,
                  textAlign: "center",
                }}
              >
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 600,
                    color: COLORS.accent,
                  }}
                >
                  W_o Projection
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: COLORS.textDim,
                    marginTop: 2,
                  }}
                >
                  Concat(head1..h) {"\u00D7"} W_o {"\u2192"} output
                </div>
              </Box>
            </div>
          </FadeIn>
        )}

        {/* GQA section */}
        {frame > phGQA && (
          <FadeIn delay={phGQA}>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 8,
                marginTop: 8,
              }}
            >
              <div
                style={{
                  fontSize: 15,
                  fontWeight: 700,
                  color: COLORS.amber,
                }}
              >
                Grouped-Query Attention (GQA)
              </div>
              <div
                style={{
                  display: "flex",
                  gap: 6,
                  alignItems: "center",
                }}
              >
                {/* 5 KV groups, each shared by 4 Q heads */}
                {[0, 1, 2, 3, 4].map((g) => {
                  const d = phGQA + 15 + g * 8;
                  const p = spring({
                    frame: frame - d,
                    fps,
                    config: { damping: 12 },
                  });
                  return (
                    <div
                      key={g}
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 3,
                        opacity: interpolate(p, [0, 1], [0, 1]),
                        transform: `scale(${interpolate(p, [0, 1], [0.5, 1])})`,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          gap: 2,
                        }}
                      >
                        {[0, 1, 2, 3].map((q) => (
                          <div
                            key={q}
                            style={{
                              width: 14,
                              height: 10,
                              borderRadius: 2,
                              background: `${COLORS.blue}60`,
                              border: `1px solid ${COLORS.blue}40`,
                            }}
                          />
                        ))}
                      </div>
                      <div
                        style={{
                          fontSize: 8,
                          color: COLORS.textDim,
                        }}
                      >
                        4Q
                      </div>
                      <div
                        style={{
                          width: 56,
                          height: 14,
                          borderRadius: 3,
                          background: `${COLORS.amber}30`,
                          border: `1px solid ${COLORS.amber}50`,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 8,
                          fontFamily: FONTS.mono,
                          color: COLORS.amber,
                        }}
                      >
                        KV {g + 1}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: COLORS.textMuted,
                  textAlign: "center",
                }}
              >
                BitNet 2B4T: 20 Q heads share 5 KV heads (50% fewer attention
                params)
              </div>
            </div>
          </FadeIn>
        )}
      </div>
    </div>
  );
}
