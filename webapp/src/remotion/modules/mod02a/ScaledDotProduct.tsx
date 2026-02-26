import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle, MatrixCell } from "../../utils";

/**
 * Step-by-step scaled dot-product attention with a 3-token numerical example.
 * Phases: Q/K matrices → Q×K^T → scale → softmax → ×V → output
 */
export default function ScaledDotProduct() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Phases (frames)
  const phTitle = 0;
  const phQK = 30; // Show Q, K matrices
  const phDot = 90; // Q × K^T result
  const phScale = 150; // divide by √d_k
  const phSoftmax = 210; // softmax row
  const phOutput = 270; // × V → output

  // Toy Q, K, V (d_k = 2, 3 tokens)
  const Q = [
    [1, 0],
    [0, 1],
    [1, 1],
  ];
  const K = [
    [1, 1],
    [0, 1],
    [1, 0],
  ];
  // Q×K^T raw
  const raw = [
    [1, 0, 1],
    [1, 1, 0],
    [2, 1, 1],
  ];
  const dk = 2;
  const sqrtDk = Math.sqrt(dk); // ~1.41
  // scaled = raw / sqrt(2)
  const scaled = raw.map((r) => r.map((v) => v / sqrtDk));
  // softmax per row (precomputed to keep it deterministic)
  const softmaxRows = [
    [0.47, 0.21, 0.32],
    [0.39, 0.39, 0.22],
    [0.50, 0.27, 0.23],
  ];

  const cellSize = 42;

  function renderMatrix(
    mat: number[][],
    color: string,
    delay: number,
    decimals = 0,
  ) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {mat.map((row, ri) => (
          <div key={ri} style={{ display: "flex", gap: 3 }}>
            {row.map((val, ci) => (
              <MatrixCell
                key={ci}
                value={val.toFixed(decimals)}
                size={cellSize}
                delay={delay + ri * 5 + ci * 3}
                color={color}
                highlight={false}
              />
            ))}
          </div>
        ))}
      </div>
    );
  }

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
        <Title>Scaled Dot-Product Attention</Title>
        <Subtitle style={{ marginTop: 4 }}>
          softmax(Q K^T / {"\u221A"}d_k) {"\u00D7"} V &mdash; step by step
        </Subtitle>
      </FadeIn>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 16,
        }}
      >
        {/* Row 1: Q and K */}
        {frame > phQK && (
          <FadeIn delay={phQK}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 32,
              }}
            >
              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    color: COLORS.blue,
                    marginBottom: 6,
                  }}
                >
                  Q (3{"\u00D7"}2)
                </div>
                {renderMatrix(Q, COLORS.blue, phQK)}
              </div>
              <div
                style={{
                  fontSize: 20,
                  color: COLORS.textDim,
                  fontWeight: 700,
                }}
              >
                {"\u00D7"}
              </div>
              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    color: COLORS.amber,
                    marginBottom: 6,
                  }}
                >
                  K^T (2{"\u00D7"}3)
                </div>
                {renderMatrix(
                  K[0].map((_, ci) => K.map((r) => r[ci])),
                  COLORS.amber,
                  phQK + 15,
                )}
              </div>
            </div>
          </FadeIn>
        )}

        {/* Row 2: Raw scores → Scaled → Softmax */}
        {frame > phDot && (
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              justifyContent: "center",
              gap: 24,
            }}
          >
            {/* Raw scores */}
            <FadeIn delay={phDot}>
              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontSize: 12,
                    color: COLORS.textDim,
                    marginBottom: 4,
                  }}
                >
                  Q {"\u00D7"} K^T
                </div>
                {renderMatrix(raw, COLORS.purple, phDot)}
              </div>
            </FadeIn>

            {/* Scale arrow */}
            {frame > phScale && (
              <FadeIn delay={phScale}>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    marginTop: 24,
                  }}
                >
                  <div
                    style={{
                      fontSize: 18,
                      color: COLORS.textDim,
                    }}
                  >
                    {"\u2192"}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: COLORS.accent,
                      fontFamily: FONTS.mono,
                    }}
                  >
                    /{"\u221A"}2
                  </div>
                </div>
              </FadeIn>
            )}

            {/* Scaled scores */}
            {frame > phScale && (
              <FadeIn delay={phScale}>
                <div style={{ textAlign: "center" }}>
                  <div
                    style={{
                      fontSize: 12,
                      color: COLORS.textDim,
                      marginBottom: 4,
                    }}
                  >
                    Scaled
                  </div>
                  {renderMatrix(scaled, COLORS.purple, phScale, 2)}
                </div>
              </FadeIn>
            )}

            {/* Softmax arrow */}
            {frame > phSoftmax && (
              <FadeIn delay={phSoftmax}>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    marginTop: 24,
                  }}
                >
                  <div style={{ fontSize: 18, color: COLORS.textDim }}>
                    {"\u2192"}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: COLORS.accent,
                      fontFamily: FONTS.mono,
                    }}
                  >
                    softmax
                  </div>
                </div>
              </FadeIn>
            )}

            {/* Softmax output */}
            {frame > phSoftmax && (
              <FadeIn delay={phSoftmax}>
                <div style={{ textAlign: "center" }}>
                  <div
                    style={{
                      fontSize: 12,
                      color: COLORS.textDim,
                      marginBottom: 4,
                    }}
                  >
                    Attention Weights
                  </div>
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: 3,
                    }}
                  >
                    {softmaxRows.map((row, ri) => (
                      <div key={ri} style={{ display: "flex", gap: 3 }}>
                        {row.map((val, ci) => {
                          const d =
                            phSoftmax + 10 + ri * 5 + ci * 3;
                          const p = spring({
                            frame: frame - d,
                            fps,
                            config: { damping: 12 },
                          });
                          return (
                            <div
                              key={ci}
                              style={{
                                width: cellSize,
                                height: cellSize,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                fontSize: 12,
                                fontFamily: FONTS.mono,
                                fontWeight: 700,
                                borderRadius: 6,
                                background: `rgba(16,185,129,${val * 0.7})`,
                                color:
                                  val > 0.4 ? COLORS.bg : COLORS.text,
                                border: `1px solid ${COLORS.border}`,
                                opacity: interpolate(p, [0, 1], [0, 1]),
                                transform: `scale(${interpolate(p, [0, 1], [0.5, 1])})`,
                              }}
                            >
                              {val.toFixed(2)}
                            </div>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                </div>
              </FadeIn>
            )}
          </div>
        )}

        {/* Final callout */}
        {frame > phOutput && (
          <FadeIn delay={phOutput}>
            <div
              style={{
                textAlign: "center",
                fontSize: 14,
                color: COLORS.accent,
                fontWeight: 600,
                marginTop: 8,
              }}
            >
              Multiply by V to get context-aware output for each token
            </div>
          </FadeIn>
        )}
      </div>
    </div>
  );
}
