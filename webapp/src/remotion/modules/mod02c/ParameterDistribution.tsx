import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle, AnimatedBar } from "../../utils";

/**
 * Bar chart showing where parameters live in a Transformer LLM,
 * and what BitNet quantizes vs. keeps full-precision.
 */
export default function ParameterDistribution() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const phTitle = 0;
  const phBars = 30;
  const phBitnet = 150;
  const phSummary = 220;

  const components = [
    { label: "Attn Q,K,V,O", pct: 33, color: COLORS.blue, quantized: true },
    { label: "FFN W1, W2", pct: 62, color: COLORS.purple, quantized: true },
    { label: "Embedding", pct: 3.5, color: COLORS.amber, quantized: false },
    { label: "Norms", pct: 0.1, color: COLORS.cyan, quantized: false },
    { label: "Output head", pct: 1.4, color: COLORS.orange, quantized: false },
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
        <Title>Parameter Distribution</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Where the parameters live in a Transformer LLM
        </Subtitle>
      </FadeIn>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 4,
        }}
      >
        {/* Bars */}
        {frame > phBars &&
          components.map((comp, i) => (
            <AnimatedBar
              key={comp.label}
              label={comp.label}
              value={comp.pct}
              maxValue={65}
              color={comp.color}
              delay={phBars + i * 12}
              width={420}
              height={32}
              showValue={`${comp.pct}%`}
            />
          ))}

        {/* BitNet overlay */}
        {frame > phBitnet && (
          <FadeIn delay={phBitnet}>
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                gap: 20,
                marginTop: 16,
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <div
                  style={{
                    width: 14,
                    height: 14,
                    borderRadius: 3,
                    background: COLORS.accent,
                  }}
                />
                <span
                  style={{
                    fontSize: 13,
                    color: COLORS.accent,
                    fontWeight: 600,
                  }}
                >
                  BitNet quantizes (~95%)
                </span>
              </div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <div
                  style={{
                    width: 14,
                    height: 14,
                    borderRadius: 3,
                    background: COLORS.textDim,
                  }}
                />
                <span
                  style={{
                    fontSize: 13,
                    color: COLORS.textDim,
                    fontWeight: 600,
                  }}
                >
                  Kept full precision (~5%)
                </span>
              </div>
            </div>

            {/* Table */}
            <div
              style={{
                display: "flex",
                justifyContent: "center",
                marginTop: 12,
              }}
            >
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "auto auto auto",
                  gap: "4px 20px",
                  fontSize: 12,
                  fontFamily: FONTS.mono,
                }}
              >
                <div style={{ color: COLORS.textDim, fontWeight: 600 }}>
                  Layer
                </div>
                <div style={{ color: COLORS.textDim, fontWeight: 600 }}>
                  Standard
                </div>
                <div style={{ color: COLORS.textDim, fontWeight: 600 }}>
                  BitNet
                </div>

                {[
                  ["Q, K, V, O", "FP16", "{-1,0,1}"],
                  ["FFN W1, W2", "FP16", "{-1,0,1}"],
                  ["Embeddings", "FP16", "FP16"],
                  ["RMSNorm", "FP32", "FP32"],
                ].map(([layer, std, bit]) => {
                  const isQuantized = bit === "{-1,0,1}";
                  return [
                    <div
                      key={`${layer}-l`}
                      style={{ color: COLORS.textMuted }}
                    >
                      {layer}
                    </div>,
                    <div key={`${layer}-s`} style={{ color: COLORS.text }}>
                      {std}
                    </div>,
                    <div
                      key={`${layer}-b`}
                      style={{
                        color: isQuantized ? COLORS.accent : COLORS.textMuted,
                        fontWeight: isQuantized ? 700 : 400,
                      }}
                    >
                      {bit}
                    </div>,
                  ];
                })}
              </div>
            </div>
          </FadeIn>
        )}

        {/* Summary */}
        {frame > phSummary && (
          <FadeIn delay={phSummary}>
            <div
              style={{
                textAlign: "center",
                fontSize: 14,
                color: COLORS.accent,
                fontWeight: 600,
                marginTop: 12,
              }}
            >
              ~95% of parameters get quantized to ternary {"\u2192"} massive
              compression
            </div>
          </FadeIn>
        )}
      </div>
    </div>
  );
}
