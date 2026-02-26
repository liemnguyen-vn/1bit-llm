import { useCurrentFrame, useVideoConfig, spring, interpolate } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle, Box } from "../../utils";

const LINEAR_CODE = [
  "class nn.Linear:",
  "  def forward(self, x):",
  "    # FP16 matrix multiply",
  "    return x @ self.weight.T + self.bias",
  "",
  "  # Weight: float16 tensor",
  "  # Ops: FP16 multiply-accumulate",
  "  # Memory: 16 bits/param",
];

const BITLINEAR_CODE = [
  "class BitLinear:",
  "  def forward(self, x):",
  "    x_norm = rms_norm(x)",
  "    x_q = activation_quant(x_norm)  # INT8",
  "    w_q = weight_quant(self.weight)  # Ternary",
  "    y = x_q @ w_q.T  # Add/sub only!",
  "    return dequantize(y, scales)",
  "",
  "  # Weight: ternary {-1,0,+1}",
  "  # Ops: additions + sign flips",
  "  # Memory: 1.58 bits/param",
];

export default function BitLinearVsLinear() {
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
        <Title>nn.Linear vs BitLinear</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Side-by-side comparison of standard and 1-bit linear layers
        </Subtitle>
      </FadeIn>

      <div style={{ display: "flex", gap: 24, marginTop: 24, flex: 1 }}>
        {/* Standard Linear */}
        <SlideIn delay={15} from="left" style={{ flex: 1 }}>
          <Box style={{ height: "100%", borderColor: `${COLORS.blue}40` }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.blue, marginBottom: 12 }}>
              nn.Linear (Standard)
            </div>
            <div
              style={{
                background: COLORS.bg,
                borderRadius: 8,
                padding: 12,
                fontFamily: FONTS.mono,
                fontSize: 11,
                lineHeight: 1.8,
              }}
            >
              {LINEAR_CODE.map((line, i) => {
                const lineDelay = 20 + i * 6;
                const p = spring({ frame: frame - lineDelay, fps, config: { damping: 15 } });
                const isComment = line.trim().startsWith("#");
                return (
                  <div
                    key={i}
                    style={{
                      opacity: interpolate(p, [0, 1], [0, 1]),
                      color: isComment ? COLORS.textDim : line.includes("class") ? COLORS.blue : COLORS.text,
                    }}
                  >
                    {line || "\u00A0"}
                  </div>
                );
              })}
            </div>
          </Box>
        </SlideIn>

        {/* BitLinear */}
        <SlideIn delay={60} from="right" style={{ flex: 1 }}>
          <Box style={{ height: "100%", borderColor: `${COLORS.accent}40` }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: COLORS.accent, marginBottom: 12 }}>
              BitLinear (1-bit)
            </div>
            <div
              style={{
                background: COLORS.bg,
                borderRadius: 8,
                padding: 12,
                fontFamily: FONTS.mono,
                fontSize: 11,
                lineHeight: 1.8,
              }}
            >
              {BITLINEAR_CODE.map((line, i) => {
                const lineDelay = 65 + i * 6;
                const p = spring({ frame: frame - lineDelay, fps, config: { damping: 15 } });
                const isComment = line.trim().startsWith("#");
                const isHighlight = line.includes("ternary") || line.includes("Add/sub") || line.includes("1.58");
                return (
                  <div
                    key={i}
                    style={{
                      opacity: interpolate(p, [0, 1], [0, 1]),
                      color: isComment
                        ? isHighlight
                          ? COLORS.accent
                          : COLORS.textDim
                        : line.includes("class")
                          ? COLORS.accent
                          : COLORS.text,
                    }}
                  >
                    {line || "\u00A0"}
                  </div>
                );
              })}
            </div>
          </Box>
        </SlideIn>
      </div>

      {frame > 180 && (
        <FadeIn delay={180}>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: 32,
              marginTop: 12,
            }}
          >
            {[
              { label: "Memory", standard: "16 bits", bit: "1.58 bits", improvement: "~10x" },
              { label: "Compute", standard: "FP16 MACs", bit: "INT add/sub", improvement: "Much faster" },
            ].map((item) => (
              <div
                key={item.label}
                style={{
                  background: COLORS.surface,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 8,
                  padding: "8px 20px",
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                  fontSize: 12,
                }}
              >
                <span style={{ color: COLORS.textDim, fontWeight: 600 }}>{item.label}:</span>
                <span style={{ color: COLORS.blue }}>{item.standard}</span>
                <span style={{ color: COLORS.textDim }}>→</span>
                <span style={{ color: COLORS.accent, fontWeight: 700 }}>{item.bit}</span>
                <span
                  style={{
                    background: `${COLORS.accent}20`,
                    color: COLORS.accent,
                    padding: "2px 8px",
                    borderRadius: 4,
                    fontWeight: 600,
                    fontSize: 11,
                  }}
                >
                  {item.improvement}
                </span>
              </div>
            ))}
          </div>
        </FadeIn>
      )}
    </div>
  );
}
