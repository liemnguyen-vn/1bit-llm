import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle } from "../../utils";

const TERNARY_WEIGHTS = [1, -1, 0, 1, -1, -1, 0, 1];
const ENCODED = ["01", "10", "00", "01", "10", "10", "00", "01"];
const PACKED_BYTE = "01 10 00 01";
const PACKED_BYTE2 = "10 10 00 01";

export default function WeightPacking() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const phase1 = 0;   // Show ternary weights
  const phase2 = 60;  // Show 2-bit encoding
  const phase3 = 160; // Show byte packing

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
        <Title>Weight Packing: Ternary to 2-Bit Encoding</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Pack 4 ternary weights into a single byte for efficient storage
        </Subtitle>
      </FadeIn>

      {/* Step 1: Ternary weights */}
      <SlideIn delay={15} from="left" style={{ marginTop: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ width: 120, fontSize: 13, color: COLORS.textDim, textAlign: "right" }}>
            Ternary weights:
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {TERNARY_WEIGHTS.map((w, i) => {
              const cellDelay = 20 + i * 5;
              const p = spring({ frame: frame - cellDelay, fps, config: { damping: 12 } });
              const color = w === 1 ? COLORS.accent : w === -1 ? COLORS.red : COLORS.textDim;
              return (
                <div
                  key={i}
                  style={{
                    width: 52,
                    height: 44,
                    borderRadius: 8,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 16,
                    fontWeight: 700,
                    fontFamily: FONTS.mono,
                    background: `${color}20`,
                    border: `1px solid ${color}40`,
                    color,
                    opacity: interpolate(p, [0, 1], [0, 1]),
                    transform: `scale(${interpolate(p, [0, 1], [0.5, 1])})`,
                  }}
                >
                  {w > 0 ? `+${w}` : w}
                </div>
              );
            })}
          </div>
        </div>
      </SlideIn>

      {/* Step 2: Encoding map */}
      {frame > phase2 && (
        <FadeIn delay={phase2}>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginTop: 20 }}>
            <div style={{ width: 120, fontSize: 13, color: COLORS.textDim, textAlign: "right" }}>
              2-bit encoding:
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              {ENCODED.map((enc, i) => {
                const cellDelay = phase2 + 10 + i * 5;
                const p = spring({ frame: frame - cellDelay, fps, config: { damping: 12 } });
                const w = TERNARY_WEIGHTS[i];
                const color = w === 1 ? COLORS.accent : w === -1 ? COLORS.red : COLORS.textDim;
                return (
                  <div
                    key={i}
                    style={{
                      width: 52,
                      height: 44,
                      borderRadius: 8,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 16,
                      fontWeight: 700,
                      fontFamily: FONTS.mono,
                      background: COLORS.surface,
                      border: `1px solid ${color}40`,
                      color: COLORS.text,
                      opacity: interpolate(p, [0, 1], [0, 1]),
                      transform: `scale(${interpolate(p, [0, 1], [0.5, 1])})`,
                    }}
                  >
                    {enc}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Encoding key */}
          <div style={{ display: "flex", gap: 16, marginLeft: 136, marginTop: 12 }}>
            {[
              { val: "+1", code: "01", color: COLORS.accent },
              { val: " 0", code: "00", color: COLORS.textDim },
              { val: "-1", code: "10", color: COLORS.red },
            ].map((item) => (
              <div key={item.val} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                <span style={{ color: item.color, fontFamily: FONTS.mono, fontWeight: 700 }}>{item.val}</span>
                <span style={{ color: COLORS.textDim }}>→</span>
                <span style={{ color: COLORS.text, fontFamily: FONTS.mono }}>{item.code}</span>
              </div>
            ))}
          </div>
        </FadeIn>
      )}

      {/* Step 3: Byte packing */}
      {frame > phase3 && (
        <SlideIn delay={phase3} from="bottom" style={{ marginTop: 24 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{ width: 120, fontSize: 13, color: COLORS.textDim, textAlign: "right" }}>
              Packed bytes:
            </div>
            <div style={{ display: "flex", gap: 16 }}>
              {[PACKED_BYTE, PACKED_BYTE2].map((byte, i) => (
                <div
                  key={i}
                  style={{
                    background: `${COLORS.accent}15`,
                    border: `1px solid ${COLORS.accent}40`,
                    borderRadius: 8,
                    padding: "10px 16px",
                    fontFamily: FONTS.mono,
                    fontSize: 16,
                    fontWeight: 700,
                    color: COLORS.accent,
                    letterSpacing: 2,
                  }}
                >
                  {byte}
                </div>
              ))}
              <div style={{ display: "flex", alignItems: "center", fontSize: 12, color: COLORS.textDim }}>
                ← 8 weights in 2 bytes!
              </div>
            </div>
          </div>
        </SlideIn>
      )}

      {/* Summary */}
      {frame > 220 && (
        <FadeIn delay={220}>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: 32,
              marginTop: 28,
            }}
          >
            {[
              { label: "Naive storage", val: "8 × 32 = 256 bits", color: COLORS.red },
              { label: "2-bit packed", val: "8 × 2 = 16 bits", color: COLORS.accent },
              { label: "Compression", val: "16× smaller", color: COLORS.accentLight },
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
                <div style={{ fontSize: 16, fontWeight: 700, color: item.color, fontFamily: FONTS.mono }}>
                  {item.val}
                </div>
              </div>
            ))}
          </div>
        </FadeIn>
      )}
    </div>
  );
}
