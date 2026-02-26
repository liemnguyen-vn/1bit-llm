import { COLORS, FONTS } from "../../theme";
import { FadeIn, AnimatedBar, Title, Subtitle } from "../../utils";

const DATA = [
  { label: "FP32", bits: 32, color: COLORS.red },
  { label: "FP16", bits: 16, color: COLORS.amber },
  { label: "INT8", bits: 8, color: COLORS.blue },
  { label: "INT4", bits: 4, color: COLORS.purple },
  { label: "Ternary", bits: 1.58, color: COLORS.accent },
];

export default function MemoryComparison() {
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
        <Title>Memory per Parameter</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Ternary weights achieve ~10x reduction over FP16
        </Subtitle>
      </FadeIn>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 12,
          marginTop: 20,
        }}
      >
        {DATA.map((d, i) => (
          <AnimatedBar
            key={d.label}
            label={d.label}
            value={d.bits}
            maxValue={32}
            color={d.color}
            delay={20 + i * 20}
            width={500}
            height={40}
            showValue={`${d.bits} bits`}
          />
        ))}
      </div>

      <FadeIn delay={140}>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 40,
            marginTop: 8,
          }}
        >
          {[
            { label: "7B FP16 model", val: "14 GB", color: COLORS.amber },
            { label: "7B Ternary model", val: "~1.4 GB", color: COLORS.accent },
          ].map((item) => (
            <div
              key={item.label}
              style={{
                background: COLORS.surface,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 12,
                padding: "16px 28px",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 13, color: COLORS.textDim, marginBottom: 4 }}>
                {item.label}
              </div>
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  color: item.color,
                  fontFamily: FONTS.mono,
                }}
              >
                {item.val}
              </div>
            </div>
          ))}
        </div>
      </FadeIn>
    </div>
  );
}
