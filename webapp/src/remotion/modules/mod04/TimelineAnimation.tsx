import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle } from "../../utils";

const EVENTS = [
  { year: 2015, label: "BinaryConnect", desc: "First binary weight networks", color: COLORS.blue },
  { year: 2016, label: "XNOR-Net", desc: "Binary weights + activations", color: COLORS.blue },
  { year: 2017, label: "TWN", desc: "Ternary Weight Networks", color: COLORS.purple },
  { year: 2020, label: "ReActNet", desc: "Improved binary accuracy", color: COLORS.amber },
  { year: 2023, label: "BitNet", desc: "1-bit weights for LLMs", color: COLORS.accent },
  { year: 2024, label: "BitNet b1.58", desc: "Ternary {-1,0,+1} LLMs", color: COLORS.accent },
  { year: 2025, label: "BitNet 2B4T", desc: "2B params, native ternary", color: COLORS.accentLight },
];

export default function TimelineAnimation() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const lineY = 280;
  const lineLeft = 60;
  const lineRight = 900;

  function yearToX(year: number) {
    return lineLeft + ((year - 2014) / 12) * (lineRight - lineLeft);
  }

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: COLORS.bg,
        padding: 40,
        fontFamily: FONTS.sans,
        position: "relative",
      }}
    >
      <FadeIn>
        <Title>Binary Neural Networks Timeline</Title>
        <Subtitle style={{ marginTop: 4 }}>
          A decade of progress from BinaryConnect to BitNet
        </Subtitle>
      </FadeIn>

      <svg width={960} height={400} style={{ position: "absolute", top: 80, left: 0 }}>
        {/* Timeline line */}
        <FadeIn delay={15}>
          <line
            x1={lineLeft}
            y1={lineY}
            x2={lineRight}
            y2={lineY}
            stroke={COLORS.border}
            strokeWidth={2}
          />
        </FadeIn>

        {/* Events */}
        {EVENTS.map((event, i) => {
          const x = yearToX(event.year);
          const above = i % 2 === 0;
          const eventDelay = 30 + i * 20;
          const p = spring({ frame: frame - eventDelay, fps, config: { damping: 12 } });
          const opacity = interpolate(p, [0, 1], [0, 1]);
          const yOffset = above ? -1 : 1;

          return (
            <g key={i} opacity={opacity}>
              {/* Dot on timeline */}
              <circle
                cx={x}
                cy={lineY}
                r={interpolate(p, [0, 1], [0, 6])}
                fill={event.color}
              />

              {/* Connector line */}
              <line
                x1={x}
                y1={lineY + yOffset * 10}
                x2={x}
                y2={lineY + yOffset * 60}
                stroke={event.color}
                strokeWidth={1}
                strokeDasharray="3,3"
                opacity={0.6}
              />

              {/* Year */}
              <text
                x={x}
                y={lineY + 20}
                fill={COLORS.textDim}
                fontSize={11}
                textAnchor="middle"
                fontFamily={FONTS.mono}
              >
                {event.year}
              </text>

              {/* Label card */}
              <foreignObject
                x={x - 60}
                y={above ? lineY - 130 : lineY + 30}
                width={120}
                height={65}
              >
                <div
                  style={{
                    background: COLORS.surface,
                    border: `1px solid ${event.color}40`,
                    borderRadius: 8,
                    padding: "8px 10px",
                    textAlign: "center",
                  }}
                >
                  <div
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: event.color,
                      marginBottom: 2,
                    }}
                  >
                    {event.label}
                  </div>
                  <div style={{ fontSize: 10, color: COLORS.textDim, lineHeight: 1.3 }}>
                    {event.desc}
                  </div>
                </div>
              </foreignObject>
            </g>
          );
        })}

        {/* Progress indicator */}
        {(() => {
          const sweepProgress = interpolate(frame, [30, 200], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const sweepX = lineLeft + sweepProgress * (lineRight - lineLeft);
          return (
            <line
              x1={lineLeft}
              y1={lineY}
              x2={sweepX}
              y2={lineY}
              stroke={COLORS.accent}
              strokeWidth={3}
              opacity={0.6}
            />
          );
        })()}
      </svg>

      {frame > 250 && (
        <FadeIn delay={250}>
          <div
            style={{
              position: "absolute",
              bottom: 30,
              left: 0,
              right: 0,
              textAlign: "center",
              fontSize: 14,
              color: COLORS.accent,
              fontWeight: 600,
            }}
          >
            From vision models to LLMs — 1-bit networks are now practical at scale
          </div>
        </FadeIn>
      )}
    </div>
  );
}
