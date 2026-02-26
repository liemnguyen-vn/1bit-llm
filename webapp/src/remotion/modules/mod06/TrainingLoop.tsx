import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle } from "../../utils";

const STEPS = [
  { label: "Shadow\nWeights (FP)", color: COLORS.blue, icon: "W" },
  { label: "Quantize\nto Ternary", color: COLORS.amber, icon: "Q" },
  { label: "Forward\nPass", color: COLORS.accent, icon: "→" },
  { label: "Compute\nLoss", color: COLORS.red, icon: "L" },
  { label: "Backward\n(STE)", color: COLORS.purple, icon: "∇" },
  { label: "Update\nShadow W", color: COLORS.cyan, icon: "↻" },
];

export default function TrainingLoop() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const centerX = 480;
  const centerY = 280;
  const radius = 160;

  const cycleSpeed = 60; // frames per step
  const activeStep = Math.floor((frame - 60) / cycleSpeed) % STEPS.length;
  const started = frame > 60;

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
        <Title>Training Loop with Quantized Weights</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Shadow weights (FP) are quantized each forward pass, updated via STE in backward pass
        </Subtitle>
      </FadeIn>

      <svg width={960} height={440} style={{ position: "absolute", top: 80, left: 0 }}>
        {/* Circular path */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius}
          fill="none"
          stroke={COLORS.border}
          strokeWidth={1}
          strokeDasharray="6,6"
        />

        {/* Steps around the circle */}
        {STEPS.map((step, i) => {
          const angle = (i / STEPS.length) * Math.PI * 2 - Math.PI / 2;
          const x = centerX + Math.cos(angle) * radius;
          const y = centerY + Math.sin(angle) * radius;

          const stepDelay = 15 + i * 10;
          const p = spring({ frame: frame - stepDelay, fps, config: { damping: 15 } });
          const opacity = interpolate(p, [0, 1], [0, 1]);

          const isActive = started && activeStep === i;
          const isPast = started && (frame - 60) / cycleSpeed >= i;

          return (
            <g key={i} opacity={opacity}>
              {/* Node */}
              <circle
                cx={x}
                cy={y}
                r={36}
                fill={isActive ? `${step.color}30` : COLORS.surface}
                stroke={isActive ? step.color : isPast ? `${step.color}60` : COLORS.border}
                strokeWidth={isActive ? 2.5 : 1.5}
              />

              {/* Icon */}
              <text
                x={x}
                y={y - 4}
                fill={isActive ? step.color : COLORS.text}
                fontSize={18}
                fontWeight={700}
                textAnchor="middle"
                fontFamily={FONTS.mono}
              >
                {step.icon}
              </text>

              {/* Label */}
              <foreignObject
                x={x - 50}
                y={y + 38}
                width={100}
                height={40}
              >
                <div
                  style={{
                    textAlign: "center",
                    fontSize: 10,
                    color: isActive ? step.color : COLORS.textMuted,
                    fontWeight: isActive ? 700 : 400,
                    whiteSpace: "pre-line",
                    lineHeight: 1.3,
                  }}
                >
                  {step.label}
                </div>
              </foreignObject>

              {/* Arrow to next */}
              {(() => {
                const nextAngle = ((i + 1) / STEPS.length) * Math.PI * 2 - Math.PI / 2;
                const midAngle = angle + (nextAngle - angle) * 0.5 + (i === STEPS.length - 1 ? Math.PI : 0);
                const ax = centerX + Math.cos(midAngle) * (radius + 6);
                const ay = centerY + Math.sin(midAngle) * (radius + 6);
                return (
                  <text
                    x={ax}
                    y={ay}
                    fill={COLORS.textDim}
                    fontSize={12}
                    textAnchor="middle"
                    dominantBaseline="middle"
                  >
                    ›
                  </text>
                );
              })()}
            </g>
          );
        })}

        {/* Active step indicator - glowing dot */}
        {started && (() => {
          const angle = (activeStep / STEPS.length) * Math.PI * 2 - Math.PI / 2;
          const x = centerX + Math.cos(angle) * radius;
          const y = centerY + Math.sin(angle) * radius;
          const pulseP = spring({
            frame: (frame - 60) % cycleSpeed,
            fps,
            config: { damping: 8 },
          });
          return (
            <circle
              cx={x}
              cy={y}
              r={40 + interpolate(pulseP, [0, 1], [0, 4])}
              fill="none"
              stroke={STEPS[activeStep].color}
              strokeWidth={1}
              opacity={0.4}
            />
          );
        })()}

        {/* Center label */}
        <foreignObject x={centerX - 60} y={centerY - 25} width={120} height={50}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 13, color: COLORS.textDim, fontWeight: 600 }}>
              Training<br />Loop
            </div>
          </div>
        </foreignObject>
      </svg>
    </div>
  );
}
