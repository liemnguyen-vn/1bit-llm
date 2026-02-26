import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle } from "../../utils";

const STAGES = [
  { label: "Input\nActivations", short: "x", color: COLORS.blue },
  { label: "RMS\nNorm", short: "LN", color: COLORS.purple },
  { label: "Activation\nQuant (INT8)", short: "Qₐ", color: COLORS.amber },
  { label: "BitLinear\nMatMul", short: "×", color: COLORS.accent },
  { label: "De-\nquantize", short: "DQ", color: COLORS.cyan },
  { label: "Output", short: "y", color: COLORS.accentLight },
];

const WEIGHT_STAGES = [
  { label: "FP Weights", short: "W", color: COLORS.textMuted },
  { label: "Weight\nQuant", short: "Qw", color: COLORS.amber },
  { label: "Ternary\nWeights", short: "Ŵ", color: COLORS.accent },
];

export default function BitLinearLayerFlow() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const stageWidth = 110;
  const stageGap = 16;
  const mainY = 200;
  const weightY = 370;

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
        <Title>BitLinear Layer Pipeline</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Input normalization → quantization → ternary matrix multiply → dequantization
        </Subtitle>
      </FadeIn>

      <svg width={960} height={430} style={{ position: "absolute", top: 80, left: 0 }}>
        {/* Main pipeline */}
        {STAGES.map((stage, i) => {
          const x = 80 + i * (stageWidth + stageGap);
          const stageDelay = 20 + i * 15;
          const p = spring({ frame: frame - stageDelay, fps, config: { damping: 15 } });
          const opacity = interpolate(p, [0, 1], [0, 1]);

          // Packet animation
          const packetStart = 90;
          const packetSpeed = 25;
          const packetPos = (frame - packetStart) / packetSpeed;
          const isActive = packetPos >= i && packetPos < i + 1;

          return (
            <g key={i} opacity={opacity}>
              {/* Box */}
              <rect
                x={x}
                y={mainY - 35}
                width={stageWidth}
                height={70}
                rx={10}
                fill={isActive ? `${stage.color}25` : COLORS.surface}
                stroke={isActive ? stage.color : COLORS.border}
                strokeWidth={isActive ? 2 : 1}
              />
              {/* Label */}
              <foreignObject x={x} y={mainY - 30} width={stageWidth} height={60}>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      color: isActive ? stage.color : COLORS.text,
                      textAlign: "center",
                      whiteSpace: "pre-line",
                      lineHeight: 1.3,
                      fontWeight: 600,
                    }}
                  >
                    {stage.label}
                  </div>
                </div>
              </foreignObject>

              {/* Arrow to next */}
              {i < STAGES.length - 1 && (
                <g>
                  <line
                    x1={x + stageWidth}
                    y1={mainY}
                    x2={x + stageWidth + stageGap - 4}
                    y2={mainY}
                    stroke={COLORS.border}
                    strokeWidth={1.5}
                  />
                  <polygon
                    points={`${x + stageWidth + stageGap - 8},${mainY - 4} ${x + stageWidth + stageGap},${mainY} ${x + stageWidth + stageGap - 8},${mainY + 4}`}
                    fill={COLORS.border}
                  />
                </g>
              )}
            </g>
          );
        })}

        {/* Weight quantization branch (bottom) */}
        {WEIGHT_STAGES.map((ws, i) => {
          const x = 240 + i * (stageWidth + stageGap);
          const wsDelay = 100 + i * 20;
          const p = spring({ frame: frame - wsDelay, fps, config: { damping: 15 } });
          const opacity = interpolate(p, [0, 1], [0, 1]);

          return (
            <g key={`w-${i}`} opacity={opacity}>
              <rect
                x={x}
                y={weightY - 25}
                width={stageWidth}
                height={50}
                rx={8}
                fill={COLORS.surface}
                stroke={`${ws.color}60`}
                strokeWidth={1}
                strokeDasharray={i === 0 ? "4,4" : "0"}
              />
              <foreignObject x={x} y={weightY - 20} width={stageWidth} height={40}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    height: "100%",
                    fontSize: 11,
                    color: ws.color,
                    fontWeight: 600,
                    textAlign: "center",
                    whiteSpace: "pre-line",
                    lineHeight: 1.3,
                  }}
                >
                  {ws.label}
                </div>
              </foreignObject>

              {i < WEIGHT_STAGES.length - 1 && (
                <line
                  x1={x + stageWidth}
                  y1={weightY}
                  x2={x + stageWidth + stageGap}
                  y2={weightY}
                  stroke={COLORS.border}
                  strokeWidth={1}
                />
              )}
            </g>
          );
        })}

        {/* Arrow from ternary weights up to MatMul */}
        {frame > 160 && (() => {
          const p = spring({ frame: frame - 160, fps, config: { damping: 15 } });
          const matMulX = 80 + 3 * (stageWidth + stageGap) + stageWidth / 2;
          const ternaryX = 240 + 2 * (stageWidth + stageGap) + stageWidth / 2;
          return (
            <g opacity={interpolate(p, [0, 1], [0, 1])}>
              <line
                x1={ternaryX}
                y1={weightY - 25}
                x2={matMulX}
                y2={mainY + 35}
                stroke={COLORS.accent}
                strokeWidth={1.5}
                strokeDasharray="4,4"
              />
              <polygon
                points={`${matMulX - 4},${mainY + 39} ${matMulX},${mainY + 35} ${matMulX + 4},${mainY + 39}`}
                fill={COLORS.accent}
              />
            </g>
          );
        })()}

        {/* Label */}
        {frame > 180 && (
          <foreignObject x={0} y={430} width={960} height={30}>
            <FadeIn delay={180}>
              <div style={{ textAlign: "center", fontSize: 13, color: COLORS.accent, fontWeight: 600 }}>
                Ternary MatMul uses only additions — no floating-point multiplications needed
              </div>
            </FadeIn>
          </foreignObject>
        )}
      </svg>
    </div>
  );
}
