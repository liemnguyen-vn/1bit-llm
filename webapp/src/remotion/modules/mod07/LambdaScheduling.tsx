import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle } from "../../utils";

export default function LambdaScheduling() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const chartLeft = 120;
  const chartRight = 840;
  const chartTop = 140;
  const chartBottom = 380;
  const chartWidth = chartRight - chartLeft;
  const chartHeight = chartBottom - chartTop;

  // Lambda goes from 0 to 1
  const drawProgress = interpolate(frame, [40, 200], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // S-shaped curve
  function lambdaAt(t: number) {
    return 1 / (1 + Math.exp(-10 * (t - 0.5)));
  }

  const points: string[] = [];
  const numPoints = 100;
  for (let i = 0; i <= numPoints * drawProgress; i++) {
    const t = i / numPoints;
    const x = chartLeft + t * chartWidth;
    const y = chartBottom - lambdaAt(t) * chartHeight;
    points.push(`${x},${y}`);
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
        <Title>Lambda Scheduling for Fine-Tuning</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Gradually transition from full-precision to ternary during training
        </Subtitle>
      </FadeIn>

      <svg width={960} height={440} style={{ position: "absolute", top: 70, left: 0 }}>
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((val) => {
          const y = chartBottom - val * chartHeight;
          return (
            <g key={val}>
              <line x1={chartLeft} y1={y} x2={chartRight} y2={y} stroke={COLORS.border} strokeWidth={0.5} strokeDasharray="4,4" />
              <text x={chartLeft - 12} y={y + 4} fill={COLORS.textDim} fontSize={11} textAnchor="end" fontFamily={FONTS.mono}>
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Axes */}
        <line x1={chartLeft} y1={chartTop} x2={chartLeft} y2={chartBottom} stroke={COLORS.border} strokeWidth={1.5} />
        <line x1={chartLeft} y1={chartBottom} x2={chartRight} y2={chartBottom} stroke={COLORS.border} strokeWidth={1.5} />

        {/* Y axis label */}
        <text x={35} y={chartTop + chartHeight / 2} fill={COLORS.textDim} fontSize={12} textAnchor="middle" transform={`rotate(-90, 35, ${chartTop + chartHeight / 2})`}>
          λ (quantization ratio)
        </text>

        {/* X axis labels */}
        <text x={chartLeft} y={chartBottom + 20} fill={COLORS.textDim} fontSize={11} textAnchor="middle">Start</text>
        <text x={chartRight} y={chartBottom + 20} fill={COLORS.textDim} fontSize={11} textAnchor="middle">End</text>
        <text x={chartLeft + chartWidth / 2} y={chartBottom + 36} fill={COLORS.textDim} fontSize={12} textAnchor="middle">
          Training Progress
        </text>

        {/* Curve */}
        {points.length > 1 && (
          <polyline
            points={points.join(" ")}
            fill="none"
            stroke={COLORS.accent}
            strokeWidth={3}
          />
        )}

        {/* Moving dot */}
        {points.length > 0 && (() => {
          const lastPoint = points[points.length - 1].split(",");
          return (
            <circle
              cx={parseFloat(lastPoint[0])}
              cy={parseFloat(lastPoint[1])}
              r={6}
              fill={COLORS.accent}
            />
          );
        })()}

        {/* Regions */}
        {frame > 80 && (
          <FadeIn delay={80}>
            <foreignObject x={chartLeft + 10} y={chartBottom - 50} width={160} height={40}>
              <div style={{ fontSize: 12, color: COLORS.blue, fontWeight: 600 }}>
                λ ≈ 0: Mostly FP weights
              </div>
            </foreignObject>
          </FadeIn>
        )}
        {frame > 160 && (
          <FadeIn delay={160}>
            <foreignObject x={chartRight - 190} y={chartTop + 10} width={180} height={40}>
              <div style={{ fontSize: 12, color: COLORS.accent, fontWeight: 600 }}>
                λ ≈ 1: Fully ternary weights
              </div>
            </foreignObject>
          </FadeIn>
        )}

        {/* Formula */}
        {frame > 120 && (
          <FadeIn delay={120}>
            <foreignObject x={chartLeft + chartWidth / 2 - 120} y={chartTop + chartHeight / 2 - 30} width={240} height={50}>
              <div
                style={{
                  background: `${COLORS.surface}ee`,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 8,
                  padding: "8px 16px",
                  textAlign: "center",
                  fontFamily: FONTS.mono,
                  fontSize: 13,
                  color: COLORS.text,
                }}
              >
                W = (1 - λ)·W_fp + λ·Q(W)
              </div>
            </foreignObject>
          </FadeIn>
        )}
      </svg>

      {frame > 240 && (
        <FadeIn delay={240}>
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
            Gradual transition prevents training instability from sudden quantization
          </div>
        </FadeIn>
      )}
    </div>
  );
}
