import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle } from "../../utils";

const DATA_FP16 = [
  { params: 0.125, ppl: 32 },
  { params: 0.35, ppl: 22 },
  { params: 1.3, ppl: 14.5 },
  { params: 3, ppl: 10.5 },
  { params: 7, ppl: 8.2 },
  { params: 13, ppl: 7.0 },
];

const DATA_BITNET = [
  { params: 0.125, ppl: 45 },
  { params: 0.35, ppl: 28 },
  { params: 1.3, ppl: 16 },
  { params: 3, ppl: 10.8 },
  { params: 7, ppl: 8.0 },
  { params: 13, ppl: 6.8 },
];

export default function ScalingCurve() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const chartLeft = 120;
  const chartRight = 860;
  const chartTop = 120;
  const chartBottom = 420;
  const chartWidth = chartRight - chartLeft;
  const chartHeight = chartBottom - chartTop;

  function paramToX(params: number) {
    const logMin = Math.log(0.1);
    const logMax = Math.log(15);
    return chartLeft + ((Math.log(params) - logMin) / (logMax - logMin)) * chartWidth;
  }

  function pplToY(ppl: number) {
    const minPpl = 5;
    const maxPpl = 50;
    return chartTop + ((maxPpl - ppl) / (maxPpl - minPpl)) * chartHeight;
  }

  const fp16Delay = 40;
  const bitnetDelay = 120;

  function drawLine(data: typeof DATA_FP16, delay: number, color: string) {
    const p = spring({ frame: frame - delay, fps, config: { damping: 20, mass: 1.5 } });
    const drawProgress = interpolate(p, [0, 1], [0, data.length - 1]);

    const points = data.map((d) => `${paramToX(d.params)},${pplToY(d.ppl)}`).join(" ");

    return (
      <g opacity={interpolate(frame - delay, [0, 15], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })}>
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth={2.5}
          strokeDasharray={`${drawProgress * 200}, 2000`}
        />
        {data.map((d, i) => {
          const dotP = spring({ frame: frame - delay - 10 - i * 8, fps, config: { damping: 12 } });
          return (
            <circle
              key={i}
              cx={paramToX(d.params)}
              cy={pplToY(d.ppl)}
              r={interpolate(dotP, [0, 1], [0, 5])}
              fill={color}
            />
          );
        })}
      </g>
    );
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
        <Title>Scaling Curves: BitNet vs FP16</Title>
        <Subtitle style={{ marginTop: 4 }}>
          BitNet matches FP16 perplexity at ~3B+ parameters
        </Subtitle>
      </FadeIn>

      <svg width={960} height={460} style={{ position: "absolute", top: 60, left: 0 }}>
        {/* Grid */}
        {[10, 20, 30, 40].map((ppl) => (
          <g key={ppl}>
            <line
              x1={chartLeft}
              y1={pplToY(ppl)}
              x2={chartRight}
              y2={pplToY(ppl)}
              stroke={COLORS.border}
              strokeWidth={0.5}
              strokeDasharray="4,4"
            />
            <text x={chartLeft - 12} y={pplToY(ppl) + 4} fill={COLORS.textDim} fontSize={11} textAnchor="end" fontFamily={FONTS.mono}>
              {ppl}
            </text>
          </g>
        ))}

        {/* Axes */}
        <line x1={chartLeft} y1={chartTop} x2={chartLeft} y2={chartBottom} stroke={COLORS.border} strokeWidth={1.5} />
        <line x1={chartLeft} y1={chartBottom} x2={chartRight} y2={chartBottom} stroke={COLORS.border} strokeWidth={1.5} />

        {/* X axis labels */}
        {[0.1, 0.5, 1, 3, 7, 13].map((p) => (
          <text key={p} x={paramToX(p)} y={chartBottom + 20} fill={COLORS.textDim} fontSize={11} textAnchor="middle" fontFamily={FONTS.mono}>
            {p >= 1 ? `${p}B` : `${p * 1000}M`}
          </text>
        ))}

        {/* Y axis label */}
        <text x={30} y={chartTop + chartHeight / 2} fill={COLORS.textDim} fontSize={12} textAnchor="middle" transform={`rotate(-90, 30, ${chartTop + chartHeight / 2})`}>
          Perplexity ↓
        </text>

        {/* X axis label */}
        <text x={chartLeft + chartWidth / 2} y={chartBottom + 40} fill={COLORS.textDim} fontSize={12} textAnchor="middle">
          Parameters
        </text>

        {/* Lines */}
        {drawLine(DATA_FP16, fp16Delay, COLORS.blue)}
        {drawLine(DATA_BITNET, bitnetDelay, COLORS.accent)}

        {/* Crossover region */}
        {frame > 200 && (() => {
          const p = spring({ frame: frame - 200, fps, config: { damping: 15 } });
          const crossX = paramToX(3);
          return (
            <g opacity={interpolate(p, [0, 1], [0, 0.6])}>
              <rect
                x={crossX - 20}
                y={chartTop}
                width={chartRight - crossX + 20}
                height={chartHeight}
                fill={`${COLORS.accent}08`}
                stroke={COLORS.accent}
                strokeWidth={1}
                strokeDasharray="4,4"
              />
              <text x={crossX + 60} y={chartTop + 20} fill={COLORS.accent} fontSize={11} fontWeight={600}>
                BitNet matches or beats FP16
              </text>
            </g>
          );
        })()}

        {/* Legend */}
        <g>
          <circle cx={chartRight - 180} cy={chartTop + 10} r={5} fill={COLORS.blue} />
          <text x={chartRight - 170} y={chartTop + 14} fill={COLORS.blue} fontSize={12}>FP16 baseline</text>
          <circle cx={chartRight - 180} cy={chartTop + 30} r={5} fill={COLORS.accent} />
          <text x={chartRight - 170} y={chartTop + 34} fill={COLORS.accent} fontSize={12}>BitNet b1.58</text>
        </g>
      </svg>
    </div>
  );
}
