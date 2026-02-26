import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle } from "../../utils";

const ORIGINAL_VALUES = [-2.3, -1.1, -0.4, 0.2, 0.7, 1.5, 2.8];
const GRID_POINTS_8 = [-3, -2, -1, 0, 1, 2, 3];
const GRID_POINTS_TERNARY = [-1, 0, 1];

export default function QuantizationVisualization() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const phase1 = 0;    // Show number line + original
  const phase2 = 60;   // INT8 snap
  const phase3 = 180;  // Ternary snap

  const lineY = 200;
  const lineLeft = 100;
  const lineRight = 860;
  const lineWidth = lineRight - lineLeft;

  function valToX(val: number) {
    return lineLeft + ((val + 3.5) / 7) * lineWidth;
  }

  function snapTo(val: number, grid: number[]) {
    let best = grid[0];
    for (const g of grid) {
      if (Math.abs(val - g) < Math.abs(val - best)) best = g;
    }
    return best;
  }

  const showTernary = frame > phase3;

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
        <Title>Quantization: Snapping to Grid Points</Title>
        <Subtitle style={{ marginTop: 4 }}>
          {showTernary
            ? "Ternary: extreme quantization to just {-1, 0, +1}"
            : "Reducing precision by mapping values to nearest grid point"}
        </Subtitle>
      </FadeIn>

      <svg width={960} height={340} style={{ marginTop: 20 }}>
        {/* Number line */}
        <line x1={lineLeft} y1={lineY} x2={lineRight} y2={lineY} stroke={COLORS.border} strokeWidth={2} />

        {/* Grid points */}
        {(showTernary ? GRID_POINTS_TERNARY : GRID_POINTS_8).map((gp) => {
          const x = valToX(gp);
          const gridDelay = showTernary ? phase3 + 10 : 20;
          const p = spring({ frame: frame - gridDelay, fps, config: { damping: 12 } });
          return (
            <g key={`grid-${gp}`} opacity={interpolate(p, [0, 1], [0, 1])}>
              <line x1={x} y1={lineY - 10} x2={x} y2={lineY + 10} stroke={COLORS.textDim} strokeWidth={2} />
              <text x={x} y={lineY + 28} fill={COLORS.textDim} fontSize={12} textAnchor="middle" fontFamily={FONTS.mono}>
                {gp}
              </text>
            </g>
          );
        })}

        {/* Original values + snapping */}
        {ORIGINAL_VALUES.map((val, i) => {
          const origX = valToX(val);
          const grid = showTernary ? GRID_POINTS_TERNARY : GRID_POINTS_8;
          const snapped = snapTo(val, grid);
          const snappedX = valToX(snapped);

          const snapPhase = showTernary ? phase3 : phase2;
          const dotDelay = 30 + i * 8;
          const dotP = spring({ frame: frame - dotDelay, fps, config: { damping: 12 } });
          const snapDelay = snapPhase + 10 + i * 10;
          const snapP = spring({ frame: frame - snapDelay, fps, config: { damping: 10 } });

          const currentX = frame > snapPhase
            ? interpolate(snapP, [0, 1], [origX, snappedX])
            : origX;

          const dotColor = showTernary
            ? (snapped === -1 ? COLORS.red : snapped === 0 ? COLORS.textDim : COLORS.accent)
            : COLORS.blue;

          return (
            <g key={i} opacity={interpolate(dotP, [0, 1], [0, 1])}>
              {/* Ghost of original position */}
              {frame > snapPhase && (
                <circle cx={origX} cy={lineY} r={4} fill={COLORS.textDim} opacity={0.3} />
              )}
              {/* Snap line */}
              {frame > snapPhase && snapP > 0.1 && (
                <line
                  x1={origX}
                  y1={lineY}
                  x2={currentX}
                  y2={lineY}
                  stroke={dotColor}
                  strokeWidth={1}
                  strokeDasharray="3,3"
                  opacity={0.5}
                />
              )}
              {/* Dot */}
              <circle cx={currentX} cy={lineY} r={7} fill={dotColor} />
              {/* Value label */}
              <text
                x={currentX}
                y={lineY - 16}
                fill={COLORS.text}
                fontSize={11}
                textAnchor="middle"
                fontFamily={FONTS.mono}
              >
                {frame > snapPhase && snapP > 0.8 ? snapped : val.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Labels */}
      <div
        style={{
          position: "absolute",
          bottom: 40,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          gap: 32,
        }}
      >
        {frame > phase2 && !showTernary && (
          <FadeIn delay={phase2 + 20}>
            <div
              style={{
                background: COLORS.surface,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 8,
                padding: "12px 24px",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 12, color: COLORS.textDim }}>Quantization Error</div>
              <div style={{ fontSize: 14, color: COLORS.blue, fontWeight: 600, fontFamily: FONTS.mono }}>
                Small — values stay close
              </div>
            </div>
          </FadeIn>
        )}
        {showTernary && (
          <FadeIn delay={phase3 + 60}>
            <div
              style={{
                background: COLORS.surface,
                border: `1px solid ${COLORS.accent}40`,
                borderRadius: 8,
                padding: "12px 24px",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 12, color: COLORS.textDim }}>Ternary Quantization</div>
              <div style={{ fontSize: 14, color: COLORS.accent, fontWeight: 600, fontFamily: FONTS.mono }}>
                Extreme compression — only 3 values possible
              </div>
            </div>
          </FadeIn>
        )}
      </div>
    </div>
  );
}
