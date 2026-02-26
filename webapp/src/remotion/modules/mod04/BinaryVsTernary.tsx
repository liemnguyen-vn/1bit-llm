import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, SlideIn, Title, Subtitle } from "../../utils";

const GRID_SIZE = 6;

function makeGrid(values: number[]) {
  const grid: number[][] = [];
  let idx = 0;
  for (let r = 0; r < GRID_SIZE; r++) {
    const row: number[] = [];
    for (let c = 0; c < GRID_SIZE; c++) {
      row.push(values[idx % values.length]);
      idx++;
    }
    grid.push(row);
  }
  return grid;
}

// Deterministic pattern
const binaryVals = [1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1];
const ternaryVals = [1, -1, 0, 1, -1, 0, 0, 1, -1, 0, 1, -1, 1, 0, -1, 0, 1, -1, -1, 0, 1, 1, -1, 0, 0, 1, -1, 1, 0, -1, -1, 0, 1, 0, 1, -1];

const binaryGrid = makeGrid(binaryVals);
const ternaryGrid = makeGrid(ternaryVals);

function WeightGrid({
  grid,
  delay,
  colorMap,
}: {
  grid: number[][];
  delay: number;
  colorMap: Record<number, string>;
}) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {grid.map((row, ri) => (
        <div key={ri} style={{ display: "flex", gap: 4 }}>
          {row.map((val, ci) => {
            const cellDelay = delay + ri * 3 + ci * 2;
            const p = spring({ frame: frame - cellDelay, fps, config: { damping: 12 } });
            return (
              <div
                key={ci}
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 14,
                  fontWeight: 700,
                  fontFamily: FONTS.mono,
                  background: colorMap[val] ?? COLORS.surface,
                  color: val === 0 ? COLORS.textDim : COLORS.bg,
                  opacity: interpolate(p, [0, 1], [0, 1]),
                  transform: `scale(${interpolate(p, [0, 1], [0.5, 1])})`,
                }}
              >
                {val > 0 ? `+${val}` : val}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

export default function BinaryVsTernary() {
  const frame = useCurrentFrame();

  const binaryColorMap: Record<number, string> = {
    1: COLORS.blue,
    "-1": COLORS.red,
  };
  const ternaryColorMap: Record<number, string> = {
    1: COLORS.accent,
    0: COLORS.surfaceLight,
    "-1": COLORS.red,
  };

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
        <Title>Binary vs Ternary Weights</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Adding zero gives ternary networks much greater expressiveness
        </Subtitle>
      </FadeIn>

      <div style={{ display: "flex", gap: 48, marginTop: 28, flex: 1, justifyContent: "center" }}>
        {/* Binary */}
        <SlideIn delay={15} from="left">
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: COLORS.blue, marginBottom: 12 }}>
              Binary: {"{-1, +1}"}
            </div>
            <WeightGrid grid={binaryGrid} delay={20} colorMap={binaryColorMap} />
            <div
              style={{
                marginTop: 12,
                fontSize: 12,
                color: COLORS.textDim,
              }}
            >
              1 bit per weight — all or nothing
            </div>
          </div>
        </SlideIn>

        {/* Ternary */}
        <SlideIn delay={60} from="right">
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: COLORS.accent, marginBottom: 12 }}>
              Ternary: {"{-1, 0, +1}"}
            </div>
            <WeightGrid grid={ternaryGrid} delay={65} colorMap={ternaryColorMap} />
            <div
              style={{
                marginTop: 12,
                fontSize: 12,
                color: COLORS.textDim,
              }}
            >
              1.58 bits — zero enables sparsity
            </div>
          </div>
        </SlideIn>
      </div>

      {frame > 160 && (
        <FadeIn delay={160}>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              gap: 32,
              marginTop: 8,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
              <div style={{ width: 16, height: 16, borderRadius: 4, background: COLORS.surfaceLight }} />
              <span style={{ color: COLORS.textDim }}>Zero = skip computation</span>
            </div>
            <div style={{ fontSize: 13, color: COLORS.accent, fontWeight: 600 }}>
              Zero weights make ternary both faster and more expressive
            </div>
          </div>
        </FadeIn>
      )}
    </div>
  );
}
