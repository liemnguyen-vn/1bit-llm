import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import { COLORS, FONTS } from "../../theme";
import { FadeIn, Title, Subtitle } from "../../utils";

const BLOCKS = [
  { label: "Input\nEmbedding", color: COLORS.blue },
  { label: "RMS\nNorm", color: COLORS.purple },
  { label: "Self\nAttention", color: COLORS.amber },
  { label: "Add &\nNorm", color: COLORS.purple },
  { label: "Feed\nForward", color: COLORS.accent },
  { label: "Add &\nNorm", color: COLORS.purple },
  { label: "Output", color: COLORS.cyan },
];

export default function TransformerBlockFlow() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const packetStart = 60;
  const packetSpeed = 30; // frames per block

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
        <Title>Transformer Decoder Block</Title>
        <Subtitle style={{ marginTop: 4 }}>
          Data flows through normalization, attention, and feed-forward layers
        </Subtitle>
      </FadeIn>

      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          position: "relative",
        }}
      >
        {/* Blocks */}
        <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
          {BLOCKS.map((block, i) => {
            const blockDelay = 15 + i * 8;
            const p = spring({ frame: frame - blockDelay, fps, config: { damping: 15 } });
            const opacity = interpolate(p, [0, 1], [0, 1]);
            const scale = interpolate(p, [0, 1], [0.8, 1]);

            // Is packet at this block?
            const packetProgress = (frame - packetStart) / packetSpeed;
            const atThisBlock = packetProgress >= i && packetProgress < i + 1;
            const pastThisBlock = packetProgress >= i + 1;

            return (
              <div key={i} style={{ display: "flex", alignItems: "center" }}>
                <div
                  style={{
                    width: 95,
                    height: 90,
                    borderRadius: 10,
                    background: atThisBlock ? `${block.color}30` : COLORS.surface,
                    border: `2px solid ${atThisBlock ? block.color : pastThisBlock ? `${block.color}60` : COLORS.border}`,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    opacity,
                    transform: `scale(${atThisBlock ? 1.05 : scale})`,
                    transition: "transform 0.1s",
                    position: "relative",
                  }}
                >
                  <div
                    style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: atThisBlock ? block.color : COLORS.text,
                      textAlign: "center",
                      whiteSpace: "pre-line",
                      lineHeight: 1.3,
                    }}
                  >
                    {block.label}
                  </div>
                  {atThisBlock && (
                    <div
                      style={{
                        position: "absolute",
                        bottom: -24,
                        fontSize: 18,
                      }}
                    >
                      ▲
                    </div>
                  )}
                </div>
                {i < BLOCKS.length - 1 && (
                  <div style={{ display: "flex", alignItems: "center", width: 24 }}>
                    <svg width={24} height={20}>
                      <line
                        x1={0} y1={10} x2={18} y2={10}
                        stroke={pastThisBlock ? block.color : COLORS.border}
                        strokeWidth={2}
                      />
                      <polygon
                        points="14,5 22,10 14,15"
                        fill={pastThisBlock ? block.color : COLORS.border}
                      />
                    </svg>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Data packet */}
        {frame >= packetStart && (
          (() => {
            const packetProgress = Math.min(
              (frame - packetStart) / packetSpeed,
              BLOCKS.length - 1,
            );
            const blockIndex = Math.floor(packetProgress);
            const blockX = blockIndex * (95 + 24) + 47;
            const progress = spring({
              frame: frame - packetStart,
              fps,
              config: { damping: 30, mass: 2 },
            });
            return (
              <div
                style={{
                  position: "absolute",
                  top: "25%",
                  left: `calc(${blockX}px + ${(packetProgress - blockIndex) * 119}px - 20px)`,
                  width: 40,
                  height: 40,
                  borderRadius: "50%",
                  background: COLORS.accent,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 14,
                  fontWeight: 700,
                  color: COLORS.bg,
                  opacity: interpolate(progress, [0, 1], [0, 1]),
                  boxShadow: `0 0 20px ${COLORS.accent}60`,
                }}
              >
                x
              </div>
            );
          })()
        )}
      </div>

      {/* Residual connection labels */}
      <FadeIn delay={90}>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            gap: 24,
            marginTop: 8,
          }}
        >
          {[
            { label: "Residual connections (Add)", color: COLORS.purple },
            { label: "Layer normalization", color: COLORS.purple },
            { label: "Repeated N times", color: COLORS.textDim },
          ].map((item) => (
            <div
              key={item.label}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: 12,
                color: item.color,
              }}
            >
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: item.color,
                }}
              />
              {item.label}
            </div>
          ))}
        </div>
      </FadeIn>
    </div>
  );
}
