import type { FC } from "react";

export interface AnimationMeta {
  id: string;
  title: string;
  component: FC;
  durationInFrames: number;
  fps: number;
  width: number;
  height: number;
}
