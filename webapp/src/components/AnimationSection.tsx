"use client";

import { getAnimationsForModule } from "@/remotion/animation-registry";
import AnimationPlayer from "@/remotion/AnimationPlayer";

export default function AnimationSection({ moduleId }: { moduleId: string }) {
  const animations = getAnimationsForModule(moduleId);
  if (animations.length === 0) return null;

  return (
    <div style={{ marginBottom: 32 }}>
      {animations.map((anim) => (
        <AnimationPlayer key={anim.id} animation={anim} />
      ))}
    </div>
  );
}
