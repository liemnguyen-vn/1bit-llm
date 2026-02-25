import { COURSE } from "@/lib/course-data";
import ModulePageClient from "@/components/ModulePageClient";

export function generateStaticParams() {
  return COURSE.modules.map((m) => ({ moduleId: m.id }));
}

export default async function ModulePage({
  params,
}: {
  params: Promise<{ moduleId: string }>;
}) {
  const { moduleId } = await params;
  return <ModulePageClient moduleId={moduleId} />;
}
