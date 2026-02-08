import { afmlDocsState } from "./afmlDocsState";

export type AfmlSection = {
  id: string;
  module: string;
  slug: string;
  status: "pending" | "in_progress" | "done";
  chapter: string;
  chapterTheme: string;
};

const flatSections: AfmlSection[] = afmlDocsState.chapters.flatMap((chapter) =>
  chapter.sections.map((section) => ({
    id: section.id,
    module: section.module,
    slug: section.slug,
    status: section.status,
    chapter: chapter.chapter,
    chapterTheme: chapter.theme,
  })),
);

const byModule = new Map<string, AfmlSection[]>();
for (const section of flatSections) {
  const items = byModule.get(section.module) ?? [];
  items.push(section);
  byModule.set(section.module, items);
}

export function getSectionsByModule(moduleName: string): AfmlSection[] {
  return byModule.get(moduleName) ?? [];
}

export function getRelatedSections(moduleName: string): AfmlSection[] {
  const sections = getSectionsByModule(moduleName);
  if (sections.length === 0) {
    return [];
  }
  const chapters = new Set(sections.map((section) => section.chapter));
  return flatSections.filter((candidate) => chapters.has(candidate.chapter) && candidate.module !== moduleName);
}

export function chapterCoverage() {
  return afmlDocsState.chapters.map((chapter) => {
    const total = chapter.sections.length;
    const done = chapter.sections.filter((section) => section.status === "done").length;
    const inProgress = chapter.sections.filter((section) => section.status === "in_progress").length;
    const pending = total - done - inProgress;

    return {
      ...chapter,
      total,
      done,
      inProgress,
      pending,
      completionPct: total === 0 ? 0 : Math.round((done / total) * 100),
    };
  });
}
