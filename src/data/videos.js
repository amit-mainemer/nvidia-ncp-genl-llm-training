/**
 * Intro videos for modules and lessons. None are produced for this course yet;
 * add entries here ({ file, length } served from public/videos/) to enable them.
 */
const modules = {}

/** 'moduleId/lessonId' -> lesson video */
const lessons = {}

function withUrls(v) {
  if (!v) return null
  const base = `${import.meta.env.BASE_URL}videos/${v.file}`
  return { ...v, mp4: `${base}.mp4`, vtt: `${base}.vtt`, poster: `${base}.jpg` }
}

export const moduleVideo = (moduleId) => withUrls(modules[moduleId])
export const lessonVideo = (moduleId, lessonId) => withUrls(lessons[`${moduleId}/${lessonId}`])
