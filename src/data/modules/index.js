import m01 from './m01-architecture.js'
import m02 from './m02-data.js'
import m03 from './m03-gpu.js'
import m04 from './m04-finetuning.js'
import m05 from './m05-evaluation.js'
import m06 from './m06-prompting.js'
import m07 from './m07-optimization.js'
import m08 from './m08-deployment.js'
import m09 from './m09-monitoring.js'
import m10 from './m10-safety.js'

export const modules = [m01, m02, m03, m04, m05, m06, m07, m08, m09, m10]
export const moduleById = Object.fromEntries(modules.map((m) => [m.id, m]))

export const allLessons = modules.flatMap((m) =>
  m.lessons.map((l) => ({ ...l, moduleId: m.id, moduleTitle: m.title }))
)

export const totalLessons = allLessons.length
export const totalQuizQuestions = modules.reduce((n, m) => n + m.quiz.length, 0)
export const totalMinutes = allLessons.reduce((n, l) => n + l.minutes, 0)

/** Every hands-on lab block, in course order, for the Labs page. */
export const allLabs = allLessons.flatMap((l) =>
  l.blocks.filter((b) => b.t === 'lab').map((b) => ({ ...b, moduleId: l.moduleId, moduleTitle: l.moduleTitle, lessonId: l.id, lessonTitle: l.title }))
)

/** Flat pool of every module quiz question, used by the practice/mock builder. */
export const questionPool = modules.flatMap((m) =>
  m.quiz.map((q) => ({ ...q, moduleId: m.id, moduleTitle: m.title }))
)
