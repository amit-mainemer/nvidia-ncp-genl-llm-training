import c01 from './m01-architecture.js'
import c02 from './m02-data.js'
import c03 from './m03-gpu.js'
import c04 from './m04-finetuning.js'
import c05 from './m05-evaluation.js'
import c06 from './m06-prompting.js'
import c07 from './m07-optimization.js'
import c08 from './m08-deployment.js'
import c09 from './m09-monitoring.js'
import c10 from './m10-safety.js'

/** End-of-lesson comprehension checks: moduleId -> lessonId -> questions. */
const checks = {
  architecture: c01,
  data: c02,
  gpu: c03,
  finetuning: c04,
  evaluation: c05,
  prompting: c06,
  optimization: c07,
  deployment: c08,
  monitoring: c09,
  safety: c10,
}

export function lessonCheck(moduleId, lessonId) {
  return checks[moduleId]?.[lessonId] || []
}

/** Share of questions needed to pass a lesson check and mark the lesson complete. */
export const CHECK_PASS = 0.7

export function checkPassMark(total) {
  return Math.ceil(total * CHECK_PASS)
}
