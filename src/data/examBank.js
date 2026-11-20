/**
 * A separate 65-question bank used for the timed mock exam, weighted to the
 * official NCP-GENL blueprint. Each domain's questions live in src/data/exam/.
 * These questions do not appear in lesson checks or module quizzes, so a mock
 * exam is a genuine test rather than a recall of what you just read.
 */
import architecture from './exam/architecture.js'
import data from './exam/data.js'
import gpu from './exam/gpu.js'
import finetuning from './exam/finetuning.js'
import evaluation from './exam/evaluation.js'
import prompting from './exam/prompting.js'
import optimization from './exam/optimization.js'
import deployment from './exam/deployment.js'
import monitoring from './exam/monitoring.js'
import safety from './exam/safety.js'

export const examBank = [
  ...architecture, ...prompting, ...data, ...optimization, ...finetuning,
  ...evaluation, ...gpu, ...deployment, ...monitoring, ...safety,
]
