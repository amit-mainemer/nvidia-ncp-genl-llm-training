/**
 * Glossary powering the hover / click term popups across every article.
 * `id`   — the key used inside lesson text as [[id|display text]]
 * `exam` — why the term matters in practice and on NCP-GENL
 *
 * Shared terms live in glossary/core.js; each module adds its own file.
 * The first definition of an id wins, so core terms cannot be overridden.
 */
import core from './glossary/core.js'
import architecture from './glossary/architecture.js'
import data from './glossary/data.js'
import gpu from './glossary/gpu.js'
import finetuning from './glossary/finetuning.js'
import evaluation from './glossary/evaluation.js'
import prompting from './glossary/prompting.js'
import optimization from './glossary/optimization.js'
import deployment from './glossary/deployment.js'
import monitoring from './glossary/monitoring.js'
import safety from './glossary/safety.js'

const seen = new Set()
export const glossary = [...core, ...architecture, ...data, ...gpu, ...finetuning, ...evaluation, ...prompting, ...optimization, ...deployment, ...monitoring, ...safety]
  .filter((g) => (seen.has(g.id) ? false : seen.add(g.id)))

export const glossaryById = Object.fromEntries(glossary.map((g) => [g.id, g]))
export const glossaryCats = [...new Set(glossary.map((g) => g.cat))]
