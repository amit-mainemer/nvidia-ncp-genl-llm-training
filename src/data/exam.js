export const exam = {
  code: 'NCP-GENL',
  name: 'NVIDIA-Certified Professional: Generative AI LLMs',
  level: 'Professional (intermediate)',
  duration: '120 minutes',
  questions: '60–70',
  price: '$200 USD',
  passing: 'Not published (aim for 75%+ here)',
  validity: '2 years',
  delivery: 'Online, remotely proctored (Certiverse)',
  language: 'English',
  prereq: '2–3 years of practical AI/ML experience with LLMs: transformers, prompt engineering, distributed parallelism, parameter-efficient fine-tuning, evaluation and performance profiling',
  officialUrl: 'https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/',
  registerUrl: 'https://www.certiverse.com/#/checkout/nvidia/store-exam/NCP-GENL',
}

/* One module per official domain, listed in LEARNING order (build → data → train →
   fine-tune → evaluate → prompt → optimise → deploy → operate → govern).
   `official` is the domain's number on NVIDIA's blueprint; `n` is the module number.
   Colours are reused across the app so a domain is always the same hue. */
export const blueprint = [
  { id: 'architecture', n: 1,  official: 1,  name: 'LLM Architecture',                       weight: 6,  color: '#22c7e8' },
  { id: 'data',         n: 2,  official: 3,  name: 'Data Preparation',                       weight: 9,  color: '#a78bfa' },
  { id: 'gpu',          n: 3,  official: 7,  name: 'GPU Acceleration and Optimization',      weight: 14, color: '#6d9dff' },
  { id: 'finetuning',   n: 4,  official: 5,  name: 'Fine-Tuning',                            weight: 13, color: '#8fdc4a' },
  { id: 'evaluation',   n: 5,  official: 6,  name: 'Evaluation',                             weight: 7,  color: '#facc15' },
  { id: 'prompting',    n: 6,  official: 2,  name: 'Prompt Engineering',                     weight: 13, color: '#ff9f5a' },
  { id: 'optimization', n: 7,  official: 4,  name: 'Model Optimization',                     weight: 17, color: '#f472b6' },
  { id: 'deployment',   n: 8,  official: 8,  name: 'Model Deployment',                       weight: 9,  color: '#2dd4bf' },
  { id: 'monitoring',   n: 9,  official: 9,  name: 'Production Monitoring and Reliability',  weight: 7,  color: '#e879f9' },
  { id: 'safety',       n: 10, official: 10, name: 'Safety, Ethics, and Compliance',         weight: 5,  color: '#f87171' },
]

export const domainById = Object.fromEntries(blueprint.map((d) => [d.id, d]))

/** Scale factor for weight bars so the largest domain (17%) fills ~95% of the track. */
export const WEIGHT_SCALE = 95 / Math.max(...blueprint.map((d) => d.weight))
