/** Glossary terms owned by Module 9 — Production Monitoring and Reliability. */
export default [
  { id: 'dcgm', term: 'DCGM (Data Center GPU Manager)', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s toolkit for monitoring and managing GPUs in clusters: health checks and diagnostics (`dcgmi diag`), policy alerts, and telemetry including profiling counters (SM active, tensor-pipe active, DRAM active, NVLink/PCIe traffic). The DCGM exporter publishes these as Prometheus metrics on port 9400 (e.g. `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_XID_ERRORS`); the GPU Operator deploys it on Kubernetes.',
    exam: 'The answer for cluster-wide GPU utilisation, memory, temperature, power and Xid-error monitoring in Prometheus/Grafana.' },

  { id: 'prometheus', term: 'Prometheus', cat: 'Serving & MLOps',
    def: 'An open-source, pull-based time-series monitoring system: it scrapes `/metrics` endpoints (counters, gauges, histograms) on an interval, stores the series, evaluates PromQL queries and alerting rules, and hands alerts to Alertmanager.',
    exam: 'vLLM, Triton (port 8002), NIM and dcgm-exporter all expose Prometheus metrics. Use `histogram_quantile` for p95/p99 latency.' },

  { id: 'grafana', term: 'Grafana', cat: 'Serving & MLOps',
    def: 'An open-source dashboarding and alerting tool that queries data sources such as Prometheus, Loki (logs) and Tempo (traces). NVIDIA and the serving engines publish ready-made dashboards for DCGM, Triton and vLLM metrics.',
    exam: 'Prometheus collects, Grafana visualises — the standard pair for "build a real-time dashboard of inference and GPU metrics".' },

  { id: 'golden-signals', term: 'Golden signals', cat: 'Serving & MLOps',
    def: 'The four SRE signals for any service — latency, traffic, errors, saturation. For LLM serving, latency splits into time to first token and time per output token, traffic into requests/s and tokens/s, and saturation into queue depth, KV-cache usage and GPU memory.',
    exam: 'Map each symptom to a signal: TTFT ↔ queueing/prefill, TPOT ↔ decode/batching, queue depth ↔ capacity.' },

  { id: 'slo', term: 'SLO / SLI / SLA', cat: 'Serving & MLOps',
    def: 'An SLI is a measured indicator (e.g. fraction of requests with TTFT < 1 s); an SLO is the internal target for it over a window (99% over 28 days); an SLA is a contractual promise to customers, usually looser than the SLO, with penalties.',
    exam: 'Alert on SLO burn (user-visible symptoms), not on raw resource metrics like GPU utilisation.' },

  { id: 'error-budget', term: 'Error budget', cat: 'Serving & MLOps',
    def: 'The amount of unreliability an SLO allows — for a 99% SLO, 1% of requests in the window. Burn-rate alerts fire when the budget is being consumed too fast; an exhausted budget is a signal to pause risky releases.',
    exam: 'Connects monitoring to release policy: plenty of budget → ship faster; budget burned → freeze and fix reliability.' },

  { id: 'observability-tracing', term: 'Observability & distributed tracing', cat: 'Serving & MLOps',
    def: 'Observability is the ability to explain a system\'s behaviour from its outputs: metrics, logs and traces. A distributed trace follows one request across services as a tree of timed spans (gateway, guardrails, retriever, vector DB, LLM), typically instrumented with OpenTelemetry and stored in Jaeger, Tempo or similar.',
    exam: '"Which stage of the RAG pipeline is slow for this request?" → tracing. Metrics show that something is wrong; traces show where.' },

  { id: 'data-drift', term: 'Data drift (and concept drift)', cat: 'Serving & MLOps',
    def: 'A change in the distribution of production inputs (or outputs) relative to a reference window: new topics, languages, longer prompts. Concept drift is a change in what the right answer is (e.g. a new policy). Measured with statistics such as PSI, KS tests or KL/JS divergence on features like prompt length, topic mix and refusal rate.',
    exam: 'Drift is a trigger to check quality on the drifted slice and possibly refresh RAG data or retrain — not proof of a regression by itself.' },

  { id: 'embedding-drift', term: 'Embedding drift', cat: 'Serving & MLOps',
    def: 'Drift measured in the space of text embeddings: embed recent and reference prompts (or responses) with a sentence-embedding model and compare the distributions — centroid cosine distance, MMD, a domain classifier, or shifts in cluster membership.',
    exam: 'The practical way to detect semantic/topic drift in free text, where simple scalar features miss it.' },

  { id: 'continuous-evaluation', term: 'Continuous evaluation', cat: 'Evaluation',
    def: 'Running evaluation suites on a schedule and on sampled live traffic — not only at release — including LLM-judge scoring against a rubric, regression sets of past failures, and safety sets, with results tracked per model version.',
    exam: 'Catches silent quality regressions caused by input drift, upstream changes or judge changes while latency metrics look healthy.' },

  { id: 'model-registry', term: 'Model registry', cat: 'Serving & MLOps',
    def: 'A versioned catalogue of model releases with metadata, lineage, evaluation results and stage/alias labels (e.g. challenger, champion). For LLMs a release should bundle weights/adapters, tokenizer and chat template, prompts, generation and engine config. Examples: MLflow Model Registry, NGC private registry, Hugging Face Hub revisions, W&B Registry.',
    exam: 'The answer for "track which model version is in production, promote and roll back, and reproduce a release".' },

  { id: 'canary-deployment', term: 'Canary deployment', cat: 'Serving & MLOps',
    def: 'Releasing a new version to a small share of real traffic (e.g. 1–5%), comparing its metrics with the stable version, and increasing the share step by step — or aborting automatically if SLOs, errors or quality degrade. Implemented with KServe `canaryTrafficPercent`, Argo Rollouts, Flagger or weighted gateway routes.',
    exam: 'Limits blast radius of a bad model. Include quality metrics in the canary analysis, not just latency and errors.' },

  { id: 'shadow-deployment', term: 'Shadow deployment', cat: 'Serving & MLOps',
    def: 'Mirroring live requests to a candidate model whose responses are recorded but never returned to users, so its latency, errors and judged quality can be measured on real traffic with no user risk.',
    exam: '"Validate on production traffic with zero user impact" → shadow. It cannot measure user reactions — that needs canary or A/B.' },

  { id: 'blue-green-deployment', term: 'Blue-green deployment', cat: 'Serving & MLOps',
    def: 'Running the new version (green) as a complete parallel environment next to the current one (blue), then switching all traffic at once; switching back is the rollback. Requires double capacity during the transition.',
    exam: 'All-at-once switch with instant rollback, versus a canary\'s gradual percentage-based exposure.' },

  { id: 'ab-testing', term: 'A/B testing', cat: 'Serving & MLOps',
    def: 'Randomly assigning users (sticky per user or session) to two or more versions for a fixed period and comparing pre-registered outcome metrics — thumbs-up rate, task success, conversion — with a statistical significance test.',
    exam: 'Use A/B when the question is "which version do users prefer / which improves the business metric", not "is it safe to roll out".' },

  { id: 'rollback', term: 'Rollback', cat: 'Serving & MLOps',
    def: 'Returning production to the previous known-good version. For LLMs it should be a configuration change (registry alias, traffic weight, image tag) to a version that is still deployed or warm, because loading large engines from cold can take minutes.',
    exam: 'Define rollback triggers before rollout and keep the previous version available; roll back prompt and model changes together if they shipped together.' },

  { id: 'retraining-trigger', term: 'Retraining trigger', cat: 'Serving & MLOps',
    def: 'A condition that starts an automated retraining or re-fine-tuning pipeline: quality below threshold on live samples, significant drift, accumulated negative feedback, a new base model, or a schedule. The pipeline still gates the result on evaluation before rollout.',
    exam: 'Stale facts → update RAG; changed input distribution or behaviour gaps → retrain/fine-tune.' },

  { id: 'root-cause-analysis', term: 'Root-cause analysis (RCA)', cat: 'Serving & MLOps',
    def: 'A structured investigation from an observed symptom to its underlying cause: scope the impact, correlate with changes, localise with traces and logs, check saturation and hardware, verify inputs, mitigate, and record a blameless post-mortem with preventive actions.',
    exam: 'Correlate the start of a symptom with deployments and traffic changes first — most incidents follow a change.' },

  { id: 'reproducibility', term: 'Reproducibility (ML)', cat: 'Serving & MLOps',
    def: 'Being able to rebuild a model and its results: pinned containers and libraries, code commit, versioned datasets, configs and seeds. Bit-exact GPU reproduction additionally needs deterministic kernels and the same hardware and parallel layout.',
    exam: 'Store lineage with each registry entry; aim for statistical reproducibility in production and bit-exactness for debugging.' },
]
