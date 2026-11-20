/* Module 9 — Production Monitoring and Reliability (official domain 9, 7%).
   Operating an LLM service after launch: what to measure, how to see problems early,
   and how to ship, compare, roll back and retrain model versions safely. */

/* ------------------------------------------------------------------ */
const observability = {
  id: 'observability',
  title: 'Observability: metrics, logs, traces, DCGM and drift',
  minutes: 22,
  outcomes: [
    'Define the golden signals of an LLM service — TTFT, inter-token latency, throughput, queue depth, errors, GPU health — and where each comes from',
    'Scrape vLLM, Triton, NIM and DCGM metrics with Prometheus and turn them into Grafana dashboards and alerts',
    'Log prompts and responses responsibly and trace requests end to end with OpenTelemetry',
    'Detect input, output and embedding drift and monitor quality with sampled LLM-judge evaluations',
    'Set SLOs and run a structured root-cause analysis when an alert fires',
  ],
  blocks: [
    { t: 'lede', x: 'A deployed LLM fails in ways a normal web service does not: it can be fast and "200 OK" while giving wrong, unsafe or rambling answers; it can slow down because users started sending longer prompts; it can run out of KV-cache memory under a traffic spike. [[observability-tracing|Observability]] means collecting enough **metrics, logs and traces** — plus quality and drift signals — that you notice these problems before users do, and can explain them when they happen.' },

    { t: 'h2', x: 'The golden signals for LLM serving' },
    { t: 'p', x: 'Classic SRE watches latency, traffic, errors and saturation — the [[golden-signals|golden signals]]. For LLMs, latency splits into two numbers because generation is streamed, and saturation is mostly about GPU memory and the request queue.' },
    { t: 'table', head: ['Signal', 'What it measures', 'Typical source', 'Why it moves'], rows: [
      ['**TTFT** (time to first token)', 'Request arrival → first streamed token; includes queueing + prefill', 'Engine histogram (e.g. `vllm:time_to_first_token_seconds`)', 'Longer prompts, queue build-up, prefill contention'],
      ['**TPOT / ITL** (time per output token, inter-token latency)', 'Average gap between streamed tokens during decode', 'Engine histogram', 'Larger batches, KV-cache pressure, memory bandwidth'],
      ['End-to-end latency', 'TTFT + output tokens × TPOT', 'Engine or gateway histogram', 'Longer outputs (prompt or model change)'],
      ['Throughput', 'Requests/s and output tokens/s', 'Counters → `rate()`', 'Traffic, batch size, model speed'],
      ['Queue depth / running requests', 'Requests waiting vs. being processed', 'Gauges (e.g. `vllm:num_requests_waiting`)', 'Load above capacity — leading indicator of TTFT breach'],
      ['KV-cache utilisation', 'Fraction of KV blocks in use', 'Engine gauge', 'Long contexts, high concurrency; at ~100% requests are preempted or queue'],
      ['Error rate', '5xx, timeouts, rejected/over-length requests, guardrail blocks', 'Gateway / server counters', 'Bugs, OOMs, overload, bad inputs'],
      ['GPU health', 'Utilisation, memory used, temperature, power, Xid errors', '[[dcgm|DCGM]] exporter', 'Hardware faults, throttling, memory leaks'],
    ]},
    { t: 'p', x: 'Always look at **percentiles (p50, p95, p99)**, never only averages: a few very long prompts can wreck p99 [[ttft|TTFT]] while the mean looks fine. And remember the two latency numbers respond to different knobs — raising the batch size improves throughput but worsens [[tpot|TPOT]]; long prompts hurt TTFT. The deployment module covers how to benchmark them; here we monitor them continuously.' },

    { t: 'h2', x: 'Metrics pipeline: Prometheus and Grafana' },
    { t: 'p', x: 'The standard stack is pull-based: each component exposes a `/metrics` endpoint in Prometheus text format, [[prometheus|Prometheus]] scrapes it every 10–30 s and stores time series, [[grafana|Grafana]] draws dashboards from PromQL queries, and Alertmanager routes alerts. On [[kubernetes|Kubernetes]], the Prometheus Operator discovers targets through ServiceMonitor/PodMonitor objects.' },
    { t: 'specs', rows: [
      ['vLLM', '`/metrics` on the API port (8000 by default): TTFT, inter-token/TPOT and e2e latency histograms, running/waiting requests, KV-cache usage, prompt/generation token counters (names prefixed `vllm:`; exact names evolve between versions)'],
      ['Triton Inference Server', '`/metrics` on port **8002**: `nv_inference_request_success`/`_failure`, `nv_inference_queue_duration_us`, `nv_inference_compute_infer_duration_us`, GPU utilisation and memory; the TensorRT-LLM backend adds in-flight-batching and KV-cache metrics'],
      ['NVIDIA NIM', 'Prometheus-format metrics endpoint exposing the underlying engine\'s request, latency, token and KV-cache metrics (see the NIM observability docs for the path and names in your version)'],
      ['DCGM exporter', '`/metrics` on port **9400**: `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_XID_ERRORS`, profiling fields such as `DCGM_FI_PROF_SM_ACTIVE`; deployed as a DaemonSet by the GPU Operator'],
    ]},
    { t: 'code', lang: 'yaml', title: 'prometheus.yml — scrape an LLM server and the GPUs', x: `global:
  scrape_interval: 15s
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["llm-server:8000"]        # vLLM serves /metrics on its API port
  - job_name: triton
    static_configs:
      - targets: ["triton:8002"]            # Triton metrics port
  - job_name: dcgm
    static_configs:
      - targets: ["gpu-node-1:9400", "gpu-node-2:9400"]   # dcgm-exporter on each GPU node
rule_files: ["llm-alerts.yml"]` },
    { t: 'code', lang: 'text', title: 'PromQL for the dashboard (check metric names against your engine version)', x: `# p95 time-to-first-token over 5 minutes
histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))

# output tokens per second across all replicas
sum(rate(vllm:generation_tokens_total[1m]))

# requests waiting in the queue (saturation)
sum(vllm:num_requests_waiting)

# error ratio from Triton
sum(rate(nv_inference_request_failure[5m]))
  / (sum(rate(nv_inference_request_success[5m])) + sum(rate(nv_inference_request_failure[5m])))

# GPU memory used per GPU (MiB) and any Xid errors
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_XID_ERRORS > 0` },

    { t: 'h2', x: 'SLOs and alerting' },
    { t: 'p', x: 'An [[slo|SLO]] is a target on a user-visible indicator over a time window, e.g. "99% of requests have TTFT < 1 s, measured over 28 days". The allowed 1% is the [[error-budget|error budget]]. Alert on **symptoms users feel** (SLO burn rate, error ratio, p95 TTFT) and keep cause-level metrics (GPU utilisation, queue depth) on dashboards for diagnosis — paging someone because GPU utilisation is 95% is noise if users are fine.' },
    { t: 'code', lang: 'yaml', title: 'llm-alerts.yml — two symptom alerts and one hardware alert', x: `groups:
  - name: llm-serving
    rules:
      - alert: HighTTFT
        expr: histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m]))) > 2
        for: 10m
        labels: { severity: page }
        annotations:
          summary: "p95 TTFT above 2 s for 10 min — check queue depth, prompt length, replicas"
      - alert: QueueBuildingUp
        expr: sum(vllm:num_requests_waiting) > 50
        for: 5m
        labels: { severity: ticket }
        annotations:
          summary: "Requests queuing — scale out before TTFT SLO burns"
      - alert: GPUXidError
        expr: DCGM_FI_DEV_XID_ERRORS > 0
        labels: { severity: page }
        annotations:
          summary: "Xid error on {{ $labels.gpu }} at {{ $labels.Hostname }} — drain node and inspect"` },
    { t: 'callout', kind: 'tip', title: 'Queue depth is your early warning', x: 'Queue depth and KV-cache utilisation rise *before* TTFT breaches its SLO. Using them as autoscaling signals (e.g. via KEDA or the Horizontal Pod Autoscaler with custom metrics) adds capacity before users notice; GPU utilisation is a poor scaling signal for LLMs because a busy engine can sit near 100% at both healthy and overloaded loads.' },

    { t: 'h2', x: 'Logs and traces, done responsibly' },
    { t: 'p', x: 'Metrics tell you *that* something is wrong; logs and traces tell you *which request* and *where*. For every request, emit one **structured** (JSON) log line with a request id, model and version, prompt/completion token counts, latency breakdown, finish reason (stop, length, error, content filter) and guardrail decisions. Whether you also store the **prompt and response text** is a privacy decision, not a default:' },
    { t: 'ul', items: [
      'Redact or hash PII before storage (names, emails, phone and account numbers); NeMo Curator, Presidio or guardrail PII detectors can do this in the pipeline.',
      'Log full text only for a sampled fraction or for flagged requests, with access controls, encryption and a retention limit that matches your privacy policy and regulations (e.g. GDPR).',
      'Never log secrets from system prompts or tool calls; keep user consent and opt-outs in mind.',
      'Keep text in a separate, restricted store from operational metrics so engineers can debug latency without reading user data.',
    ]},
    { t: 'p', x: '[[observability-tracing|Distributed tracing]] with OpenTelemetry links the spans of one request across services — API gateway, guardrails, retriever, vector database, re-ranker, LLM engine — so you can see that a slow answer spent 1.8 s in retrieval and only 300 ms generating. Several serving engines ([[vllm|vLLM]] among them) can export OpenTelemetry traces themselves; OpenTelemetry also defines `gen_ai.*` semantic-convention attributes for model name and token usage.' },
    { t: 'code', lang: 'python', title: 'Tracing a RAG request with OpenTelemetry', x: `from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("rag-service")

def answer(question: str, user_tier: str) -> str:
    with tracer.start_as_current_span("rag.request") as span:
        span.set_attribute("app.user_tier", user_tier)          # no raw prompt text in spans
        with tracer.start_as_current_span("retrieve"):
            docs = retriever.search(question, k=5)
        with tracer.start_as_current_span("llm.generate") as gen:
            resp = client.chat.completions.create(model="my-llm-v3", messages=build_messages(question, docs))
            gen.set_attribute("gen_ai.request.model", "my-llm-v3")
            gen.set_attribute("gen_ai.usage.input_tokens", resp.usage.prompt_tokens)
            gen.set_attribute("gen_ai.usage.output_tokens", resp.usage.completion_tokens)
            gen.set_attribute("app.finish_reason", resp.choices[0].finish_reason)
        return resp.choices[0].message.content` },

    { t: 'h2', x: 'Drift and quality monitoring' },
    { t: 'p', x: 'Latency dashboards cannot tell you that answers got worse. Two families of signals can. **[[data-drift|Drift]]** compares the distribution of recent traffic with a reference window (e.g. the evaluation set or last month): input drift (prompt length, language, topic mix), output drift (response length, refusal rate, finish-reason mix, sentiment), and [[embedding-drift|embedding drift]] — embed prompts with a sentence-embedding model and compare the clouds. **Quality monitoring** directly scores a sample of live traffic.' },
    { t: 'code', lang: 'python', title: 'Two cheap drift checks: PSI on a scalar, centroid shift on embeddings', x: `import numpy as np

def psi(reference, current, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate, >0.25 significant shift."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference) + 1e-6
    cur = np.histogram(current, edges)[0] / len(current) + 1e-6
    return float(np.sum((cur - ref) * np.log(cur / ref)))

def centroid_shift(ref_emb, cur_emb):
    """Cosine distance between mean embeddings of two windows (0 = same direction)."""
    a, b = ref_emb.mean(0), cur_emb.mean(0)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# prompt lengths in tokens: last month vs. today
print("PSI prompt length:", psi(ref_prompt_lens, today_prompt_lens))
# embeddings from any sentence-embedding model, shape (n, d)
print("embedding centroid shift:", centroid_shift(ref_embs, today_embs))
# Also track: refusal rate, mean output length, % finish_reason == "length", topic-cluster shares.` },
    { t: 'p', x: 'For quality, sample 1–5% of production requests (plus anything users flagged with a thumbs-down), score them offline with an LLM judge against a rubric — groundedness/faithfulness for [[rag|RAG]], helpfulness, policy compliance, format validity — and chart the pass rate per model version. Calibrate the judge against a small set of human labels, and route low-scoring samples to human review. A falling judge score with stable latency is exactly the failure that only this monitoring catches.' },
    { t: 'callout', kind: 'warn', title: 'Drift is a signal, not a verdict', x: [
      'A topic shift after a product launch is drift, but may be harmless if quality scores hold. Pair every drift alert with a quality check on the drifted slice before reacting.',
      'Judge models drift too — pin the judge model and prompt version, or a change in the judge will look like a change in your model.',
    ]},

    { t: 'h2', x: 'Root-cause analysis' },
    { t: 'p', x: 'When an alert fires, work from symptom to cause with the data you collected, and change one thing at a time:' },
    { t: 'flow', steps: [
      { b: 'Scope', s: 'Which signal, since when, which model version, replicas, regions or tenants? Correlate the start time with deploys, config or traffic changes.' },
      { b: 'Localise', s: 'Use traces to find the slow or failing span (gateway, guardrail, retriever, engine). Compare a good and a bad request.' },
      { b: 'Check saturation', s: 'Queue depth, KV-cache usage, GPU memory and DCGM health (Xid errors, throttling) on the affected replicas.' },
      { b: 'Check inputs', s: 'Prompt-length and topic drift, a new client sending huge contexts, a prompt-template change.' },
      { b: 'Mitigate', s: 'Roll back the change, scale out, drain bad nodes, or rate-limit the offending client — restore service first.' },
      { b: 'Learn', s: 'Blameless post-mortem: root cause, why monitoring did (not) catch it early, new alert or test to prevent recurrence.' },
    ]},
    { t: 'table', head: ['Symptom', 'Likely cause', 'Confirm with'], rows: [
      ['p95 TTFT up, TPOT flat', 'Queueing or longer prompts (prefill)', 'Queue depth, prompt-token histogram'],
      ['TPOT up, TTFT flat', 'Bigger batches or KV-cache pressure', 'Running requests, KV-cache usage'],
      ['Error spikes on one replica', 'Node/GPU fault or OOM', 'DCGM Xid errors, pod restarts, engine logs'],
      ['Latency fine, user complaints up', 'Quality regression or drift', 'Judge pass rate, refusal rate, recent model/prompt changes'],
      ['Responses suddenly much longer', 'Prompt/template or sampling config change', 'Output-length histogram, finish reasons, config diff'],
    ]},
    { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
      'Map the metric to the problem: TTFT ↔ prefill/queueing, TPOT ↔ decode/batching, queue depth ↔ saturation, KV-cache usage ↔ memory pressure.',
      '"Collect GPU utilisation, memory, temperature and Xid errors cluster-wide into Prometheus" → **DCGM exporter**.',
      '"Find which microservice in a RAG pipeline adds latency" → **distributed tracing** (OpenTelemetry).',
      '"Quality degrades but latency and errors are normal" → **drift + sampled quality evaluation**, not more GPU metrics.',
      'Triton metrics live on port 8002; dcgm-exporter on 9400.',
    ]},

    { t: 'h2', x: 'Key takeaways' },
    { t: 'ul', items: [
      'Monitor TTFT, TPOT/ITL, e2e latency (as percentiles), throughput, queue depth, KV-cache usage, error rate and DCGM GPU health.',
      'Prometheus scrapes `/metrics` from vLLM, Triton (8002), NIM and dcgm-exporter (9400); Grafana visualises; alert on SLO symptoms, diagnose with cause metrics.',
      'Log structured per-request metadata always; log prompt/response text only with redaction, sampling, access control and retention limits.',
      'OpenTelemetry traces localise latency across gateway, retrieval and generation.',
      'Detect input/output/embedding drift and score sampled traffic with a calibrated LLM judge — the only way to see silent quality regressions.',
    ]},
  ],
}

/* ------------------------------------------------------------------ */
const lifecycle = {
  id: 'lifecycle',
  title: 'Reliability and the model lifecycle: versioning, canaries, rollback and retraining',
  minutes: 20,
  outcomes: [
    'Version an LLM release as a complete bundle — weights, tokenizer, prompt template, generation and serving config — in a model registry',
    'Benchmark a candidate against the current production version and gate promotion on regressions',
    'Choose between shadow, canary, blue-green and A/B rollouts, and plan a fast rollback',
    'Define retraining triggers and an automated, reproducible retraining and continuous-evaluation pipeline',
  ],
  blocks: [
    { t: 'lede', x: 'Models are not deployed once; they are replaced — new fine-tunes, new base models, new quantizations, new prompts. Each change can improve one thing and silently break another. Reliability in the LLM lifecycle comes from four habits: **version everything that affects outputs, compare every candidate against what is in production, roll out gradually with an instant way back, and retrain on evidence, not on a calendar alone.**' },

    { t: 'h2', x: 'What a "model version" really is' },
    { t: 'p', x: 'For an LLM, the weights are only part of the behaviour. The same weights with a different tokenizer, [[chat-template|chat template]], system prompt, sampling settings or quantization can behave very differently. A release in the [[model-registry|model registry]] should pin all of it:' },
    { t: 'table', head: ['Component', 'Example', 'Why it must be versioned'], rows: [
      ['Weights (+ adapters)', 'Base model hash, LoRA adapter id, merged or not', 'The core of behaviour; adapters can change independently'],
      ['Tokenizer + chat template', 'Tokenizer files, Jinja chat template, special tokens', 'A template mismatch silently degrades quality'],
      ['System prompt & prompt templates', 'Prompt version `support-v14`', 'Prompt edits change outputs as much as fine-tunes'],
      ['Generation config', 'Temperature, top-p, max tokens, stop sequences', 'Affects length, determinism, cost'],
      ['Optimisation & engine', 'FP8/INT4 quantization recipe, TensorRT-LLM / vLLM / NIM version, engine build flags', 'Quantization and kernels can shift accuracy and latency'],
      ['Data & training lineage', 'Dataset versions, training config, code commit, container tag', 'Needed to reproduce, audit and retrain'],
      ['Evaluation results', 'Benchmark scores, judge pass rates, safety evals', 'The evidence the release was approved on'],
    ]},
    { t: 'code', lang: 'python', title: 'Registering a release bundle in MLflow and promoting it with an alias', x: `import mlflow
from mlflow import MlflowClient

with mlflow.start_run(run_name="support-llm-ft-2026-11") as run:
    mlflow.log_params({
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct", "adapter": "lora-r16-support-v7",
        "prompt_version": "support-v14", "temperature": 0.3, "max_tokens": 512,
        "quantization": "fp8", "engine": "vllm", "train_data": "support-sft@v7",
        "git_commit": "a1b2c3d", "container": "nvcr.io/nvidia/pytorch:<tag>",
    })
    mlflow.log_metrics({"judge_pass_rate": 0.91, "exact_match": 0.78, "p95_ttft_s": 0.42})
    # Directory containing weights/adapter, tokenizer, chat template, generation_config.json, prompts/
    mlflow.log_artifacts("release/", artifact_path="bundle")

mv = mlflow.register_model(f"runs:/{run.info.run_id}/bundle", "support-llm")
client = MlflowClient()
client.set_registered_model_alias("support-llm", "challenger", mv.version)
# After the canary passes:   client.set_registered_model_alias("support-llm", "champion", mv.version)
# Rollback = point "champion" back at the previous version number.` },
    { t: 'p', x: 'The same idea exists in every stack: Hugging Face Hub revisions (git commits and tags), NGC private registry model versions, Weights & Biases artifacts, or a container image tag that bakes the bundle in. What matters is that production references an **immutable version**, and that "which version served this request?" is answerable from the request log.' },

    { t: 'h2', x: 'Regression benchmarking against the previous version' },
    { t: 'p', x: 'Before any traffic sees a candidate, run the **same** evaluation suite on the candidate and the current production version and compare per metric and per slice. Benchmarks with fixed seeds, pinned judge model and pinned prompts make the comparison fair.' },
    { t: 'ul', items: [
      '**Task quality**: held-out task set (exact match, F1, pass@k), LLM-judge pass rate on a golden set of real user prompts, RAG faithfulness.',
      '**Regression set**: every past production bug becomes a test case that must keep passing.',
      '**Safety**: jailbreak and harmful-content sets, refusal rate on benign prompts (over-refusal is a regression too).',
      '**Performance**: TTFT, TPOT and throughput at target concurrency (e.g. with genai-perf), GPU memory, cost per 1k tokens.',
      '**Slices**: languages, customer tiers, long vs. short prompts — an average can hide a large drop on one slice.',
    ]},
    { t: 'code', lang: 'python', title: 'A promotion gate in CI', x: `# Compare candidate vs. production on the same suite; block promotion on any regression.
GATES = {                       # metric: (higher_is_better, max allowed regression)
    "judge_pass_rate": (True, 0.01),
    "exact_match":     (True, 0.01),
    "jailbreak_block": (True, 0.00),
    "benign_refusal":  (False, 0.01),
    "p95_ttft_s":      (False, 0.10),
}

def gate(prod: dict, cand: dict) -> list[str]:
    failures = []
    for m, (higher_better, tol) in GATES.items():
        delta = cand[m] - prod[m] if higher_better else prod[m] - cand[m]
        if delta < -tol:
            failures.append(f"{m}: prod={prod[m]:.3f} cand={cand[m]:.3f}")
    return failures

fails = gate(prod_scores, cand_scores)
if fails:
    raise SystemExit("Promotion blocked:\\n" + "\\n".join(fails))
print("All gates passed — promote candidate to canary")` },

    { t: 'h2', x: 'Rolling out: shadow, canary, blue-green, A/B' },
    { t: 'vs', cards: [
      { h: 'Shadow', items: [
        'Copy live requests to the candidate; its responses are logged, **never returned**',
        'Zero user risk; measures latency, errors and (with a judge) quality on real traffic',
        'Costs extra GPU capacity; cannot measure user reactions',
      ]},
      { h: 'Canary', items: [
        'Route a small share (1–5%) of real traffic to the candidate, then 25%, 50%, 100%',
        'Automatic promotion/abort on SLOs, error rate and quality metrics per step',
        'Limits blast radius; the standard default for model updates',
      ]},
      { h: 'Blue-green', items: [
        'Stand up the full new stack (green) next to the old (blue) and switch all traffic at once',
        'Instant rollback by switching back; needs double capacity during the switch',
      ]},
      { h: 'A/B test', items: [
        'Split users randomly between versions for a fixed period to measure **business/user outcomes** (thumbs-up rate, task completion, retention)',
        'Needs enough traffic and a pre-registered metric for statistical significance',
      ]},
    ]},
    { t: 'p', x: 'Typical sequence: offline gate → shadow → [[canary-deployment|canary]] with automated analysis → full rollout, with an [[ab-testing|A/B test]] when the question is "do users prefer it?" rather than "is it safe?". [[kubernetes|Kubernetes]] tooling does the traffic splitting: KServe\'s `canaryTrafficPercent`, Argo Rollouts or Flagger with Prometheus-based analysis, or weighted routes in a service mesh or API gateway. Use sticky routing (by user or session) so one conversation does not hop between versions mid-dialogue.' },
    { t: 'code', lang: 'yaml', title: 'KServe canary: send 10% of traffic to the new model revision', x: `apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: support-llm
spec:
  predictor:
    canaryTrafficPercent: 10          # the latest revision gets 10%, the previous keeps 90%
    model:
      modelFormat: { name: huggingface }
      storageUri: s3://models/support-llm/v8     # new version; v7 stays live as the stable revision
      resources:
        limits: { nvidia.com/gpu: "1" }
# Promote: remove canaryTrafficPercent (or set 100). Roll back: set it to 0.` },

    { t: 'h2', x: 'Rollback: the most important feature you hope not to use' },
    { t: 'p', x: 'A [[rollback|rollback]] must be **fast, rehearsed and boring**: keep the previous version deployed (or warm) during and after rollout, make the switch a config change (alias, traffic weight, image tag) rather than a rebuild, and define the trigger in advance — e.g. "abort if canary error rate > 1% or judge pass rate drops > 2 points or p95 [[ttft|TTFT]] exceeds the SLO for 10 minutes". LLM-specific gotchas: large engines take minutes to load, so a cold rollback breaks the SLO; prompt and model changes should roll back together if they shipped together; and cached responses or conversation state produced by the bad version may need purging.' },
    { t: 'callout', kind: 'trap', title: 'The "silent" bad release', x: 'A new fine-tune passes latency and error checks in canary, so it is promoted — and a week later users complain it refuses legitimate requests. Canary analysis that only watches HTTP errors and latency cannot catch quality regressions. Include sampled judge scores, refusal rate and output-length shifts in the canary gate.' },

    { t: 'h2', x: 'Retraining triggers and continuous evaluation' },
    { t: 'p', x: 'Retraining (or re-fine-tuning, or updating the [[rag|RAG]] index) should be driven by signals from monitoring, with a scheduled baseline:' },
    { t: 'table', head: ['Trigger', 'Signal', 'Typical response'], rows: [
      ['Quality decline', 'Judge pass rate or task metric below threshold on live samples', 'Collect failing cases, add to SFT/preference data, re-fine-tune'],
      ['Data drift', 'PSI / embedding drift on inputs; new topics or languages appear', 'Refresh data, update RAG index first (cheaper), then fine-tune if needed'],
      ['Knowledge staleness', 'Questions about new products/policies fail', 'Update the RAG corpus — retraining is the wrong tool for fresh facts'],
      ['User feedback', 'Thumbs-down rate, escalations to humans', 'Label and mine feedback into training data'],
      ['New base model or optimisation', 'Better base release, cheaper quantization', 'Re-run the full fine-tune + evaluation pipeline'],
      ['Schedule', 'Monthly/quarterly cadence', 'Retrain on accumulated data, still gated by evaluation'],
    ]},
    { t: 'flow', steps: [
      { b: 'Trigger', s: 'Monitoring alert, accumulated feedback or schedule starts the pipeline (e.g. Kubeflow, Argo Workflows, Airflow, NeMo microservices).' },
      { b: 'Data', s: 'Pull versioned production samples + feedback, redact PII, deduplicate, decontaminate against eval sets, version the dataset.' },
      { b: 'Train', s: 'Fine-tune (often LoRA) from the pinned base with a pinned container, config and seed.' },
      { b: 'Evaluate', s: 'Full regression suite vs. production: quality, safety, slices, performance.' },
      { b: 'Register', s: 'Log the bundle and results to the registry as a challenger.' },
      { b: 'Roll out', s: 'Shadow → canary with automated analysis → promote, or auto-rollback.' },
    ]},
    { t: 'p', x: '[[continuous-evaluation|Continuous evaluation]] keeps the evaluation suite running on a schedule against production — not just at release time — so you notice when the world (inputs, user expectations, upstream APIs, the judge) changes under a fixed model. Keep a human in the loop for promotion of anything customer-facing: automated pipelines should *propose* a new version with evidence, and gates should be able to stop it.' },

    { t: 'h2', x: 'Reproducibility' },
    { t: 'p', x: 'You cannot debug or audit a model you cannot rebuild. Pin the container image (an NGC tag or digest), library versions, code commit, dataset versions (hashes or a data-versioning tool such as DVC or lakeFS), training config and random seeds; store all of them with the registry entry. Exact bit-for-bit reproducibility on GPUs additionally needs deterministic kernels (`torch.use_deterministic_algorithms(True)`, fixed `CUBLAS_WORKSPACE_CONFIG`) and the same hardware and parallel layout — usually you aim for *statistical* reproducibility (same metrics within noise), and bit-exactness only for debugging.' },
    { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
      '"Validate a new model on real traffic with zero user impact" → **shadow** deployment.',
      '"Expose a small percentage of users and expand if metrics hold" → **canary**.',
      '"Measure which version users prefer / business KPI" → **A/B test**.',
      '"Benchmark against previous versions before promoting" → same eval suite on candidate and production, gate on regressions.',
      '"Answers about recent policies are wrong" → update **RAG** data, not retrain; "quality drops on a new input distribution" → drift-triggered **retraining**.',
    ]},

    { t: 'h2', x: 'Key takeaways' },
    { t: 'ul', items: [
      'A model version = weights + adapters + tokenizer/chat template + prompts + generation config + quantization/engine + lineage + eval results, stored immutably in a registry.',
      'Gate every candidate on the same regression suite as production, including safety, slices and latency.',
      'Roll out shadow → canary → full, use A/B tests for user preference, and keep the previous version warm for instant rollback.',
      'Retrain on monitoring triggers (quality, drift, feedback) plus a schedule; fix stale knowledge with RAG updates.',
      'Pin containers, code, data, configs and seeds so any release can be rebuilt and audited.',
    ]},
  ],
}

/* ------------------------------------------------------------------ */
export default {
  id: 'monitoring',
  n: 9,
  domain: 'monitoring',
  weight: 7,
  title: 'Production Monitoring and Reliability',
  blurb: 'Run LLM services you can trust: monitor TTFT, TPOT, queues and GPU health with Prometheus, Grafana and DCGM, trace and log responsibly, catch drift and silent quality regressions, and ship model versions safely with registries, canaries, rollback and evidence-driven retraining.',
  hours: '1.5–2 h',
  lessons: [observability, lifecycle],
  quiz: [
    { id: 'm01', domain: 'monitoring',
      scenario: 'After a marketing campaign, p95 time-to-first-token on a vLLM deployment rises from 0.4 s to 3 s, while time per output token stays flat and the error rate is unchanged.',
      q: 'Which metric should you check first to confirm the cause?',
      opts: [
        'GPU temperature from DCGM',
        'The number of requests waiting in the queue',
        'The LLM-judge pass rate on sampled traffic',
        'Average output length per request',
      ],
      a: 1,
      why: 'TTFT includes queueing time plus prefill; flat TPOT means decode is unaffected. A traffic surge beyond capacity shows up first as **queue depth** (e.g. `vllm:num_requests_waiting`), which also argues for scaling out.',
      distract: '**Temperature** would affect all latencies, not only TTFT. **Judge pass rate** measures quality, not latency. **Output length** changes end-to-end latency, not time to the first token.' },

    { id: 'm02', domain: 'monitoring',
      q: 'Which component exposes GPU utilisation, framebuffer memory, temperature, power and Xid errors from every GPU node to Prometheus?',
      opts: [
        'The Triton metrics endpoint on port 8002',
        'Nsight Systems',
        'An OpenTelemetry collector',
        'The DCGM exporter',
      ],
      a: 3,
      why: 'The [[dcgm|DCGM]] exporter (port 9400, deployed by the GPU Operator as a DaemonSet) publishes `DCGM_FI_*` fields for every GPU on the node in Prometheus format.',
      distract: '**Triton** reports metrics for the GPUs it uses, not a fleet-wide hardware view with Xid errors. **Nsight Systems** is an offline profiler. **OpenTelemetry collectors** forward telemetry; they do not read GPU counters themselves.' },

    { id: 'm03', domain: 'monitoring',
      scenario: 'Latency and error dashboards for a customer-support LLM are all green, but customer satisfaction has fallen for two weeks. A product change brought many questions about a new subscription plan.',
      q: 'Which monitoring would most directly have caught this?',
      opts: [
        'Input drift detection plus sampled LLM-judge scoring of live responses',
        'A lower alert threshold on p99 end-to-end request latency',
        'DCGM SM-occupancy and memory alerts on the inference GPUs',
        'OpenTelemetry tracing of every request through the API gateway',
      ],
      a: 0,
      why: 'Topic [[data-drift|drift]] (new questions) plus a quality signal on live samples reveals silent quality regressions that operational metrics cannot see. The fix is likely a RAG/knowledge update.',
      distract: 'Latency, GPU occupancy and gateway traces all measure speed and plumbing, which were fine. None measure answer correctness.' },

    { id: 'm04', domain: 'monitoring',
      q: 'A team wants to evaluate a new fine-tuned model on real production prompts without any user ever seeing its output. Which strategy fits?',
      opts: [
        'Canary deployment at 1% of traffic',
        'Blue-green deployment',
        'Shadow deployment',
        'A/B testing with a holdout group',
      ],
      a: 2,
      why: 'A [[shadow-deployment|shadow deployment]] mirrors live requests to the candidate and logs its responses without returning them, so latency, errors and judged quality can be measured at zero user risk.',
      distract: 'A **canary** and an **A/B test** both serve the candidate\'s answers to some users. **Blue-green** switches all traffic to the new stack.' },

    { id: 'm05', domain: 'monitoring',
      scenario: 'A new model version passed the canary on latency and HTTP error rate and was promoted. One week later, users report it refuses many legitimate requests. The previous version is no longer deployed.',
      q: 'Which two process changes best prevent a repeat?',
      opts: [
        'Raise the canary traffic share to 50% and shorten the canary phase to one hour',
        'Retrain the model every week and skip the canary phase for small changes',
        'Replace the canary with a blue-green switch and drop the latency gates entirely',
        'Add judge scores and refusal rate to the canary gate, and keep the previous version warm',
      ],
      a: 3,
      why: 'The canary only watched operational metrics, so a quality regression slipped through, and [[rollback|rollback]] was slow because the old version was gone. Quality-aware gates and a warm previous version fix both.',
      distract: 'More traffic for less time increases blast radius without adding a quality signal. Skipping canaries and removing gates reduce safety. Blue-green does not add a quality check.' },

    { id: 'm06', domain: 'monitoring',
      q: 'Which set of artefacts should a model-registry entry pin so that a production LLM release is reproducible and rollback-safe?',
      opts: [
        'Weights, adapters, tokenizer, chat template, prompts, generation and engine config, evals',
        'The model weights file together with its SHA-256 checksum and file size',
        'The serving container image tag and the Kubernetes deployment manifest',
        'The training dataset snapshot and the final training and validation loss',
      ],
      a: 0,
      why: 'Outputs depend on everything that turns a request into tokens: weights/adapters, tokenizer and template, prompts, sampling and quantization/engine settings. Lineage and eval results make the release auditable and reproducible in the [[model-registry|registry]].',
      distract: 'Weights alone miss the template and prompts that change behaviour. A container tag and manifest omit data lineage and eval evidence. Data and final loss do not describe what is served.' },
  ],
}
