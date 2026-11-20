# NCP-GENL Academy — curriculum spec (authoring reference)

**Goal of the course:** take a developer who knows Python from "what is a tensor" to *confidently training,
fine-tuning, evaluating, optimising and serving small LLMs (10M–1B params)* — and, along the way, cover every
domain of the **NVIDIA-Certified Professional: Generative AI LLMs (NCP-GENL)** exam.

Practical first: every module has runnable code. Concepts are explained intuitively, then precisely (with the
formula / memory arithmetic where it matters), then shown in code, then tied to the exam.

Exam facts (official page https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/):
60–70 questions · 120 min · $200 · online proctored (Certiverse) · valid 2 years · Professional level.
NVIDIA publishes domain weights but **no numbered sub-objectives**, so each lesson lists its own `outcomes`.

Official domains (official number · weight): 1 LLM Architecture 6% · 2 Prompt Engineering 13% ·
3 Data Preparation 9% · 4 Model Optimization 17% · 5 Fine-Tuning 13% · 6 Evaluation 7% ·
7 GPU Acceleration and Optimization 14% · 8 Model Deployment 9% · 9 Production Monitoring and Reliability 7% ·
10 Safety, Ethics, and Compliance 5%.
Official domain descriptions worth keeping in mind:
- Model Optimization (17%): latency/throughput optimisation of LLMs — quantization, pruning, distillation, TensorRT-LLM, batching, KV cache, Triton tuning.
- GPU Acceleration (14%): "multi-GPU/distributed setups, parallelism techniques, troubleshooting, memory and batch optimization, and performance profiling".
- Data Preparation (9%): "cleaning, curating, analyzing, and organizing datasets, tokenization, and vocabulary management".
- Prompt Engineering (13%): "prompt engineering, chain-of-thought, domain adaptation, zero/one/few-shot learning, and output control".
- Evaluation (7%): "quantitative and qualitative metrics, framework design, benchmarking, error analysis, and scalable evaluation".
- Deployment (9%): "containerized pipelines, scalable orchestration, efficient batch/model serving, and real-time monitoring".
- Monitoring (7%): "dashboards and reliability metrics… logs and anomalies for root cause analysis… benchmarking against previous versions… automated tuning, retraining, and versioning".
- Safety (5%): "auditing for bias and fairness, implementing guardrails, configuring monitoring for ethical compliance… bias detection and mitigation".

## Modules → lessons (learning order)

| # | Module id · file | Weight | Lessons (id — title) | Quiz | Exam bank |
|---|---|---|---|---|---|
| 1 | `architecture` · `m01-architecture` | 6% | `pytorch-foundations` — Tensors, autograd and your first training loop · `attention` — Self-attention and the transformer block, from scratch · `mini-gpt` — Lab: build and train a tiny GPT · `modern-architectures` — Modern LLM design: RoPE, GQA, SwiGLU, MoE and scaling laws | 8 | 4 |
| 2 | `data` · `m02-data` | 9% | `tokenization` — Tokenizers: BPE, vocabulary and special tokens · `curation` — Pretraining data: cleaning, dedup, filtering and NeMo Curator · `sft-datasets` — Instruction data: chat templates, packing and loss masking | 8 | 6 |
| 3 | `gpu` · `m03-gpu` | 14% | `memory-math` — GPU memory math: weights, gradients, optimizer states, activations · `mixed-precision` — Mixed precision, gradient accumulation, checkpointing and FlashAttention · `pretrain-small-llm` — Lab: pretrain a small LLM end to end · `distributed` — Distributed training: DDP, FSDP/ZeRO, tensor, pipeline and sequence parallelism · `profiling` — Profiling and troubleshooting training: MFU, Nsight, OOMs and NaNs | 12 | 9 |
| 4 | `finetuning` · `m04-finetuning` | 13% | `when-to-finetune` — Choosing an adaptation method: prompting, RAG, PEFT, full FT, continued pretraining · `full-sft` — Lab: supervised fine-tuning with TRL · `lora-qlora` — Lab: LoRA and QLoRA on a single GPU · `alignment` — Preference alignment: RLHF, PPO, DPO and GRPO · `nemo-customization` — NeMo customization, p-tuning, adapters and avoiding catastrophic forgetting | 12 | 9 |
| 5 | `evaluation` · `m05-evaluation` | 7% | `metrics` — Metrics: loss, perplexity, BLEU/ROUGE, exact match, pass@k · `benchmarks` — Lab: benchmarks with lm-evaluation-harness, and contamination · `llm-judge` — LLM-as-a-judge, human eval, error analysis and RAG evaluation | 8 | 5 |
| 6 | `prompting` · `m06-prompting` | 13% | `prompting-basics` — Zero-, one- and few-shot prompting, system prompts and in-context learning · `reasoning-decoding` — Chain-of-thought, self-consistency and decoding controls (temperature, top-k/top-p, structured output) · `rag-domain` — Domain adaptation without training: RAG and prompt-based adaptation | 10 | 8 |
| 7 | `optimization` · `m07-optimization` | 17% | `quantization` — Lab: quantization — FP8, INT8, INT4, GPTQ, AWQ, SmoothQuant · `pruning-distillation` — Pruning and knowledge distillation (Minitron) · `inference-engine` — Inside the inference engine: KV cache, paged attention, in-flight batching, speculative decoding · `tensorrt-llm` — TensorRT-LLM and Triton: building and tuning optimised engines | 12 | 10 |
| 8 | `deployment` · `m08-deployment` | 9% | `serving` — Lab: serving your model — vLLM, NIM, Triton, merged vs multi-LoRA · `containers-k8s` — Containers, NGC and Kubernetes: scaling LLM inference · `latency-throughput` — Latency, throughput and capacity planning (TTFT, TPOT, genai-perf) | 8 | 6 |
| 9 | `monitoring` · `m09-monitoring` | 7% | `observability` — Observability: metrics, logs, traces, DCGM and drift · `lifecycle` — Reliability and the model lifecycle: versioning, canaries, rollback and retraining | 6 | 5 |
| 10 | `safety` · `m10-safety` | 5% | `guardrails` — Guardrails: NeMo Guardrails, jailbreaks, prompt injection and PII · `responsible-ai` — Responsible AI: bias auditing, red teaming, model cards and licensing | 6 | 3 |

Totals: 34 lessons · 90 quiz questions · 65 exam-bank questions (blueprint-weighted).

## Files each module owns

- `src/data/modules/mNN-<id>.js` — module object (below).
- `src/data/checks/mNN-<id>.js` — `export default { '<lessonId>': [5 questions], … }` — **exactly 5 per lesson**.
- `src/data/glossary/<id>.js` — `export default [ { id, term, cat, def, exam }, … ]` — terms *owned* by the module (~12–25).
- `src/data/exam/<id>.js` — `export default [ …questions ]` — the exam-bank count from the table. Never reuse a quiz question.

## Module object shape

```js
export default {
  id: 'gpu', n: 3, domain: 'gpu', weight: 14,
  title: 'GPU Acceleration and Optimization',
  blurb: '1–2 sentences: what you will be able to do after this module.',
  hours: '5–7 h',
  lessons: [
    { id: 'memory-math', title: '…', minutes: 18,
      outcomes: ['Estimate …', 'Explain …', 'Choose …'],   // 3–5 outcomes, each starting with an imperative verb
      blocks: [ … ] },
  ],
  quiz: [ { id: 'g01', domain: 'gpu', scenario: '…optional…', q, opts: [4], a, why, distract }, … ],
}
```

Module titles = the official domain names exactly (see `src/data/exam.js`).

## Content blocks

`lede`, `h2`, `h3`, `p` (`x` string) · `ul` / `ol` (`items: [...]`) · `code` (`lang`, `title`, `x`) ·
`callout` (`kind`: exam | tip | warn | trap, `title?`, `x`: string | string[]) · `table` (`head`, `rows`) ·
`vs` (`cards: [{h, items}]`) · `flow` (`steps: [{b, s}]`) · `specs` (`rows: [[k, v]]`) ·
**`lab`** (`title`, `x` = goal paragraph, `gpu` = hardware needed e.g. 'CPU / Apple Silicon OK', 'Free Colab T4', '1× 24 GB GPU', `time` e.g. '30 min', `steps: [strings]`).
A `lab` block introduces a hands-on exercise; put its code in the `code` blocks right after it.

Inline markup (inside any string): `[[glossary-id|display text]]` or `[[glossary-id]]`, `**bold**`, `` `code` ``, `[label](https://url)`.
Do not put `**` inside backticks; keep `**` balanced. Use template literals for code with `\` escaped as `\\` and `${` escaped as `\${`.

## Questions

`{ q, opts: [4 distinct strings], a: 0–3, why, distract }`; quiz and exam questions also need `id`, `domain` (module id),
optional `scenario`. `why` explains the right answer (may use glossary links), `distract` says why EACH wrong option fails.
Spread correct answers evenly across A/B/C/D. Options of similar length — the longest option must not be a giveaway.
ID prefixes: architecture `a`, data `d`, gpu `g`, finetuning `f`, evaluation `e`, prompting `p`, optimization `o`,
deployment `y`, monitoring `m`, safety `s`. Quiz ids `a01…`, exam-bank ids `xa01…`.

## Glossary

Shared terms are already in `src/data/glossary/core.js` — link to them, never redefine them:
llm, tensor, autograd, backprop, loss, cross-entropy, logits, softmax, gradient-descent, adamw, learning-rate,
batch-size, overfitting, parameters, transformer, self-attention, embedding, context-window, decoder-only, token,
tokenizer, pretraining, bf16, fp16, fp8, cuda, tensor-core, hbm, nccl, fine-tuning, sft, peft, lora, rlhf,
perplexity, rag, hallucination, inference, kv-cache, quantization, nemo, megatron, tensorrt-llm, triton, nim, hugging-face.

Term ownership (define in the owner's file; others may link to these ids):
- architecture: multi-head-attention, causal-mask, positional-encoding, rope, layernorm, rmsnorm, residual, ffn, swiglu, gqa, mqa, moe, scaling-laws, chinchilla, weight-tying, encoder-only, encoder-decoder
- data: bpe, sentencepiece, vocabulary, special-tokens, chat-template, deduplication, minhash, quality-filtering, nemo-curator, data-contamination (also used by evaluation), packing, loss-masking, synthetic-data, pii-redaction
- gpu: mixed-precision, loss-scaling, gradient-accumulation, activation-checkpointing, flash-attention, mfu, ddp, fsdp, zero, tensor-parallelism, pipeline-parallelism, sequence-parallelism, context-parallelism, expert-parallelism, all-reduce, nvlink, transformer-engine, nsight-systems, torch-compile, oom
- finetuning: full-fine-tuning, qlora, nf4, adapters, prompt-tuning, p-tuning, dpo, ppo, grpo, reward-model, catastrophic-forgetting, continued-pretraining, instruction-tuning, merge-adapter
- evaluation: bleu, rouge, bertscore, exact-match, pass-at-k, mmlu, lm-eval-harness, llm-as-judge, human-eval, error-analysis, faithfulness, ragas
- prompting: zero-shot, few-shot, in-context-learning, chain-of-thought, self-consistency, system-prompt, temperature, top-k, top-p, greedy-decoding, beam-search, structured-output, repetition-penalty, react, vector-database
- optimization: ptq, qat, int8, int4, awq, gptq, smoothquant, fp4, pruning, distillation, minitron, paged-attention, inflight-batching, speculative-decoding, prefill, decode, model-optimizer
- deployment: vllm, openai-api, docker, ngc, kubernetes, gpu-operator, autoscaling, multi-lora, ttft, tpot, throughput, genai-perf, triton-model-repository, dynamic-batching
- monitoring: dcgm, prometheus, grafana, data-drift, slo, canary-deployment, ab-testing, model-registry, rollback, observability-tracing
- safety: nemo-guardrails, jailbreak, prompt-injection, red-teaming, bias, fairness-metrics, model-card, toxicity, llama-guard, data-privacy

You may add other terms to your own file if needed (check no other file uses the id).
If you link to a term owned by another module, use exactly the id above.
Glossary `cat` must be one of: Foundations · Architecture · Data & tokenization · Training & GPUs · Distributed training ·
Fine-tuning & alignment · Evaluation · Prompting & RAG · Inference optimization · Serving & MLOps · Safety & governance ·
NVIDIA stack · Tools & frameworks.

## Style

- Voice: direct, practical, second person. Explain *why*, not just *what*. Short paragraphs.
- Every lesson: `lede` → 4–7 `h2` sections → at least one `table`/`vs`/`flow`/`specs` → code where it helps →
  ≥1 `callout` of kind `exam` (how it is tested) and ≥1 `trap` or `warn` → a "Key takeaways" `h2` + `ul` at the end.
- Code: real, runnable, current APIs (PyTorch 2.x, transformers ≥4.45, trl ≥0.12, peft ≥0.13, datasets ≥3).
  Prefer small models that run on a free Colab T4 or a single 24 GB GPU: e.g. `Qwen/Qwen2.5-0.5B`, `HuggingFaceTB/SmolLM2-135M`,
  `roneneldan/TinyStories` dataset. Add comments that teach. Mention the NVIDIA-scale equivalent (NeMo, Megatron, TensorRT-LLM, NIM).
- Do numbers: memory arithmetic, FLOPs, token counts — the exam loves them.
- Lesson length: 12–25 minutes of reading (`minutes`). Labs can be longer.
- Validate with `node tools/check-content.mjs <moduleId>` until it prints OK.
