# NCP-GENL Academy — Train Your Own LLM

A free, hands-on course that takes you from a hand-written PyTorch training loop to **training, fine-tuning,
evaluating, optimising and serving small LLMs** — mapped to every domain of the
**NVIDIA-Certified Professional: Generative AI LLMs (NCP-GENL)** exam.

**Live site:** https://amit-mainemer.github.io/nvidia-ncp-genl-llm-training/

Built as a single-page React app with no backend — all progress lives in your browser's local storage.
Sister projects: [NCP-AAI Academy](https://amit-mainemer.github.io/nvidia-aai-certificate-training/) (Agentic AI) ·
[NCP-AII Academy](https://amit-mainemer.github.io/nvidia-infra-aii-certificate-training/) (AI Infrastructure).

---

## What's inside

| | |
|---|---|
| **10 modules, 34 lessons** | One module per official exam domain, ordered the way you actually build a model |
| **Hands-on labs** | Build a GPT from scratch, train a BPE tokenizer, pretrain a small LLM, SFT, LoRA/QLoRA, DPO, lm-eval-harness, quantization, serve with vLLM/NIM — complete runnable code, most on a free Colab T4 |
| **Lesson checks** | Five questions at the end of every lesson; passing marks the lesson complete |
| **Module quizzes** | Exam-style questions per module with explanations and distractor analysis |
| **65-question mock exam** | A separate, blueprint-weighted bank on a 120-minute clock, scored per domain |
| **Glossary** | Every technical term is hoverable inline, with a definition *and* how it's tested |
| **Resources** | NVIDIA's official learning path, key papers and docs, and study plans |

### Learning path

| # | Module | Exam weight | You will… |
|---|---|---|---|
| 1 | LLM Architecture | 6% | write tensors, autograd, attention and a tiny GPT from scratch |
| 2 | Data Preparation | 9% | train a tokenizer, curate/dedup data, build chat-formatted SFT data |
| 3 | GPU Acceleration and Optimization | 14% | do memory math, mixed precision, pretrain a small LLM, go distributed, profile |
| 4 | Fine-Tuning | 13% | run full SFT, LoRA/QLoRA and DPO; learn NeMo customization |
| 5 | Evaluation | 7% | measure perplexity, run benchmarks, build LLM-as-judge evals |
| 6 | Prompt Engineering | 13% | master few-shot, CoT, decoding controls, structured output and RAG |
| 7 | Model Optimization | 17% | quantize, prune, distill; understand KV cache, batching, TensorRT-LLM |
| 8 | Model Deployment | 9% | serve your model with vLLM / NIM / Triton on Kubernetes; plan capacity |
| 9 | Production Monitoring and Reliability | 7% | monitor TTFT/TPOT/DCGM, detect drift, version and roll back |
| 10 | Safety, Ethics, and Compliance | 5% | add guardrails, red-team, audit bias, write model cards |

### Exam facts (verify on the [official page](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/))

- **Code:** NCP-GENL · **Level:** Professional
- **Format:** 60–70 questions, 120 minutes, online remotely proctored (Certiverse)
- **Price:** $200 · **Validity:** 2 years
- **Background:** 2–3 years of practical AI/ML experience with LLMs

---

## Running locally

```bash
npm install
npm run dev      # http://localhost:5173/nvidia-ncp-genl-llm-training/
npm run build    # production build into dist/
node tools/check-content.mjs   # validate glossary links, answers and question counts
```

Requires Node 18+. Dependencies: React 18, React Router, Vite.

## Project layout

```
src/data/exam.js            exam facts + blueprint (domain ids, weights, colours, learning order)
src/data/modules/mNN-*.js   lessons (content blocks, outcomes, labs) + module quiz
src/data/checks/mNN-*.js    end-of-lesson checks
src/data/exam/*.js          65-question mock exam bank, one file per domain
src/data/glossary/*.js      glossary (core + per module)
src/data/resources.js       official training, reading list, study plans, exam-day tips
tools/CURRICULUM.md         authoring spec
```

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds and publishes `dist/` to GitHub Pages
(**Pages → Source: GitHub Actions**). The Vite `base` in `vite.config.js` must match the repository name.

---

An independent, community-built study companion. Not affiliated with or endorsed by NVIDIA.
