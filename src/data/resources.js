/* Training and reading list. `officialTraining` is NVIDIA's NCP-GENL learning path from the
   official exam page; `docs` are primary sources grouped by exam domain. */

export const officialTraining = [
  { name: 'Building RAG Agents With LLMs', price: '$90', hours: '8 h', mode: 'Self-paced',
    why: 'The entry point of NVIDIA\'s learning path for this exam: LLM pipelines, retrieval, dialog state and guardrails — the Prompt Engineering and Deployment domains in practice.',
    url: 'https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-FX-15+V1' },
  { name: 'Adding New Knowledge to LLMs', price: '$500', hours: '8 h', mode: 'Instructor-led',
    why: 'Continued pretraining, SFT, PEFT and alignment with NeMo, plus data curation and evaluation — the closest official course to Modules 2, 4 and 5 here.',
    url: 'https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-26+V1' },
  { name: 'Model Parallelism: Building and Deploying Large Neural Networks', price: '$500', hours: '8 h', mode: 'Instructor-led',
    why: 'Data, tensor and pipeline parallelism, memory optimisation and multi-node training — the GPU Acceleration domain (Module 3).',
    url: 'https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-07+V1' },
  { name: 'Deploying RAG Pipelines for Production at Scale', price: '$500', hours: '8 h', mode: 'Self-paced',
    why: 'NIM, Kubernetes, scaling and monitoring inference services — Modules 8 and 9.',
    url: 'https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-18+V1' },
  { name: 'Optimizing CUDA ML Codes With NVIDIA Nsight Profiling Tools', price: '$30', hours: '4 h', mode: 'Self-paced',
    why: 'Nsight Systems and Nsight Compute for finding GPU bottlenecks — the profiling half of the GPU Acceleration domain.',
    url: 'https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-AC-03+V2' },
]

export const docs = [
  { group: 'Certification', items: [
    { name: 'Official NCP-GENL exam page', url: 'https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/', note: 'Format, price, blueprint weights and the learning path. Re-check it before booking — weights change between revisions.' },
    { name: 'Register on Certiverse', url: 'https://www.certiverse.com/#/checkout/nvidia/store-exam/NCP-GENL', note: 'The proctoring platform that delivers the exam.' },
    { name: 'NCA-GENL (associate) exam page', url: 'https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/', note: 'The associate-level stepping stone if you want an easier first certification.' },
  ]},
  { group: 'Foundations — build it from scratch', items: [
    { name: 'Attention Is All You Need', url: 'https://arxiv.org/abs/1706.03762', note: 'The original transformer paper. Read sections 3.1–3.5 after Module 1.' },
    { name: 'Andrej Karpathy — Neural Networks: Zero to Hero', url: 'https://karpathy.ai/zero-to-hero.html', note: 'Free video series that builds backprop, a bigram model and GPT by hand. The best companion to Module 1.' },
    { name: 'nanoGPT', url: 'https://github.com/karpathy/nanoGPT', note: 'The minimal GPT training repo the Module 1 lab is modelled on.' },
    { name: 'PyTorch tutorials', url: 'https://pytorch.org/tutorials/', note: 'Tensors, autograd, nn.Module and the training loop.' },
    { name: 'Hugging Face LLM course', url: 'https://huggingface.co/learn/llm-course', note: 'Tokenizers, transformers, datasets and fine-tuning with the libraries used in the labs.' },
  ]},
  { group: 'Data, training and fine-tuning', items: [
    { name: 'Training Compute-Optimal LLMs (Chinchilla)', url: 'https://arxiv.org/abs/2203.15556', note: 'Where "about 20 tokens per parameter" comes from.' },
    { name: 'NeMo Curator', url: 'https://github.com/NVIDIA/NeMo-Curator', note: 'GPU-accelerated data curation: dedup, quality filtering, PII redaction.' },
    { name: 'NeMo Framework user guide', url: 'https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html', note: 'Pretraining, SFT, PEFT and alignment recipes at scale.' },
    { name: 'LoRA paper', url: 'https://arxiv.org/abs/2106.09685', note: 'Low-rank adaptation — the method behind most fine-tuning you will do.' },
    { name: 'QLoRA paper', url: 'https://arxiv.org/abs/2305.14314', note: 'NF4, double quantization and paged optimizers.' },
    { name: 'DPO paper', url: 'https://arxiv.org/abs/2305.18290', note: 'Preference alignment without a reward model.' },
    { name: 'TRL documentation', url: 'https://huggingface.co/docs/trl', note: 'SFTTrainer, DPOTrainer, GRPOTrainer — used in Module 4 labs.' },
    { name: 'PEFT documentation', url: 'https://huggingface.co/docs/peft', note: 'LoRA, QLoRA, prompt tuning and adapter merging.' },
  ]},
  { group: 'GPU acceleration and distributed training', items: [
    { name: 'Hugging Face — The Ultra-Scale Playbook', url: 'https://huggingface.co/spaces/nanotron/ultrascale-playbook', note: 'A superb, free deep dive into DP, ZeRO, TP, PP, CP and EP with memory maths.' },
    { name: 'Megatron-LM / Megatron-Core', url: 'https://github.com/NVIDIA/Megatron-LM', note: 'NVIDIA\'s reference implementation of tensor and pipeline parallelism.' },
    { name: 'PyTorch FSDP tutorial', url: 'https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html', note: 'Sharded data parallel training.' },
    { name: 'Transformer Engine', url: 'https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html', note: 'FP8 training on Hopper and Blackwell.' },
    { name: 'NCCL user guide', url: 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html', note: 'Collectives and environment variables for troubleshooting.' },
    { name: 'Nsight Systems', url: 'https://developer.nvidia.com/nsight-systems', note: 'Timeline profiling of training and inference.' },
  ]},
  { group: 'Evaluation', items: [
    { name: 'lm-evaluation-harness', url: 'https://github.com/EleutherAI/lm-evaluation-harness', note: 'The standard benchmark runner used in the Module 5 lab.' },
    { name: 'NeMo Evaluator', url: 'https://docs.nvidia.com/nemo/microservices/latest/evaluate/index.html', note: 'NVIDIA\'s evaluation microservice: benchmarks, LLM-as-judge and RAG metrics.' },
    { name: 'Judging LLM-as-a-Judge (MT-Bench)', url: 'https://arxiv.org/abs/2306.05685', note: 'Biases of LLM judges and how to mitigate them.' },
  ]},
  { group: 'Optimization, deployment and monitoring', items: [
    { name: 'TensorRT-LLM documentation', url: 'https://nvidia.github.io/TensorRT-LLM/', note: 'LLM API, quantization, in-flight batching and benchmarking.' },
    { name: 'TensorRT Model Optimizer', url: 'https://github.com/NVIDIA/TensorRT-Model-Optimizer', note: 'FP8, INT4-AWQ, pruning and distillation tooling.' },
    { name: 'Triton Inference Server', url: 'https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html', note: 'Model repository, dynamic batching and metrics.' },
    { name: 'NVIDIA NIM for LLMs', url: 'https://docs.nvidia.com/nim/large-language-models/latest/introduction.html', note: 'Optimised inference microservices with an OpenAI-compatible API.' },
    { name: 'vLLM documentation', url: 'https://docs.vllm.ai/', note: 'PagedAttention serving used in the Module 8 lab.' },
    { name: 'Mastering LLM techniques: inference optimization', url: 'https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/', note: 'NVIDIA\'s overview of KV cache, batching, quantization and parallelism for inference.' },
    { name: 'GenAI-Perf / AIPerf', url: 'https://github.com/ai-dynamo/aiperf', note: 'Benchmarking TTFT, ITL and throughput of LLM endpoints.' },
    { name: 'DCGM exporter', url: 'https://github.com/NVIDIA/dcgm-exporter', note: 'GPU metrics for Prometheus and Grafana.' },
  ]},
  { group: 'Safety, ethics and compliance', items: [
    { name: 'NeMo Guardrails', url: 'https://docs.nvidia.com/nemo/guardrails/latest/index.html', note: 'Input, output, dialog and retrieval rails with Colang.' },
    { name: 'garak — LLM vulnerability scanner', url: 'https://github.com/NVIDIA/garak', note: 'Automated red teaming for jailbreaks, prompt injection and leakage.' },
    { name: 'NVIDIA Trustworthy AI', url: 'https://www.nvidia.com/en-us/ai-data-science/trustworthy-ai/', note: 'NVIDIA\'s principles and Model Card++ practice.' },
  ]},
]

export const studyPlans = [
  { name: '4-week sprint', per: '~10 h/week', for: 'You already fine-tune models at work and mainly need exam coverage.',
    weeks: [
      'Modules 1–3 (architecture, data, GPU acceleration). Skim the labs you already know; do every quiz.',
      'Modules 4–6 (fine-tuning, evaluation, prompting). Run the LoRA/QLoRA lab end to end.',
      'Modules 7–8 (optimization — the biggest domain — and deployment). First full mock at the end of the week.',
      'Modules 9–10, then two more mocks. Review every wrong answer and its distractor analysis.',
    ]},
  { name: '8-week builder', per: '~6 h/week', for: 'You know Python and some ML, and want to genuinely train models — the recommended path.',
    weeks: [
      'Module 1: PyTorch, attention, and build your own tiny GPT. Get it generating text.',
      'Module 2: train a tokenizer, build an instruction dataset with a chat template.',
      'Module 3: memory math, mixed precision, then pretrain a small LLM end to end.',
      'Module 3 (distributed + profiling) and Module 4 lessons 1–2: your first full SFT.',
      'Module 4 lessons 3–5: LoRA/QLoRA and DPO on your own data.',
      'Modules 5–6: evaluate your models with lm-eval-harness and an LLM judge; prompting and RAG.',
      'Modules 7–8: quantize your fine-tuned model and serve it behind an OpenAI-compatible API.',
      'Modules 9–10, then three mock exams with targeted review of your weakest domain.',
    ]},
  { name: '12-week foundation', per: '~4 h/week', for: 'You are new to deep learning and want to start from first principles.',
    weeks: [
      'Karpathy\'s "Zero to Hero" micrograd + makemore videos, alongside Module 1 lesson 1.',
      'Module 1 lessons 2–3: attention and the tiny-GPT lab.', 'Module 1 lesson 4 + quiz; Module 2 lesson 1.',
      'Module 2 lessons 2–3 + quiz.', 'Module 3 lessons 1–2.', 'Module 3 lesson 3: the pretraining lab — let it run.',
      'Module 3 lessons 4–5 + quiz.', 'Module 4 lessons 1–3: SFT and LoRA labs.',
      'Module 4 lessons 4–5 + quiz; Module 5.', 'Module 6 + quiz; Module 7 lessons 1–2.',
      'Module 7 lessons 3–4 + quiz; Module 8.', 'Modules 9–10, then mock exams and review.',
    ]},
]

export const examDayTips = [
  'Model Optimization (17%), GPU Acceleration (14%), Prompt Engineering (13%) and Fine-Tuning (13%) are well over half the exam. If time is short, spend it there.',
  'Do the arithmetic: bytes per parameter (2 for bf16, 1 for FP8/INT8, 0.5 for INT4), ~16 bytes/param for full mixed-precision training with AdamW, KV cache = 2 × layers × kv_heads × head_dim × tokens × bytes.',
  'Pick the cheapest method that works: prompting → RAG → PEFT/LoRA → full fine-tuning → continued pretraining. Fresh or citable knowledge points to RAG; style, format or a skill points to fine-tuning.',
  'Name the bottleneck before choosing the fix: prefill is compute-bound (TTFT), decode is memory-bandwidth-bound (TPOT). Quantization and batching help decode; tensor parallelism helps both at a communication cost.',
  'Memory problems in training have a standard ladder: smaller micro-batch + gradient accumulation → activation checkpointing → mixed precision → ZeRO/FSDP sharding → PEFT → more GPUs with TP/PP.',
  'Map NVIDIA tools to the job: NeMo Curator (data), NeMo Framework/Megatron (training), NeMo-RL (alignment), NeMo Evaluator (eval), Model Optimizer (quantize/prune/distill), TensorRT-LLM (engine), Triton/NIM (serve), DCGM (GPU telemetry), NeMo Guardrails (safety).',
  'Watch for absolute words — "always", "never", "guarantees". They are usually the wrong option.',
  'With 60–70 questions in 120 minutes you have nearly 2 minutes each. Flag long scenario questions and come back to them.',
]
