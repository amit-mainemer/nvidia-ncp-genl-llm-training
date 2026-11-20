/**
 * Core glossary — terms used across many modules. Module-specific terms live in
 * the sibling files (architecture.js, data.js, …). Categories used site-wide:
 *   Foundations · Architecture · Data & tokenization · Training & GPUs ·
 *   Distributed training · Fine-tuning & alignment · Evaluation · Prompting & RAG ·
 *   Inference optimization · Serving & MLOps · Safety & governance · NVIDIA stack · Tools & frameworks
 */
export default [
  /* ---------------- Foundations ---------------- */
  { id: 'llm', term: 'LLM (large language model)', cat: 'Foundations',
    def: 'A neural network — almost always a decoder-only transformer — trained to predict the next token over a very large text corpus. "Large" today means roughly 1B to 1T+ parameters; the "small LLMs" you will train in this course are 10M–1B.',
    exam: 'Every domain assumes you can reason about an LLM as "a next-token predictor whose behaviour is shaped by data, objective and decoding". Most wrong answers break one of those three.' },

  { id: 'tensor', term: 'Tensor', cat: 'Foundations',
    def: 'An n-dimensional array with a dtype (fp32, bf16, int8…) and a device (CPU or GPU). In PyTorch every input, weight, activation and gradient is a `torch.Tensor`; LLM activations are usually shaped (batch, sequence, hidden).',
    exam: 'Shape reasoning — (B, T, C) and how attention reshapes it into heads — shows up in architecture and memory questions.' },

  { id: 'autograd', term: 'Autograd', cat: 'Foundations',
    def: 'PyTorch\'s automatic differentiation engine. It records the operations of the forward pass as a graph and, on `loss.backward()`, applies the chain rule to fill `.grad` on every parameter that has `requires_grad=True`.',
    exam: 'Explains why activations must be kept in memory until the backward pass — the root of activation memory and of activation checkpointing.' },

  { id: 'backprop', term: 'Backpropagation', cat: 'Foundations',
    def: 'The algorithm that computes the gradient of the loss with respect to every parameter by applying the chain rule backwards through the network, reusing intermediate results from the forward pass.',
    exam: 'A training step = forward + backward + optimizer step. Backward costs roughly 2× the forward FLOPs, which is where the "6 × params × tokens" training-compute rule comes from.' },

  { id: 'loss', term: 'Loss function', cat: 'Foundations',
    def: 'A scalar the training process minimises. For language modelling it is the average cross-entropy between the model\'s predicted next-token distribution and the actual next token.',
    exam: 'Know what a healthy loss curve looks like (fast drop, then slow decline) and what spikes, plateaus or NaNs usually mean.' },

  { id: 'cross-entropy', term: 'Cross-entropy loss', cat: 'Foundations',
    def: 'For one token, −log p(correct token). Averaged over all predicted positions it is the standard LLM training objective. A model guessing uniformly over a vocabulary of V tokens has loss ln(V) — about 10.8 for a 50k vocabulary.',
    exam: 'Perplexity is exp(cross-entropy). An initial loss far above ln(V) usually means a bug in initialisation or labels.' },

  { id: 'logits', term: 'Logits', cat: 'Foundations',
    def: 'The raw, unnormalised scores the model outputs for every vocabulary token at each position — a tensor of shape (batch, sequence, vocab). Softmax turns them into probabilities.',
    exam: 'Temperature, top-k, top-p and logit bias all operate on logits before sampling.' },

  { id: 'softmax', term: 'Softmax', cat: 'Foundations',
    def: 'softmax(z)_i = exp(z_i) / Σ_j exp(z_j). Converts a vector of logits into a probability distribution. Used for next-token probabilities and inside attention to turn scores into weights.',
    exam: 'Dividing logits by a temperature before softmax sharpens (T<1) or flattens (T>1) the distribution.' },

  { id: 'gradient-descent', term: 'Gradient descent', cat: 'Foundations',
    def: 'Updating parameters in the direction that reduces the loss: θ ← θ − lr · ∇L. LLMs use mini-batch stochastic variants with adaptive optimizers such as AdamW.',
    exam: 'Learning rate is the single most important hyperparameter — too high diverges, too low wastes compute.' },

  { id: 'adamw', term: 'AdamW', cat: 'Foundations',
    def: 'The default LLM optimizer: Adam (per-parameter adaptive learning rates from running estimates of the gradient\'s first and second moments) with decoupled weight decay. It stores two extra fp32 values per parameter.',
    exam: 'Those two optimizer states are why full fine-tuning needs ~16 bytes per parameter in mixed precision — a classic memory question.' },

  { id: 'learning-rate', term: 'Learning rate & schedule', cat: 'Foundations',
    def: 'The step size of each optimizer update. LLM training uses a short linear warmup followed by cosine (or linear / WSD) decay to a small fraction of the peak.',
    exam: 'Fine-tuning uses much smaller learning rates than pretraining (e.g. 1e-5 – 2e-5 for full SFT, 1e-4 – 2e-4 for LoRA). A too-high LR is the first suspect for loss spikes and catastrophic forgetting.' },

  { id: 'batch-size', term: 'Batch size', cat: 'Foundations',
    def: 'How many sequences (or tokens) contribute to one optimizer step. The effective batch = micro-batch × gradient-accumulation steps × data-parallel replicas. LLM batches are often quoted in tokens (e.g. 0.5M–4M tokens per step).',
    exam: 'If memory forces a smaller micro-batch, keep the effective batch with gradient accumulation — a very common exam answer.' },

  { id: 'overfitting', term: 'Overfitting', cat: 'Foundations',
    def: 'When a model fits its training data (including noise) at the expense of generalisation: training loss keeps falling while validation loss rises. Common in fine-tuning on small datasets with many epochs.',
    exam: 'Fixes: more/better data, fewer epochs, early stopping on validation loss, lower LR, dropout, or PEFT with a smaller rank.' },

  { id: 'parameters', term: 'Parameters (weights)', cat: 'Foundations',
    def: 'The learned numbers in a model. "7B" means about 7 billion. Memory for weights alone = params × bytes per param: 7B in bf16 ≈ 14 GB, in INT4 ≈ 3.5 GB.',
    exam: 'Be fast at this arithmetic — it is the basis of every "will it fit on this GPU" question.' },

  /* ---------------- Architecture ---------------- */
  { id: 'transformer', term: 'Transformer', cat: 'Architecture',
    def: 'The neural architecture introduced in "Attention Is All You Need" (2017): stacks of blocks combining multi-head self-attention and a position-wise feed-forward network, with residual connections and normalisation. Modern LLMs use the decoder-only variant.',
    exam: 'Know the three families — encoder-only (BERT), encoder-decoder (T5), decoder-only (GPT, Llama) — and which tasks each suits.' },

  { id: 'self-attention', term: 'Self-attention', cat: 'Architecture',
    def: 'Each token builds a query, key and value vector; attention weights = softmax(QKᵀ / √d) decide how much each token reads from every (earlier) token\'s value. Cost grows with the square of sequence length.',
    exam: 'Quadratic cost in sequence length motivates FlashAttention, sliding-window attention and KV caching.' },

  { id: 'embedding', term: 'Embedding', cat: 'Architecture',
    def: 'A learned lookup table mapping each token id to a dense vector (d_model wide). Also used loosely for a vector representation of a whole text, as in embedding models for retrieval.',
    exam: 'Do not confuse the LLM\'s input embedding matrix with a sentence-embedding model used in RAG.' },

  { id: 'context-window', term: 'Context window', cat: 'Architecture',
    def: 'The maximum number of tokens (prompt + generated) a model can attend to at once, e.g. 4k, 8k, 128k. Set by training and by the positional-encoding scheme.',
    exam: 'Long contexts cost memory (KV cache grows linearly) and compute (attention grows quadratically). Extending context needs RoPE scaling plus long-sequence training.' },

  { id: 'decoder-only', term: 'Decoder-only (causal) model', cat: 'Architecture',
    def: 'A transformer that uses causal (masked) self-attention so each position only sees earlier tokens, trained with next-token prediction. GPT, Llama, Mistral, Qwen, Nemotron are all decoder-only.',
    exam: 'The default answer for text generation and chat; encoder-only models are for classification/embeddings.' },

  /* ---------------- Data & tokenization ---------------- */
  { id: 'token', term: 'Token', cat: 'Data & tokenization',
    def: 'The unit an LLM reads and writes — usually a sub-word piece produced by a tokenizer. In English, 1 token ≈ 0.75 words ≈ 4 characters.',
    exam: 'Context limits, pricing, throughput (tokens/s) and training budgets are all counted in tokens.' },

  { id: 'tokenizer', term: 'Tokenizer', cat: 'Data & tokenization',
    def: 'The component that converts text to token ids and back, using a fixed vocabulary learned from data with an algorithm such as BPE, WordPiece or Unigram (SentencePiece).',
    exam: 'A model must always be used with the tokenizer it was trained with — swapping tokenizers silently destroys quality.' },

  /* ---------------- Training & GPUs ---------------- */
  { id: 'pretraining', term: 'Pretraining', cat: 'Training & GPUs',
    def: 'Training a model from random initialisation on a large unlabeled corpus with the next-token objective. It produces a "base" model with broad knowledge but no instruction-following behaviour.',
    exam: 'Pretraining is by far the most expensive stage. Continued pretraining on domain text is a lighter variant used for domain adaptation.' },

  { id: 'bf16', term: 'BF16 (bfloat16)', cat: 'Training & GPUs',
    def: '16-bit floating point with the same 8-bit exponent as fp32 but only 7 mantissa bits. Same dynamic range as fp32, so it rarely overflows and does not need loss scaling. The default training dtype on Ampere and newer GPUs.',
    exam: 'BF16 vs FP16: BF16 = range, no loss scaler; FP16 = more precision but needs dynamic loss scaling to avoid underflow/overflow.' },

  { id: 'fp16', term: 'FP16 (half precision)', cat: 'Training & GPUs',
    def: 'IEEE 16-bit float: 5 exponent bits, 10 mantissa bits. Narrow range (max ≈ 65,504), so mixed-precision training with fp16 needs a gradient (loss) scaler.',
    exam: 'Choose fp16 on pre-Ampere GPUs like V100/T4 that lack bf16 support.' },

  { id: 'fp8', term: 'FP8', cat: 'Training & GPUs',
    def: '8-bit floating point supported by Hopper and Blackwell Tensor Cores in two flavours: E4M3 (more precision, used for weights/activations) and E5M2 (more range, used for gradients). Used via NVIDIA Transformer Engine with per-tensor scaling.',
    exam: 'FP8 roughly doubles Tensor Core throughput vs BF16 on H100 and halves memory for those tensors. Know the E4M3/E5M2 split.' },

  { id: 'cuda', term: 'CUDA', cat: 'Training & GPUs',
    def: 'NVIDIA\'s parallel computing platform and programming model. PyTorch calls CUDA libraries (cuBLAS, cuDNN, NCCL) under the hood; `model.to("cuda")` places tensors in GPU memory.',
    exam: 'Most performance questions reduce to: is the GPU busy doing math (Tensor Cores), waiting on memory, or idle waiting on the CPU/data loader/communication?' },

  { id: 'tensor-core', term: 'Tensor Cores', cat: 'Training & GPUs',
    def: 'Specialised GPU units that perform small matrix multiply-accumulates in reduced precision (FP16/BF16/TF32/FP8/INT8/FP4) at many times the speed of regular CUDA cores.',
    exam: 'Using bf16/fp16/fp8 and dimensions that are multiples of 8 (or 64) is what lets matmuls hit Tensor Cores.' },

  { id: 'hbm', term: 'HBM (GPU memory)', cat: 'Training & GPUs',
    def: 'High-bandwidth memory stacked next to the GPU die: 80 GB at ~3.35 TB/s on H100, 141 GB on H200, 192 GB on B200. Holds weights, gradients, optimizer states, activations and KV cache.',
    exam: 'LLM decoding is usually memory-bandwidth bound: tokens/s ≈ HBM bandwidth ÷ bytes read per token.' },

  { id: 'nccl', term: 'NCCL', cat: 'Distributed training',
    def: 'The NVIDIA Collective Communications Library: optimised all-reduce, all-gather, reduce-scatter and broadcast across GPUs over NVLink, PCIe, InfiniBand and RoCE. PyTorch distributed uses it as the GPU backend.',
    exam: 'Data-parallel training all-reduces gradients with NCCL; FSDP/ZeRO uses all-gather + reduce-scatter. NCCL_DEBUG=INFO is the first troubleshooting step.' },

  /* ---------------- Fine-tuning & alignment ---------------- */
  { id: 'fine-tuning', term: 'Fine-tuning', cat: 'Fine-tuning & alignment',
    def: 'Continuing training of a pretrained model on a smaller, task- or domain-specific dataset so it adapts its behaviour. Can update all weights (full fine-tuning) or a small set of added parameters (PEFT).',
    exam: 'The exam expects you to pick the cheapest adequate method: prompting → RAG → PEFT → full fine-tuning → continued pretraining.' },

  { id: 'sft', term: 'SFT (supervised fine-tuning)', cat: 'Fine-tuning & alignment',
    def: 'Fine-tuning on (prompt, ideal response) pairs with the normal next-token loss, usually computed only on the response tokens. Turns a base model into an instruction-following model.',
    exam: 'SFT comes before preference alignment (RLHF/DPO) in the standard post-training pipeline.' },

  { id: 'peft', term: 'PEFT (parameter-efficient fine-tuning)', cat: 'Fine-tuning & alignment',
    def: 'Methods that freeze the base model and train a small number of new parameters: LoRA, QLoRA, adapters, prefix/prompt tuning, p-tuning, IA³. Also the name of Hugging Face\'s library implementing them.',
    exam: 'PEFT cuts trainable params to <1%, slashes optimizer memory, and lets one base model serve many task adapters.' },

  { id: 'lora', term: 'LoRA (low-rank adaptation)', cat: 'Fine-tuning & alignment',
    def: 'Freezes a weight matrix W and learns an update ΔW = B·A where A (r×d) and B (d×r) have a small rank r (e.g. 8–64). Output = Wx + (α/r)·BAx. B starts at zero so training begins exactly at the base model.',
    exam: 'Know rank r, alpha, target modules (q,k,v,o and MLP projections), that adapters can be merged into W for zero-latency inference, and that QLoRA = LoRA on a 4-bit frozen base.' },

  { id: 'rlhf', term: 'RLHF', cat: 'Fine-tuning & alignment',
    def: 'Reinforcement learning from human feedback: collect human preference comparisons, train a reward model on them, then optimise the policy LLM against that reward (classically with PPO) with a KL penalty to stay close to the SFT model.',
    exam: 'Contrast with DPO, which skips the reward model and RL loop by optimising directly on preference pairs.' },

  /* ---------------- Evaluation ---------------- */
  { id: 'perplexity', term: 'Perplexity', cat: 'Evaluation',
    def: 'exp(average cross-entropy) on held-out text — the effective number of tokens the model is "choosing between". Lower is better. Only comparable between models that share a tokenizer.',
    exam: 'Good for tracking pretraining and comparing checkpoints; poor at measuring instruction-following or helpfulness.' },

  /* ---------------- Prompting & RAG ---------------- */
  { id: 'rag', term: 'RAG (retrieval-augmented generation)', cat: 'Prompting & RAG',
    def: 'Retrieving relevant documents (via embeddings + a vector index, keyword search or both) at query time and inserting them into the prompt, so the model answers from fresh or private knowledge without retraining.',
    exam: 'RAG is the right answer when knowledge changes frequently or must be cited; fine-tuning is better for teaching style, format or a skill.' },

  { id: 'hallucination', term: 'Hallucination', cat: 'Prompting & RAG',
    def: 'Fluent output that is not supported by the input or by facts. Arises because the model optimises for plausible text, not truth.',
    exam: 'Mitigations: grounding with RAG and citations, lower temperature, instructing the model to say "I don\'t know", output verification, and guardrails.' },

  /* ---------------- Inference optimization ---------------- */
  { id: 'inference', term: 'Inference', cat: 'Inference optimization',
    def: 'Running a trained model to produce outputs. For LLMs it has two phases: prefill (process the whole prompt in parallel, compute-bound) and decode (generate one token at a time, memory-bandwidth-bound).',
    exam: 'Prefill drives time-to-first-token (TTFT); decode drives time-per-output-token (TPOT) and throughput.' },

  { id: 'kv-cache', term: 'KV cache', cat: 'Inference optimization',
    def: 'During generation, the keys and values of all previous tokens are stored so each new token only computes its own Q/K/V. Size = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes.',
    exam: 'The KV cache — not the weights — often limits batch size and context length in serving. GQA/MQA, FP8 KV cache and paged attention all attack it.' },

  { id: 'quantization', term: 'Quantization', cat: 'Inference optimization',
    def: 'Representing weights (and optionally activations and KV cache) with fewer bits — FP8, INT8, INT4, FP4 — to cut memory and increase speed, with a small accuracy cost.',
    exam: 'Know weight-only (W4A16: AWQ, GPTQ) vs weight-and-activation (W8A8: SmoothQuant, FP8) and PTQ vs QAT.' },

  /* ---------------- NVIDIA stack ---------------- */
  { id: 'nemo', term: 'NVIDIA NeMo', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s end-to-end framework for building generative AI models: NeMo Framework (pretraining and fine-tuning at scale on Megatron-Core), NeMo Curator (data), NeMo-RL/Aligner (alignment), NeMo Evaluator and NeMo Guardrails, plus NeMo microservices for customization.',
    exam: 'Expect to map each lifecycle stage to the right NeMo component.' },

  { id: 'megatron', term: 'Megatron-LM / Megatron-Core', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s library for training transformers at scale with tensor, pipeline, sequence, context and expert parallelism plus optimised kernels. NeMo Framework is built on Megatron-Core.',
    exam: 'The reference implementation behind "tensor parallelism" and "pipeline parallelism" questions.' },

  { id: 'tensorrt-llm', term: 'TensorRT-LLM', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s open-source library for optimising LLM inference: fused kernels, in-flight batching, paged KV cache, FP8/INT8/INT4/FP4 quantization, speculative decoding, and tensor/pipeline parallelism across GPUs.',
    exam: 'The default NVIDIA answer for "maximise inference throughput/latency on NVIDIA GPUs". It powers NIM and the Triton TensorRT-LLM backend.' },

  { id: 'triton', term: 'Triton Inference Server', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s open-source model server (now branded NVIDIA Dynamo-Triton): serves models from many frameworks (TensorRT-LLM, vLLM, PyTorch, ONNX…) over HTTP/gRPC with dynamic batching, model ensembles, concurrent model execution and Prometheus metrics.',
    exam: 'Know the model repository layout, config.pbtxt, dynamic batching and the metrics endpoint (port 8002). Not to be confused with OpenAI\'s Triton kernel language.' },

  { id: 'nim', term: 'NVIDIA NIM', cat: 'NVIDIA stack',
    def: 'Prebuilt, optimised inference microservice containers for popular models, exposing an OpenAI-compatible API. Each NIM picks an optimised engine (TensorRT-LLM, vLLM or SGLang) for the detected GPU automatically.',
    exam: 'The "fastest path to production on NVIDIA GPUs" answer; supports LoRA adapters and is part of NVIDIA AI Enterprise.' },

  { id: 'hugging-face', term: 'Hugging Face ecosystem', cat: 'Tools & frameworks',
    def: 'The de facto open-source LLM toolkit: `transformers` (models), `datasets`, `tokenizers`, `peft` (LoRA etc.), `trl` (SFT/DPO/GRPO trainers), `accelerate` (distributed), plus the Hub for sharing models and data.',
    exam: 'The labs in this course use it because it runs on a single consumer or Colab GPU; the same concepts map one-to-one onto NeMo.' },
]
