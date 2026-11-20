/**
 * Model Optimization glossary — terms owned by Module 7.
 * Shared terms (quantization, kv-cache, fp8, bf16, fp16, tensorrt-llm, triton…) live in core.js.
 */
export default [
  { id: 'ptq', term: 'PTQ (post-training quantization)', cat: 'Inference optimization',
    def: 'Quantizing an already-trained model without further training: compute scales (and zero points) from the weights and from activation statistics on a small calibration set, then convert. GPTQ, AWQ, SmoothQuant and FP8 calibration are all PTQ.',
    exam: 'PTQ is the default first move — minutes to hours, no training data. Escalate to QAT only if PTQ accuracy is unacceptable (typically at 4 bits or below).' },

  { id: 'qat', term: 'QAT (quantization-aware training)', cat: 'Inference optimization',
    def: 'Fine-tuning with fake-quantization in the forward pass (quantize → dequantize) and a straight-through estimator in the backward pass, so the weights adapt to the rounding error of the target format.',
    exam: 'Recovers more accuracy than PTQ at aggressive bit widths (INT4, FP4) at the cost of a training run. ModelOpt and NeMo support QAT and QAD (QAT with distillation).' },

  { id: 'int8', term: 'INT8', cat: 'Inference optimization',
    def: '8-bit integer format with 256 levels (−128…127 signed). Used for weights (W8A16), weights and activations (W8A8 via SmoothQuant) and KV cache. Supported by Tensor Cores since Turing with ~2× BF16 math throughput.',
    exam: 'Halves weight memory vs FP16. W8A8 INT8 needs outlier handling (SmoothQuant); on Hopper/Blackwell FP8 is usually preferred because it is easier to calibrate.' },

  { id: 'int4', term: 'INT4', cat: 'Inference optimization',
    def: '4-bit integer format with 16 levels (−8…7). Used almost exclusively for weights (W4A16) with per-group scales (group size 64–128), via GPTQ or AWQ; the weights are dequantized to FP16/BF16 inside the GEMM kernel.',
    exam: 'Cuts weight memory ~4× (≈4.1–4.5 effective bits with scales) and speeds up memory-bound decode at small batch; gains shrink at large batch where the GEMMs become compute-bound.' },

  { id: 'fp4', term: 'FP4 / NVFP4', cat: 'Inference optimization',
    def: '4-bit floating point (E2M1: representable magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6). NVIDIA\'s NVFP4 adds an FP8 (E4M3) scale per 16-value block plus a per-tensor FP32 scale; the OCP MXFP4 variant uses a power-of-two scale per 32-value block. Native on Blackwell Tensor Cores.',
    exam: 'Blackwell-only Tensor Core format for weights and activations: ~4.5 bits per value with NVFP4 block scales, roughly 2× FP8 math throughput. Fine-grained block scaling is what keeps accuracy acceptable.' },

  { id: 'awq', term: 'AWQ (activation-aware weight quantization)', cat: 'Inference optimization',
    def: 'A weight-only INT4 PTQ method: about 1% of weight channels matter most, and they are identified by activation magnitude on calibration data. AWQ scales those channels up before quantization (and the matching activations down) so they lose less precision; no backprop or weight reconstruction is needed.',
    exam: 'W4A16, fast to apply, robust across domains because it does not overfit the calibration set. Available as `INT4_AWQ` in TensorRT Model Optimizer.' },

  { id: 'gptq', term: 'GPTQ', cat: 'Inference optimization',
    def: 'A one-shot weight-only PTQ method that quantizes each layer column by column and uses approximate second-order (Hessian) information from calibration activations to update the remaining weights, compensating the error introduced so far.',
    exam: 'Strong 3–4-bit accuracy; slower to apply than AWQ and more sensitive to the calibration data. Both are weight-only — activations stay in FP16/BF16.' },

  { id: 'smoothquant', term: 'SmoothQuant', cat: 'Inference optimization',
    def: 'A W8A8 PTQ technique that migrates activation outliers into the weights with a per-input-channel factor s_j = max|X_j|^α / max|W_j|^(1−α) (α ≈ 0.5): X·W = (X/s)·(s·W). The smoothed activations and the slightly harder weights both fit INT8.',
    exam: 'The answer whenever a question says "INT8 activations lose accuracy because of outlier channels". It is mathematically equivalent before quantization — the scale is folded into the previous layer.' },

  { id: 'calibration', term: 'Calibration (quantization)', cat: 'Inference optimization',
    def: 'Running a small, representative dataset (typically 128–512 samples) through the model to record activation ranges (max, percentile, MSE-optimal) from which static quantization scales are computed.',
    exam: 'Calibration data should look like production traffic. Too few or off-domain samples give bad activation scales — a classic cause of PTQ accuracy loss.' },

  { id: 'quant-granularity', term: 'Quantization granularity (per-tensor / channel / group)', cat: 'Inference optimization',
    def: 'How many values share one scale: per-tensor (one scale), per-channel (one per output row of a weight matrix), per-token (one per activation row), or per-group/block (one per 16–128 consecutive values). Finer granularity tracks outliers better at a small storage cost.',
    exam: 'INT4 weight-only needs per-group scales (e.g. group 128 → +0.125 bits per weight for an FP16 scale). FP8 on Hopper typically uses per-tensor scales; NVFP4 uses 16-value blocks.' },

  { id: 'kv-cache-quantization', term: 'KV-cache quantization', cat: 'Inference optimization',
    def: 'Storing cached keys and values in FP8 or INT8 (sometimes INT4) instead of FP16/BF16, with per-tensor or per-head scales from calibration. Halves (or quarters) KV memory, allowing larger batches and longer contexts.',
    exam: 'Independent of weight quantization: an FP16-weight model can still use an FP8 KV cache. The win is capacity and memory bandwidth in long-context, high-concurrency serving.' },

  { id: 'pruning', term: 'Pruning', cat: 'Inference optimization',
    def: 'Removing parameters judged unimportant. Unstructured pruning zeroes individual weights (needs sparse kernels to speed up); structured pruning removes whole units — attention heads, MLP neurons, embedding channels (width) or entire layers (depth) — producing a smaller dense model.',
    exam: 'Only structured pruning (or hardware-supported 2:4 sparsity) gives real speedups on GPUs. Pruned models need retraining — ideally distillation — to recover accuracy.' },

  { id: 'structured-sparsity', term: '2:4 structured sparsity', cat: 'Inference optimization',
    def: 'A fine-grained sparsity pattern where at most 2 of every 4 consecutive weights are non-zero. Ampere and later Sparse Tensor Cores skip the zeros, storing half the values plus 2-bit indices, for up to 2× math throughput on those GEMMs.',
    exam: '50% sparsity in a pattern the hardware understands. Real end-to-end LLM speedups are smaller than 2× and accuracy usually needs fine-tuning to recover.' },

  { id: 'distillation', term: 'Knowledge distillation', cat: 'Inference optimization',
    def: 'Training a smaller student to match a larger teacher\'s output distribution (soft targets) rather than only the one-hot labels: loss = α·CE(labels) + (1−α)·τ²·KL(teacher_τ ‖ student_τ), where τ is a softmax temperature that exposes "dark knowledge" in the non-top probabilities.',
    exam: 'Soft targets carry far more information per example than hard labels, so distillation needs much less data than training from scratch. Know the τ² factor and the direction of the KL.' },

  { id: 'minitron', term: 'Minitron', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s recipe for compressing LLMs by structured pruning (width: heads, MLP neurons, embedding channels; or depth: layers) using activation-based importance, followed by knowledge distillation from the original model. Examples: Minitron 8B/4B from Nemotron-4 15B, Llama-3.1-Minitron 4B from Llama 3.1 8B, Mistral-NeMo-Minitron 8B from Mistral NeMo 12B.',
    exam: 'Key claim: a family of smaller models from one large model with up to ~40× fewer training tokens per model than training from scratch. Width pruning generally recovered better than depth pruning.' },

  { id: 'prefill', term: 'Prefill (context phase)', cat: 'Inference optimization',
    def: 'The first phase of generation: the whole prompt is processed in one parallel forward pass, filling the KV cache and producing the first output token. Large matrix–matrix multiplies make it compute-bound.',
    exam: 'Prefill dominates time-to-first-token (TTFT). Long prompts, chunked prefill and prefix caching are all about managing prefill cost.' },

  { id: 'decode', term: 'Decode (generation phase)', cat: 'Inference optimization',
    def: 'The autoregressive phase: one new token per sequence per step, each step reading all weights and the sequence\'s KV cache from HBM to do matrix–vector-like work. Memory-bandwidth-bound at small batch.',
    exam: 'Decode dominates time-per-output-token. Batching, weight quantization, KV-cache compression and speculative decoding attack it.' },

  { id: 'arithmetic-intensity', term: 'Arithmetic intensity (roofline)', cat: 'Inference optimization',
    def: 'FLOPs performed per byte moved from memory. A kernel is memory-bound when its intensity is below the GPU\'s ridge point (peak FLOP/s ÷ memory bandwidth) — about 990 TFLOPS ÷ 3.35 TB/s ≈ 300 FLOPs/byte for dense BF16 on an H100 SXM.',
    exam: 'Batch-1 decode with FP16 weights is ~1 FLOP/byte — hundreds of times below the ridge — which is why batching and weight quantization help decode so much.' },

  { id: 'paged-attention', term: 'Paged attention (paged KV cache)', cat: 'Inference optimization',
    def: 'Storing each sequence\'s KV cache in fixed-size blocks (e.g. 16–64 tokens) allocated on demand from a shared pool and addressed through a block table, like virtual-memory pages. Introduced by vLLM; TensorRT-LLM calls it the paged KV cache.',
    exam: 'Eliminates the fragmentation and over-reservation of contiguous per-request max-length buffers (the vLLM paper reports 60–80% waste before, under 4% after), so many more requests fit — and it enables block-level prefix sharing.' },

  { id: 'inflight-batching', term: 'In-flight (continuous) batching', cat: 'Inference optimization',
    def: 'Iteration-level scheduling: after every decode step, finished sequences leave the batch and waiting requests join immediately, instead of the whole batch waiting for its longest member. TensorRT-LLM calls it in-flight batching; vLLM and others call it continuous batching.',
    exam: 'The single biggest throughput win in LLM serving versus static/dynamic request batching, because output lengths vary wildly. In Triton\'s TensorRT-LLM backend it is `batching_strategy: inflight_fused_batching`.' },

  { id: 'chunked-prefill', term: 'Chunked prefill (chunked context)', cat: 'Inference optimization',
    def: 'Splitting a long prompt\'s prefill into chunks (bounded by a per-step token budget such as `max_num_tokens`) that are scheduled alongside ongoing decode steps.',
    exam: 'Prevents one long prompt from stalling every other user\'s token stream (TPOT spikes) and bounds activation memory. Trade-off: slightly higher TTFT for that long request.' },

  { id: 'prefix-caching', term: 'Prefix caching (KV-cache reuse)', cat: 'Inference optimization',
    def: 'Keeping KV-cache blocks of previously processed prefixes (system prompts, few-shot examples, earlier chat turns) and reusing them for new requests that start with the same tokens, skipping their prefill.',
    exam: 'Cuts TTFT and prefill compute for shared-prefix workloads. Called automatic prefix caching in vLLM and KV cache block reuse (`enable_block_reuse` / `enable_kv_cache_reuse`) in TensorRT-LLM.' },

  { id: 'speculative-decoding', term: 'Speculative decoding', cat: 'Inference optimization',
    def: 'A cheap drafter (small model, extra heads such as Medusa/EAGLE, or n-gram lookup) proposes γ tokens; the target model verifies them all in one forward pass and accepts the longest correct prefix plus one token of its own. With rejection sampling the output distribution is identical to the target model\'s.',
    exam: 'Lossless latency reduction for memory-bound decode. Expected tokens per target pass = (1 − α^(γ+1)) / (1 − α) for acceptance rate α. Benefit shrinks at large batch, where decode is no longer memory-bound.' },

  { id: 'eagle-medusa', term: 'Medusa and EAGLE', cat: 'Inference optimization',
    def: 'Self-speculative decoding methods that avoid a separate draft model. Medusa adds several extra LM heads that each predict a future token, verified with tree attention. EAGLE trains a small autoregressive head on the target model\'s hidden features, giving higher acceptance rates (EAGLE-2/3 add dynamic draft trees and multi-layer features).',
    exam: 'Both need a short training run for the heads but no separate model to serve. TensorRT-LLM supports draft-model, Medusa, EAGLE and n-gram/lookahead speculation.' },

  { id: 'cuda-graphs', term: 'CUDA Graphs', cat: 'Inference optimization',
    def: 'Capturing a sequence of GPU kernel launches once and replaying it as a single graph launch, removing per-kernel CPU launch overhead. Inference engines capture one graph per batch size (padding to the nearest captured size).',
    exam: 'Matters most for decode at small batch with small models, where hundreds of tiny kernels per step make the CPU the bottleneck.' },

  { id: 'model-optimizer', term: 'TensorRT Model Optimizer (ModelOpt)', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s library (`nvidia-modelopt`, imported as `modelopt`) for PTQ (FP8, NVFP4, INT8 SmoothQuant, INT4 AWQ, KV-cache quantization), QAT/QAD, pruning (including Minitron-style), distillation and speculative-decoding head training. Exports checkpoints for TensorRT-LLM, vLLM and SGLang.',
    exam: 'The NVIDIA tool for producing quantized/pruned checkpoints; TensorRT-LLM is the tool that runs them. `mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)` is the core call.' },

  { id: 'trtllm-build', term: 'trtllm-build (engine build)', cat: 'NVIDIA stack',
    def: 'The TensorRT-LLM CLI that compiles a TensorRT-LLM checkpoint into a serialized TensorRT engine for a fixed GPU type, TP/PP layout and set of limits (max batch size, input/sequence length, max tokens per step) and plugins. Part of the TensorRT backend; the newer PyTorch backend skips the build step.',
    exam: 'Engines are not portable across GPU architectures and are bounded by the build-time limits — requests beyond `max_seq_len` or batches beyond `max_batch_size` cannot be served without a rebuild.' },

  { id: 'trtllm-bench', term: 'trtllm-bench', cat: 'NVIDIA stack',
    def: 'TensorRT-LLM\'s benchmarking CLI: measures offline throughput and latency for a model on a synthetic or real dataset of input/output lengths, reporting tokens/s, request throughput and latency percentiles.',
    exam: 'Use it (or genai-perf against a running server) to compare tuning knobs on your real length distribution — never tune on a single prompt.' },
]
