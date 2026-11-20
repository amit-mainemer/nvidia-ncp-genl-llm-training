// End-of-lesson comprehension checks for Module 7: Model Optimization. Keyed by lesson id.
export default {
  quantization: [
    {
      q: 'How much GPU memory do the weights of a 70B-parameter model need in FP8?',
      opts: ['About 35 GB', 'About 140 GB', 'About 70 GB', 'About 280 GB'],
      a: 2,
      why: 'FP8 is 1 byte per parameter: 70×10⁹ × 1 B = 70 GB (vs 140 GB in BF16).',
      distract: '35 GB is the 4-bit figure (before scale overhead). 140 GB is BF16/FP16. 280 GB is FP32.',
    },
    {
      q: 'What distinguishes AWQ from GPTQ?',
      opts: [
        'AWQ protects salient weight channels, identified by activation magnitude, by scaling them before quantization; GPTQ compensates quantization error using second-order (Hessian) information',
        'AWQ quantizes activations to INT8, while GPTQ quantizes only weights',
        'AWQ requires quantization-aware training, while GPTQ is post-training',
        'AWQ works only on Blackwell GPUs, while GPTQ works on any GPU',
      ],
      a: 0,
      why: 'Both are weight-only [[ptq|PTQ]] methods for ~4-bit weights. [[awq|AWQ]] uses activation-aware per-channel scaling with no reconstruction; [[gptq|GPTQ]] quantizes column by column and updates the remaining weights to absorb error.',
      distract: 'Neither quantizes activations (both are W4A16-style). Both are post-training. Both run on Ampere and later GPUs.',
    },
    {
      q: 'What does SmoothQuant do mathematically?',
      opts: [
        'It removes outlier tokens from the calibration data',
        'It keeps outlier channels in FP16 and the rest in INT8',
        'It replaces INT8 with FP8 for layers that have outliers',
        'It divides activations by a per-channel factor and multiplies the weights by the same factor, so X·W is unchanged but activation outliers shrink',
      ],
      a: 3,
      why: '[[smoothquant|SmoothQuant]] applies X·W = (X·diag(s)⁻¹)·(diag(s)·W) with s_j = max|X_j|^α / max|W_j|^(1−α), making both factors INT8-friendly. The scale is folded into the previous layer.',
      distract: 'It does not filter calibration data. Keeping outliers in FP16 is LLM.int8(). Switching formats is a different approach.',
    },
    {
      q: 'You serve a chat model with an INT4 AWQ weight-only checkpoint. At batch size 1 decode is much faster than BF16, but at batch 128 the gain almost disappears. Why?',
      opts: [
        'INT4 weights are dequantized to FP32, which is slower than BF16',
        'At large batch the GEMMs become compute-bound and still run in 16-bit after dequantization, so fewer weight bytes no longer matter much',
        'AWQ disables in-flight batching above batch 64',
        'The KV cache is automatically stored in INT4 at large batch, which is slow',
      ],
      a: 1,
      why: 'Weight-only quantization reduces bytes read per step, which helps memory-bound decode. With 128 sequences sharing each weight read, arithmetic intensity rises and the 16-bit compute becomes the bottleneck; W8A8 FP8 can then win.',
      distract: 'Dequantization targets FP16/BF16, not FP32. AWQ has no interaction with batching limits. The KV-cache format is a separate setting and is not changed automatically.',
    },
    {
      q: 'Your FP8 PTQ code assistant scores well on general benchmarks but much worse on code than the BF16 model. Calibration used 512 news articles. What should you try first?',
      opts: [
        'Switch to INT4 AWQ, which needs no calibration',
        'Increase the calibration set to 50,000 news articles',
        'Recalibrate with a representative sample of real code prompts',
        'Disable FP8 for the KV cache only',
      ],
      a: 2,
      why: '[[calibration|Calibration]] sets activation scales; code activations can have different ranges from news text. Calibrating on production-like data is the first fix.',
      distract: 'AWQ also uses calibration data to find salient channels. More off-domain data does not fix a domain mismatch. The KV cache is not the cause of a domain-specific drop.',
    },
  ],

  'pruning-distillation': [
    {
      q: 'You prune 50% of a model\'s weights with unstructured magnitude pruning and run it with standard dense kernels. What happens to latency?',
      opts: [
        'It halves, because half the multiplications are skipped',
        'It is essentially unchanged, because dense GEMMs still process the zeros',
        'It doubles, because sparse matrices are slower to read',
        'It halves only on Hopper GPUs',
      ],
      a: 1,
      why: 'Unstructured [[pruning|sparsity]] gives no speedup with dense kernels; you need structured pruning, 2:4 sparse kernels, or very high sparsity with special kernels.',
      distract: 'Dense kernels multiply zeros like any other value. Storing zeros densely does not make reads slower. Hopper has no general unstructured-sparsity acceleration.',
    },
    {
      q: 'Why can\'t you directly apply a logit-distillation KL loss between a Llama 3 teacher and a Qwen student?',
      opts: [
        'Because KL divergence is only defined for models of the same size',
        'Because Qwen models cannot be fine-tuned',
        'Because distillation requires the teacher to be smaller than the student',
        'Because their tokenizers and vocabularies differ, so their output distributions are over different token sets',
      ],
      a: 3,
      why: 'Logit [[distillation]] compares probability distributions position by position over the same vocabulary. Different tokenizers break that alignment; use sequence-level distillation (train on teacher outputs) instead.',
      distract: 'KL does not depend on model size. Qwen models are routinely fine-tuned. The teacher is normally the larger model.',
    },
    {
      q: 'In `F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="batchmean") * T * T`, which KL divergence is computed?',
      opts: [
        'KL(teacher ‖ student): the teacher distribution is the target, the student log-probabilities are the input',
        'KL(student ‖ teacher): the student is the target',
        'The cross-entropy between labels and student, scaled by T²',
        'The Jensen–Shannon divergence between student and teacher',
      ],
      a: 0,
      why: '`F.kl_div(input, target)` computes Σ target · (log target − input) = KL(target ‖ input-distribution). With target = teacher and input = student log-probs, this is forward KL(teacher ‖ student), the standard distillation term.',
      distract: 'The argument order makes the teacher the target, not the student. Hard labels do not appear in this expression. No mixture distribution is computed, so it is not Jensen–Shannon.',
    },
    {
      q: 'According to the Minitron results, how do width-pruned and depth-pruned students of the same size typically compare?',
      opts: [
        'Depth-pruned students are more accurate and faster',
        'They are identical in accuracy and speed',
        'Width-pruned students usually recover better accuracy, while depth-pruned students give larger latency gains',
        'Width pruning cannot be followed by distillation',
      ],
      a: 2,
      why: 'In [[minitron|Minitron]], width pruning (heads, neurons, channels) generally retained more accuracy after distillation; removing whole layers reduces sequential depth and so improves latency more.',
      distract: 'Depth pruning usually costs more accuracy. The trade-off is precisely that they differ. Both variants are followed by distillation.',
    },
    {
      q: 'A team must ship a 4B-class model for edge devices and has a strong 8B model. They have a modest token budget. What is the best plan?',
      opts: [
        'Structurally prune the 8B model to ~4B and distill from the 8B teacher, then optionally quantize',
        'Train a 4B model from scratch on the same trillions of tokens',
        'Apply 50% unstructured pruning to the 8B model without retraining',
        'Quantize the 8B model to INT4 and call it a 4B model',
      ],
      a: 0,
      why: 'Prune-then-distill reuses the 8B model\'s knowledge and needs far fewer tokens than pretraining; quantization can then cut memory further.',
      distract: 'Training from scratch needs a huge token budget. Unstructured pruning without retraining hurts accuracy and gives no dense speedup. INT4 reduces bytes, not parameter count or FLOPs per token.',
    },
  ],

  'inference-engine': [
    {
      q: 'Which phase of LLM inference primarily determines time to first token (TTFT), and what bounds it?',
      opts: [
        'Decode; memory bandwidth',
        'Detokenization; CPU speed',
        'Decode; network latency',
        'Prefill; compute (it processes all prompt tokens in parallel matrix–matrix multiplies)',
      ],
      a: 3,
      why: '[[prefill|Prefill]] processes the whole prompt before the first token can be emitted; with many tokens per weight read it is compute-bound.',
      distract: 'Decode determines time per output token. Detokenization and network add little compared with prefill for non-trivial prompts.',
    },
    {
      q: 'Llama 3.1 70B has 80 layers, 8 KV heads and head dim 128. What is its BF16 KV-cache size per token?',
      opts: ['128 KiB', '320 KiB', '640 KiB', '1.25 MiB'],
      a: 1,
      why: '2 × 80 × 8 × 128 × 2 bytes = 327,680 B = 320 KiB per token — so a 128k-token context needs ~40 GiB. See [[kv-cache]].',
      distract: '128 KiB is Llama 3.1 8B. 640 KiB would be an FP32 cache. 1.25 MiB would correspond to 32 KV heads (no GQA).',
    },
    {
      q: 'What problem does paged attention primarily solve?',
      opts: [
        'The quadratic cost of attention over long sequences',
        'Loss of accuracy from FP8 KV-cache quantization',
        'KV-cache memory waste from reserving contiguous max-length buffers per request and fragmentation',
        'Slow tokenization of long prompts',
      ],
      a: 2,
      why: '[[paged-attention|Paged attention]] allocates the KV cache in small blocks on demand, cutting waste from 60–80% to a few percent so far more sequences fit.',
      distract: 'Quadratic attention compute is addressed by FlashAttention-style kernels, not paging. Paging is format-agnostic. Tokenization is unrelated.',
    },
    {
      q: 'Requests in your workload have output lengths from 20 to 2,000 tokens. Compared with static batching, what does in-flight batching improve?',
      opts: [
        'GPU utilisation and throughput, because finished sequences are replaced by waiting requests at every iteration instead of the batch waiting for its longest member',
        'Model accuracy, because longer outputs get more compute',
        'TTFT for the first request only',
        'Memory use, because it disables the KV cache',
      ],
      a: 0,
      why: '[[inflight-batching|In-flight batching]] schedules at iteration granularity, eliminating idle slots caused by variable output lengths.',
      distract: 'Batching does not change model outputs. It benefits all requests, not just the first. The KV cache is still used — paged KV makes in-flight batching efficient.',
    },
    {
      q: 'Why does speculative decoding usually give smaller speedups at very high batch sizes?',
      opts: [
        'Because the draft model\'s tokenizer becomes slower',
        'Because at high batch decode is no longer memory-bound, so verifying extra draft tokens is no longer nearly free',
        'Because rejection sampling changes the output distribution at high batch',
        'Because the KV cache is disabled during verification',
      ],
      a: 1,
      why: '[[speculative-decoding|Speculative decoding]] exploits idle compute in memory-bound decode. At high batch the GPU is already busy, so extra verification work costs real time.',
      distract: 'Tokenizer speed is unaffected by batch size. Rejection sampling keeps the distribution exact at any batch. The KV cache is used throughout verification.',
    },
  ],

  'tensorrt-llm': [
    {
      q: 'In TensorRT-LLM, what does `max_num_tokens` limit?',
      opts: [
        'The total number of tokens (prefill chunks plus decode tokens of all scheduled requests) processed in one iteration',
        'The maximum output length of a single request',
        'The vocabulary size of the model',
        'The number of tokens stored in the KV cache across all requests',
      ],
      a: 0,
      why: '`max_num_tokens` is the per-iteration token budget. It bounds activation memory and, with chunked context, how much prefill can be packed into a step.',
      distract: 'Per-request length is governed by `max_seq_len` / sampling `max_tokens`. Vocabulary is a model property. KV capacity is set by the KV-cache memory fraction or token limits.',
    },
    {
      q: 'Which Triton model in the TensorRT-LLM reference repository actually runs the engine with in-flight batching?',
      opts: ['ensemble', 'preprocessing', 'tensorrt_llm', 'postprocessing'],
      a: 2,
      why: 'The `tensorrt_llm` model uses the tensorrtllm backend with `batching_strategy: inflight_fused_batching`. The ensemble wires preprocessing → tensorrt_llm → postprocessing.',
      distract: 'The ensemble only chains models. Preprocessing tokenizes and postprocessing detokenizes, both in the Python backend.',
    },
    {
      q: 'Your SLA forbids pausing in-progress requests, even at the cost of some throughput. Which batch scheduler policy fits?',
      opts: ['max_utilization', 'guaranteed_no_evict', 'inflight_fused_batching', 'decoupled'],
      a: 1,
      why: '`guaranteed_no_evict` only admits a request when KV blocks for its full length can be guaranteed, so no request is ever paused or evicted.',
      distract: '`max_utilization` admits aggressively and may pause requests. `inflight_fused_batching` is the batching strategy, not the scheduler policy. `decoupled` controls streaming.',
    },
    {
      q: 'Why should you benchmark TensorRT-LLM configurations with trtllm-bench on a dataset matching production input/output lengths?',
      opts: [
        'Because trtllm-bench refuses synthetic datasets',
        'Because engines must be rebuilt for every new dataset',
        'Because accuracy depends on benchmark length',
        'Because the optimal knobs (batch size, max_num_tokens, chunking, quantization) depend on the prefill/decode balance, which depends on lengths',
      ],
      a: 3,
      why: 'Long-prompt workloads are prefill/compute-heavy; short-prompt long-output workloads are decode/memory-heavy. The best configuration differs; see [[trtllm-bench]].',
      distract: 'trtllm-bench supports synthetic length distributions. Engines are not tied to a dataset. Throughput benchmarking does not measure accuracy.',
    },
    {
      q: 'Which statement best compares TensorRT-LLM and vLLM?',
      opts: [
        'Only vLLM supports paged attention and continuous batching',
        'Only TensorRT-LLM supports tensor parallelism',
        'vLLM requires compiling a TensorRT engine for each GPU',
        'Both implement paged KV, continuous batching and FP8; TensorRT-LLM is NVIDIA-optimised and integrated with ModelOpt, Triton and NIM, while vLLM emphasises broad model coverage and portability',
      ],
      a: 3,
      why: 'The core techniques are shared; the differences are in kernel optimisation for NVIDIA hardware, ecosystem integration and model coverage. Benchmark both on your workload.',
      distract: 'TensorRT-LLM has paged KV and in-flight batching. vLLM supports tensor parallelism. vLLM does not compile TensorRT engines.',
    },
  ],
}
