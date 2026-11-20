/** Glossary terms owned by Module 3 — GPU Acceleration and Optimization. */
export default [
  /* ---------------- Training & GPUs ---------------- */
  { id: 'optimizer-states', term: 'Optimizer states', cat: 'Training & GPUs',
    def: 'Per-parameter values an optimizer keeps between steps. AdamW stores a first moment (m) and second moment (v) in fp32; in mixed precision there is also an fp32 master copy of the weights — 12 bytes per parameter in total.',
    exam: 'The 12 of the "2 + 2 + 12 = 16 bytes per parameter" rule. ZeRO stage 1 shards exactly this.' },

  { id: 'activation-memory', term: 'Activation memory', cat: 'Training & GPUs',
    def: 'Memory for intermediate tensors saved in the forward pass so the backward pass can compute gradients. Roughly 34·s·b·h bytes per transformer layer in 16-bit with FlashAttention (plus 5·a·s²·b without it), so it scales with micro-batch, sequence length, hidden size and depth.',
    exam: 'Batch size and sequence length change activation memory, not weight/gradient/optimizer memory. Fix with smaller micro-batches + accumulation, checkpointing or FlashAttention.' },

  { id: 'mixed-precision', term: 'Mixed-precision training', cat: 'Training & GPUs',
    def: 'Running most matmuls in a 16-bit (or 8-bit) format on Tensor Cores while keeping master weights, optimizer states and sensitive reductions in fp32. In PyTorch: `torch.autocast` with bf16, or fp16 plus a GradScaler.',
    exam: 'About 2× faster math and half the activation memory with no accuracy loss when done right. bf16 needs no loss scaling; fp16 does.' },

  { id: 'tf32', term: 'TF32 (TensorFloat-32)', cat: 'Training & GPUs',
    def: 'A Tensor Core math mode on Ampere and newer GPUs that runs fp32 matmuls with fp32\'s 8-bit exponent but a 10-bit mantissa. Inputs and outputs stay fp32. Enabled in PyTorch with `torch.set_float32_matmul_precision("high")`.',
    exam: 'A free speed-up for remaining fp32 matmuls; not a storage format and not a replacement for bf16 mixed precision.' },

  { id: 'loss-scaling', term: 'Loss scaling (GradScaler)', cat: 'Training & GPUs',
    def: 'Multiplying the loss by a large factor before backward so small fp16 gradients do not underflow, then dividing the gradients back before the optimizer step. Dynamic scaling skips steps with inf/NaN gradients and lowers the scale, and raises it after a run of clean steps.',
    exam: 'Required for fp16 mixed precision, unnecessary for bf16. Unscale before gradient clipping.' },

  { id: 'gradient-accumulation', term: 'Gradient accumulation', cat: 'Training & GPUs',
    def: 'Running several micro-batches forward and backward, summing their gradients, and stepping the optimizer once. Effective batch = micro-batch × accumulation steps × data-parallel ranks. Divide each micro-batch loss by the number of accumulation steps.',
    exam: 'The standard answer to "keep the global batch size while reducing per-GPU memory". Costs time, not accuracy.' },

  { id: 'gradient-clipping', term: 'Gradient clipping', cat: 'Training & GPUs',
    def: 'Rescaling the full gradient vector when its global L2 norm exceeds a threshold (1.0 is the usual LLM default), so a single bad batch cannot produce a huge update. `torch.nn.utils.clip_grad_norm_` returns the pre-clip norm, which is worth logging.',
    exam: 'First-line defence against loss spikes; a rising gradient norm is an early warning of divergence.' },

  { id: 'activation-checkpointing', term: 'Activation (gradient) checkpointing', cat: 'Training & GPUs',
    def: 'Saving only selected activations (typically each transformer block\'s input) in the forward pass and recomputing the rest during backward. Full checkpointing costs about one extra forward pass (~30% more compute); selective checkpointing recomputes only cheap, large pieces such as attention internals.',
    exam: '"Trade compute for memory" in a question stem means activation checkpointing. It does not reduce weight or optimizer memory.' },

  { id: 'flash-attention', term: 'FlashAttention', cat: 'Training & GPUs',
    def: 'An exact attention algorithm that tiles Q, K and V into on-chip SRAM and computes softmax incrementally, never writing the s × s score matrix to HBM. Memory becomes linear in sequence length and attention becomes much faster. PyTorch SDPA dispatches to it; FlashAttention-3 targets Hopper and FP8.',
    exam: 'The answer for "attention memory grows quadratically with sequence length" and a prerequisite for long-context training and fast prefill.' },

  { id: 'torch-compile', term: 'torch.compile', cat: 'Tools & frameworks',
    def: 'PyTorch 2\'s compiler: TorchDynamo captures the Python model as a graph and TorchInductor generates fused GPU kernels (written in OpenAI Triton), mainly fusing memory-bound elementwise ops around the matmuls. First calls are slow while it compiles; shape changes can trigger recompilation.',
    exam: 'A one-line speed-up (often 1.2–2×) for training and inference; prefer static shapes (packed fixed-length sequences).' },

  { id: 'transformer-engine', term: 'Transformer Engine', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s library of transformer building blocks (te.Linear, te.LayerNormLinear, te.TransformerLayer) and an autocast context that runs GEMMs in FP8 on Hopper and Blackwell, managing per-tensor scaling (delayed scaling from an amax history) or block scaling (MXFP8 on Blackwell). Used inside Megatron-Core and NeMo.',
    exam: '"Train in FP8 on H100" → Transformer Engine, hybrid recipe: E4M3 forward, E5M2 gradients.' },

  { id: 'mfu', term: 'MFU (model FLOPs utilization)', cat: 'Training & GPUs',
    def: 'Achieved model FLOPs per second divided by the hardware\'s peak. For dense transformer training, achieved ≈ 6 × parameters × tokens/s (plus attention FLOPs at long context). HFU (hardware FLOPs utilization) additionally counts recomputation.',
    exam: 'Compute it: 6·N·(tokens/s per GPU) ÷ peak (H100 dense BF16 ≈ 989 TFLOP/s). Roughly 35–55% is good for large dense models.' },

  { id: 'oom', term: 'OOM (CUDA out of memory)', cat: 'Training & GPUs',
    def: 'The error raised when the caching allocator cannot satisfy a GPU allocation. The message reports allocated vs reserved memory; a large reserved-but-unallocated amount signals fragmentation (try `expandable_segments:True`).',
    exam: 'Fix ladder: bf16 → smaller micro-batch + accumulation → activation checkpointing → FlashAttention → leaner optimizer → ZeRO/FSDP sharding → offload → PEFT.' },

  /* ---------------- Distributed training ---------------- */
  { id: 'ddp', term: 'DDP (DistributedDataParallel)', cat: 'Distributed training',
    def: 'PyTorch data parallelism: each GPU (one process per GPU, launched by torchrun) holds a full model replica, processes a different data shard, and gradients are averaged with a bucketed all-reduce overlapped with the backward pass.',
    exam: 'Scales throughput, not memory — every rank still needs 16 bytes per parameter. Pair with DistributedSampler and `sampler.set_epoch()`.' },

  { id: 'fsdp', term: 'FSDP (Fully Sharded Data Parallel)', cat: 'Distributed training',
    def: 'PyTorch\'s native implementation of ZeRO-3-style sharding: parameters, gradients and optimizer states are sharded across data-parallel ranks; each unit\'s parameters are all-gathered just before use and gradients are reduce-scattered. FSDP2 (`fully_shard`) uses per-parameter DTensor sharding. Hybrid sharding shards inside a node and replicates across nodes.',
    exam: 'Memory per GPU ≈ 16Ψ/N for Ψ parameters on N GPUs, at about 1.5× DDP\'s communication volume.' },

  { id: 'zero', term: 'ZeRO (Zero Redundancy Optimizer)', cat: 'Distributed training',
    def: 'DeepSpeed\'s data-parallel memory optimisation in three stages: stage 1 shards optimizer states, stage 2 also shards gradients, stage 3 also shards parameters. ZeRO-Offload/Infinity move states to CPU or NVMe.',
    exam: 'Know what each stage shards and the per-GPU memory: 4Ψ + 12Ψ/N, 2Ψ + 14Ψ/N, 16Ψ/N. Stages 1–2 cost the same bandwidth as DDP.' },

  { id: 'all-reduce', term: 'All-reduce', cat: 'Distributed training',
    def: 'A collective in which every rank contributes a tensor and every rank receives the element-wise sum (or mean). A ring all-reduce is a reduce-scatter followed by an all-gather; each rank sends about 2(N−1)/N times the tensor size.',
    exam: 'DDP all-reduces gradients; tensor parallelism all-reduces layer outputs (twice per layer forward).' },

  { id: 'reduce-scatter', term: 'Reduce-scatter', cat: 'Distributed training',
    def: 'A collective that sums tensors across ranks and leaves each rank with only its 1/N slice of the result.',
    exam: 'How ZeRO-2/3 and FSDP produce sharded gradients, and how sequence parallelism replaces tensor-parallel all-reduces.' },

  { id: 'all-gather', term: 'All-gather', cat: 'Distributed training',
    def: 'A collective in which each rank contributes a slice and every rank ends up with the concatenated full tensor.',
    exam: 'ZeRO-3/FSDP all-gathers each layer\'s parameters before forward and again before backward.' },

  { id: 'nvlink', term: 'NVLink / NVSwitch', cat: 'Distributed training',
    def: 'NVIDIA\'s high-bandwidth GPU-to-GPU interconnect. NVLink 4 on H100 gives 900 GB/s per GPU; NVLink 5 on Blackwell 1.8 TB/s. NVSwitch connects all GPUs in a node (or 72 GPUs in a GB200 NVL72 rack) as one fast domain.',
    exam: 'Keep tensor parallelism inside the NVLink domain; it is roughly an order of magnitude faster than the inter-node network.' },

  { id: 'infiniband', term: 'InfiniBand (and RoCE)', cat: 'Distributed training',
    def: 'The low-latency, RDMA-capable network used between GPU nodes — e.g. 400 Gb/s NDR, usually one NIC per GPU, with GPUDirect RDMA letting NICs read GPU memory directly. RoCE provides RDMA over Ethernet (e.g. Spectrum-X).',
    exam: 'Inter-node traffic suits pipeline and data parallelism. NCCL_DEBUG=INFO shows whether NCCL is using IB or has fallen back to sockets.' },

  { id: 'tensor-parallelism', term: 'Tensor parallelism (TP)', cat: 'Distributed training',
    def: 'Splitting individual weight matrices across GPUs (Megatron style): column-parallel then row-parallel linears in the MLP, attention split by heads. Each GPU holds and computes 1/TP of every layer, with all-reduces of activations on the critical path.',
    exam: 'Use when a layer is too big or to cut per-GPU memory and compute; keep TP ≤ GPUs per NVLink domain (typically 8).' },

  { id: 'pipeline-parallelism', term: 'Pipeline parallelism (PP)', cat: 'Distributed training',
    def: 'Splitting the stack of layers into stages on different GPUs and streaming micro-batches through them, with point-to-point activation transfers between stages. Schedules include GPipe, 1F1B and interleaved 1F1B.',
    exam: 'Low communication, so it works across nodes, but suffers pipeline bubbles; needs many micro-batches.' },

  { id: 'pipeline-bubble', term: 'Pipeline bubble', cat: 'Distributed training',
    def: 'Idle time while a pipeline fills and drains. For p stages and m micro-batches the idle fraction is (p − 1)/(m + p − 1). 1F1B keeps the bubble but bounds activation memory to p micro-batches; interleaved stages divide it by the number of virtual stages.',
    exam: 'Bigger m (≥ 4p) or interleaving shrinks the bubble; fewer micro-batches make it worse.' },

  { id: 'sequence-parallelism', term: 'Sequence parallelism (SP)', cat: 'Distributed training',
    def: 'In Megatron, an add-on to tensor parallelism that shards the activations of LayerNorm/dropout regions along the sequence dimension within the TP group, replacing each all-reduce with a reduce-scatter + all-gather. Cuts activation memory at almost no extra cost. (DeepSpeed Ulysses uses the name for an all-to-all scheme over attention heads.)',
    exam: 'Turn it on whenever TP > 1. Different from context parallelism, which splits the sequence for every layer including attention.' },

  { id: 'context-parallelism', term: 'Context parallelism (CP)', cat: 'Distributed training',
    def: 'Splitting the sequence dimension across GPUs for the whole network, including attention; ranks exchange key/value blocks in a ring (ring attention) so each token can attend to the full context.',
    exam: 'The parallelism for very long context (e.g. 128k–1M tokens) where activations of one sequence do not fit on a GPU.' },

  { id: 'expert-parallelism', term: 'Expert parallelism (EP)', cat: 'Distributed training',
    def: 'Placing different Mixture-of-Experts experts on different GPUs. Tokens are dispatched to the GPUs holding their routed experts with an all-to-all and returned with another all-to-all.',
    exam: 'MoE scaling = EP combined with TP/PP/DP; all-to-all bandwidth and expert load balance are the bottlenecks.' },

  { id: 'hf-accelerate', term: 'Hugging Face Accelerate', cat: 'Tools & frameworks',
    def: 'A thin library that runs the same PyTorch training loop on CPU, one GPU, DDP, FSDP or DeepSpeed ZeRO depending on a config (`accelerate config`, `accelerate launch`), handling mixed precision and gradient accumulation. The Hugging Face Trainer uses it internally.',
    exam: 'The Hugging Face answer for "scale my script to multiple GPUs with minimal code change"; Megatron-Core/NeMo is the NVIDIA answer at large scale.' },

  /* ---------------- NVIDIA tools ---------------- */
  { id: 'nsight-systems', term: 'Nsight Systems (nsys)', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s system-wide profiler: a timeline of CPU threads, OS runtime, CUDA API calls, GPU kernels, memory copies, NCCL and NVTX ranges, across processes and GPUs. `nsys profile -t cuda,nvtx,osrt -o out python train.py`.',
    exam: 'Use it to find GPU idle gaps, data-loader stalls and whether communication overlaps compute.' },

  { id: 'nsight-compute', term: 'Nsight Compute (ncu)', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s kernel-level profiler: replays individual CUDA kernels to report occupancy, memory and compute throughput, roofline position and warp-stall reasons.',
    exam: 'Choose it to optimise one specific kernel; choose Nsight Systems for the whole application timeline.' },
]
