/* Module 3 — GPU Acceleration and Optimization (official domain 7, 14%).
   The "how do you actually train an LLM" module: memory, precision, a pretraining lab,
   distributed parallelism and performance debugging. */

/* ------------------------------------------------------------------ */
const memoryMath = {
  id: 'memory-math',
  title: 'GPU memory math: weights, gradients, optimizer states, activations',
  minutes: 20,
  outcomes: [
    'Calculate the bytes per parameter needed for weights, gradients and AdamW states in mixed-precision training',
    'Estimate activation memory from batch size, sequence length, hidden size and layer count',
    'Decide whether a given model can be trained or only served on a given GPU',
    'Distinguish training memory from inference memory, including the KV cache',
    'Write a small Python estimator that answers "will it fit?" before you launch a job',
  ],
  blocks: [
    { t: 'lede', x: 'Almost every GPU question — on the exam and in real life — starts with "will it fit?". Before you pick a parallelism strategy, a batch size or a GPU, you should be able to do the memory arithmetic in your head: **how many bytes does each parameter cost, and how much do activations add on top?** This lesson gives you the formulas, three worked examples (1B, 7B, 70B) and a script that does it for you.' },

    { t: 'h2', x: 'Four things live in GPU memory during training' },
    { t: 'p', x: 'When you call `loss.backward()` and `optimizer.step()`, the GPU\'s [[hbm|HBM]] holds four kinds of tensor. The first three scale with the number of [[parameters|parameters]] and are fixed for a given model and optimizer. The fourth scales with how much data you push through at once.' },
    { t: 'table', head: ['What', 'Why it exists', 'Scales with', 'Typical bytes'], rows: [
      ['Weights', 'The model itself, used in forward and backward', 'params', '2 per param in [[bf16|bf16]] (4 in fp32)'],
      ['Gradients', '∂loss/∂w for every trainable weight, produced by [[backprop|backprop]]', 'trainable params', '2 per param in bf16 (4 in fp32)'],
      ['[[optimizer-states|Optimizer states]]', '[[adamw|AdamW]] keeps a running mean (m) and variance (v) per parameter, plus an fp32 master copy in mixed precision', 'trainable params', '12 per param (4 + 4 + 4)'],
      ['[[activation-memory|Activations]]', 'Intermediate results saved in the forward pass because [[autograd|autograd]] needs them for the backward pass', 'batch × seq × hidden × layers', 'Varies — can exceed everything else'],
    ]},
    { t: 'p', x: 'On top of that there are smaller costs you should budget for but not compute exactly: the CUDA context (~0.5–1 GB per process), temporary buffers inside kernels, communication buffers for [[nccl|NCCL]], and fragmentation in PyTorch\'s caching allocator. A safe habit is to leave **10–20% headroom**.' },

    { t: 'h2', x: 'The 16-bytes-per-parameter rule' },
    { t: 'p', x: 'Standard [[mixed-precision|mixed-precision]] training with AdamW keeps the working weights and gradients in 16-bit, but keeps everything the optimizer touches in fp32 so tiny updates are not rounded away. Add it up:' },
    { t: 'specs', rows: [
      ['bf16 weights', '2 bytes'],
      ['bf16 gradients', '2 bytes'],
      ['fp32 master weights', '4 bytes'],
      ['fp32 Adam first moment (m)', '4 bytes'],
      ['fp32 Adam second moment (v)', '4 bytes'],
      ['Total ("2 + 2 + 12")', '**16 bytes per parameter**, before activations'],
    ]},
    { t: 'p', x: 'This is the number the [[zero|ZeRO]] paper uses and the one the exam expects. Variants you will meet: pure fp32 training with Adam is also 16 bytes (4 + 4 + 8) but slower; some frameworks keep gradients in fp32 (18 bytes); 8-bit Adam (bitsandbytes) cuts the two moments to ~2 bytes total; SGD with momentum needs only one state. For inference, only the weights are needed: **2 bytes per parameter in bf16**, 1 in FP8/INT8, ~0.5 in INT4.' },
    { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
      '"How much memory to fully fine-tune a 7B model with AdamW in mixed precision, excluding activations?" → 7B × 16 B = **112 GB** — more than one 80 GB H100.',
      '"How much to serve it in bf16?" → 7B × 2 B = **14 GB** of weights, plus the [[kv-cache|KV cache]].',
      'Watch the units: 1B params × 1 byte = 1 GB (10⁹ bytes). Exams usually use decimal GB; `nvidia-smi` reports MiB.',
    ]},

    { t: 'h2', x: 'Worked examples: 1B, 7B and 70B' },
    { t: 'table', head: ['Model', 'Inference weights (bf16)', 'Training states (16 B/param)', 'What it takes to train'], rows: [
      ['1B', '2 GB', '16 GB', 'One 24 GB GPU with small micro-batches; comfortable on a 40/80 GB GPU'],
      ['7B', '14 GB', '112 GB', 'Does not fit on one 80 GB GPU. Shard states with [[fsdp|FSDP]]/[[zero|ZeRO-3]] over 2–8 GPUs, or use [[lora|LoRA]]/QLoRA to avoid most of it'],
      ['70B', '140 GB', '1,120 GB (1.12 TB)', 'At least ~16 H100s just for states; in practice 32–64+ GPUs with tensor + pipeline + data parallelism'],
    ]},
    { t: 'p', x: 'Notice how PEFT changes the picture. With [[lora|LoRA]] on a frozen 7B base, the base weights still cost 14 GB in bf16, but gradients and optimizer states exist only for the adapter (often <1% of parameters), so the total drops to roughly 15–20 GB plus activations. That is why "fine-tune a 7B model on a single 24 GB GPU" almost always means LoRA or [[qlora|QLoRA]], not full fine-tuning.' },

    { t: 'h2', x: 'Activation memory: the part that grows with your batch' },
    { t: 'p', x: 'Every layer saves tensors during the forward pass for use in the backward pass: the inputs to each matmul, the attention probabilities, the MLP intermediate, normalisation statistics. Korthikanti et al. (the Megatron "Reducing Activation Recomputation" paper) derived a widely used estimate for a GPT-style layer in 16-bit precision:' },
    { t: 'code', lang: 'text', title: 'Activation bytes per transformer layer (16-bit, no recomputation)', x: `activations_per_layer ≈ s · b · h · (34 + 5 · a · s / h)

  s = sequence length    b = micro-batch size
  h = hidden size        a = number of attention heads

  34·s·b·h      → linear-layer inputs, MLP intermediate, norms, dropout masks
  5·a·s²·b      → the s × s attention score/probability matrices

With FlashAttention / SDPA the s × s matrices are never stored,
so the estimate drops to ≈ 34 · s · b · h bytes per layer.
Total ≈ that × number of layers L.` },
    { t: 'p', x: 'The constants assume a classic 4h MLP; a Llama-style SwiGLU MLP and [[gqa|GQA]] change them a little, but the scaling is what matters: **activations grow linearly with micro-batch, sequence length, hidden size and depth** — and quadratically with sequence length if attention is materialised. Worked example for a 7B-class model (h = 4096, L = 32, a = 32) at s = 4096, b = 1 with [[flash-attention|FlashAttention]]: 34 × 4096 × 1 × 4096 × 32 ≈ **18 GB**. Without FlashAttention the attention term adds 5 × 32 × 4096² × 32 ≈ 86 GB more — which is why nobody trains long contexts without it.' },
    { t: 'p', x: 'The fix when activations do not fit is [[activation-checkpointing|activation checkpointing]]: keep only each layer\'s input (2·s·b·h bytes per layer, ≈1 GB total in the example above) and recompute the rest during the backward pass, at the cost of roughly one extra forward pass (~30% more compute). You will use it in the next lesson.' },
    { t: 'callout', kind: 'trap', title: 'Batch size does not change state memory', x: 'Halving the micro-batch halves activation memory but does **nothing** to weights, gradients or optimizer states. If a model\'s 16 bytes/param already exceed the GPU, no batch size will make it fit — you need sharding (FSDP/ZeRO), model parallelism, or PEFT. Questions often offer "reduce the batch size" as a tempting wrong answer for exactly this case.' },

    { t: 'h2', x: 'Inference is different: weights plus the KV cache' },
    { t: 'p', x: 'Serving does not need gradients or optimizer states, and it does not keep activations for a backward pass. What it does keep is the [[kv-cache|KV cache]]: the keys and values of every previous token in every layer, so each new token only computes its own. Its size is 2 × layers × kv_heads × head_dim × seq_len × batch × bytes. For Llama-3-8B (32 layers, 8 KV heads with GQA, head_dim 128) in bf16 that is 2 × 32 × 8 × 128 × 2 B = **128 KB per token** — a 32-request batch at 8k context needs ~34 GB of KV cache, more than double the 16 GB of weights. The inference module goes deeper; here just remember that "will it fit for serving?" means weights + KV cache, not 16 bytes/param.' },

    { t: 'h2', x: 'A memory estimator you can trust' },
    { t: 'p', x: 'Turn the formulas into a script and run it before every new configuration. It is deliberately simple — it ignores framework overhead, so compare its answer against your GPU size minus ~15%.' },
    { t: 'code', lang: 'python', title: 'mem_estimate.py — will it fit?', x: `"""Rough GPU memory estimate for training / serving a decoder-only transformer."""

GB = 1e9

def param_count(vocab, hidden, layers, intermediate, n_heads, n_kv_heads, tied=True):
    head_dim = hidden // n_heads
    attn = hidden * hidden * 2 + 2 * hidden * (n_kv_heads * head_dim)   # q, o + k, v (GQA-aware)
    mlp = 3 * hidden * intermediate                                     # SwiGLU: gate, up, down
    embed = vocab * hidden * (1 if tied else 2)                         # embedding (+ LM head)
    return layers * (attn + mlp) + embed

def training_gb(params, bytes_per_param=16, shards=1):
    """Weights + grads + AdamW states. shards = data-parallel ranks for ZeRO-3/FSDP full sharding."""
    return params * bytes_per_param / shards / GB

def activation_gb(seq, micro_bs, hidden, layers, n_heads, flash=True, checkpointing=False):
    if checkpointing:                       # keep only each layer's input (bf16)
        return 2 * seq * micro_bs * hidden * layers / GB
    per_layer = 34 * seq * micro_bs * hidden
    if not flash:
        per_layer += 5 * n_heads * seq * seq * micro_bs
    return per_layer * layers / GB

def kv_cache_gb(layers, n_kv_heads, head_dim, seq, batch, bytes_=2):
    return 2 * layers * n_kv_heads * head_dim * seq * batch * bytes_ / GB

if __name__ == "__main__":
    # A Llama-3-8B-like shape
    cfg = dict(vocab=128256, hidden=4096, layers=32, intermediate=14336, n_heads=32, n_kv_heads=8)
    p = param_count(*cfg.values(), tied=False)   # positional, in dict order
    print(f"params            : {p/1e9:.2f} B")
    print(f"serve weights bf16: {p*2/GB:.1f} GB")
    print(f"train states      : {training_gb(p):.0f} GB on 1 GPU, {training_gb(p, shards=8):.1f} GB/GPU with FSDP on 8")
    for ckpt in (False, True):
        a = activation_gb(4096, 1, cfg['hidden'], cfg['layers'], cfg['n_heads'], checkpointing=ckpt)
        print(f"activations s=4096 b=1 checkpointing={ckpt}: {a:.1f} GB")
    print(f"KV cache, 32 seqs x 8k tokens: {kv_cache_gb(32, 8, 128, 8192, 32):.1f} GB")` },
    { t: 'p', x: 'For the Llama-3-8B shape this prints about 8.0 B parameters, 16 GB to serve, 128 GB of training states on one GPU (16 GB/GPU sharded over 8), ~18 GB of activations at 4k context (~1 GB with checkpointing) and ~34 GB of KV cache for 32 × 8k-token requests. Hugging Face\'s `accelerate estimate-memory <model-id>` command gives a similar quick answer for Hub models.' },

    { t: 'h2', x: 'Key takeaways' },
    { t: 'ul', items: [
      'Mixed-precision AdamW training costs **16 bytes per parameter** (2 weights + 2 grads + 12 optimizer/master) before activations; inference in bf16 costs 2 bytes per parameter.',
      '1B → 16 GB, 7B → 112 GB, 70B → 1.12 TB of training states. Serving: 2, 14 and 140 GB of weights.',
      'Activations ≈ 34·s·b·h bytes per layer with FlashAttention; they scale with batch and sequence, and [[activation-checkpointing|checkpointing]] trades ~30% compute to shrink them to the layer inputs.',
      'Batch size only affects activations. If the states do not fit, you need sharding, model parallelism or PEFT.',
      'Serving memory = weights + [[kv-cache|KV cache]]; the KV cache often dominates at large batch and long context.',
    ]},
  ],
}

/* ------------------------------------------------------------------ */
const mixedPrecision = {
  id: 'mixed-precision',
  title: 'Mixed precision, gradient accumulation, checkpointing and FlashAttention',
  minutes: 22,
  outcomes: [
    'Configure bf16 and fp16 mixed precision in PyTorch, including loss scaling for fp16',
    'Implement gradient accumulation and gradient clipping correctly in a training loop',
    'Apply activation checkpointing, SDPA/FlashAttention and torch.compile to cut memory and time',
    'Explain how FP8 training works with Transformer Engine and when it pays off',
    'Choose the right technique for a given memory or speed bottleneck',
  ],
  blocks: [
    { t: 'lede', x: 'The previous lesson told you how much memory a training run needs. This one gives you the six levers every LLM trainer pulls to make it fit and run fast: **mixed precision, gradient accumulation, gradient clipping, activation checkpointing, fused attention and compilation** — plus FP8 for Hopper and Blackwell GPUs. All of them are a few lines of PyTorch.' },

    { t: 'h2', x: 'Mixed precision: bf16, fp16 and TF32' },
    { t: 'p', x: '[[mixed-precision|Mixed precision]] runs the heavy matmuls on [[tensor-core|Tensor Cores]] in a 16-bit format while keeping numerically sensitive pieces (master weights, optimizer states, reductions like softmax and loss) in fp32. You get ~2× the math throughput and half the activation memory with essentially no accuracy loss. In PyTorch this is `torch.autocast`, which picks a dtype per operation.' },
    { t: 'table', head: ['Format', 'Bits (exp/mantissa)', 'Range', 'Needs loss scaling?', 'Use it when'], rows: [
      ['fp32', '8 / 23', '~1e±38', 'No', 'Master weights, optimizer states, reductions'],
      ['[[tf32|TF32]]', '8 / 10 (19 bits used)', 'Same as fp32', 'No', 'Free speed-up for fp32 matmuls on Ampere+; opt-in in PyTorch'],
      ['[[bf16|bf16]]', '8 / 7', 'Same as fp32', '**No**', 'Default on Ampere, Hopper, Blackwell (A100, H100, B200, L4, RTX 30/40)'],
      ['[[fp16|fp16]]', '5 / 10', 'max ≈ 65,504', '**Yes** — gradients underflow', 'Pre-Ampere GPUs without bf16: V100, T4'],
      ['[[fp8|FP8]] E4M3 / E5M2', '4/3 or 5/2', 'Narrow — per-tensor scales', 'Scaling handled by Transformer Engine', 'Hopper/Blackwell, large models, via [[transformer-engine|Transformer Engine]]'],
    ]},
    { t: 'p', x: 'Why fp16 needs a scaler: gradients of a deep network are often smaller than fp16\'s smallest normal number (~6e-5) and flush to zero. [[loss-scaling|Loss scaling]] multiplies the loss by a large factor (e.g. 65,536) before backward so gradients land in range, then divides them back before the optimizer step. PyTorch\'s `GradScaler` does this dynamically: if it sees inf/NaN gradients it skips that step and halves the scale; after a run of clean steps it doubles it. bf16 has fp32\'s exponent range, so it does not need any of this.' },
    { t: 'code', lang: 'python', title: 'One training step, both ways', x: `import torch

# TF32 for any remaining fp32 matmuls (Ampere+). Equivalent to allow_tf32=True for matmul.
torch.set_float32_matmul_precision("high")

use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
# GradScaler is only needed (and only enabled) for fp16
scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

model = model.cuda()                       # weights stay fp32 = the "master" copy
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, fused=True)

def train_step(batch):
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(batch["input_ids"])                      # matmuls run in bf16/fp16
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),          # loss in fp32
            batch["labels"].view(-1), ignore_index=-100)
    scaler.scale(loss).backward()          # no-op scaling when the scaler is disabled
    scaler.unscale_(opt)                   # bring grads back to true scale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                       # skipped automatically if grads contain inf/NaN
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return loss.item()` },
    { t: 'p', x: 'Two details people get wrong. First, in this "autocast" style the parameters stay fp32 and autocast casts them on the fly — that fp32 copy *is* the master weight. Frameworks like [[fsdp|FSDP]], DeepSpeed and Megatron instead store bf16 working weights plus a separate fp32 master copy in the optimizer, which is where the 2 + 2 + 12 accounting comes from. Second, you must `unscale_` before clipping, otherwise you clip scaled gradients against an unscaled threshold.' },
    { t: 'callout', kind: 'trap', title: 'fp16 on a T4, bf16 on an A100', x: [
      'A free Colab T4 (Turing, compute capability 7.5) has no native bf16 Tensor Core support — use fp16 + GradScaler there. Recent PyTorch may report bf16 as "supported" on such GPUs via emulation, but it is slow; check the compute capability (≥ 8) rather than trusting `is_bf16_supported()` alone.',
      'If a run in fp16 produces NaN losses or "gradient overflow, skipping step" forever, the fix on Ampere+ is simply to switch to bf16.',
    ]},

    { t: 'h2', x: 'Gradient accumulation and clipping' },
    { t: 'p', x: 'LLM recipes specify a global [[batch-size|batch size]] in tokens (e.g. 0.5M–4M per step for pretraining). When that does not fit, [[gradient-accumulation|gradient accumulation]] runs several micro-batches forward and backward, letting gradients sum in `.grad`, and only then steps the optimizer. The math is identical to one big batch (except for batch-norm-style layers, which transformers do not use) — you trade time for memory.' },
    { t: 'code', lang: 'python', title: 'Accumulate 8 micro-batches per optimizer step', x: `accum_steps = 8          # effective batch = micro_bs * accum_steps * world_size
opt.zero_grad(set_to_none=True)

for step, batch in enumerate(loader):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids=batch["input_ids"], labels=batch["labels"]).loss / accum_steps   # scale so the summed grad is an average
    loss.backward()                                 # grads accumulate in .grad

    if (step + 1) % accum_steps == 0:
        # clip on the full accumulated gradient, right before the step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()                                 # LR schedule counts optimizer steps
        opt.zero_grad(set_to_none=True)` },
    { t: 'p', x: '[[gradient-clipping|Gradient clipping]] rescales the whole gradient vector if its global L2 norm exceeds a threshold (1.0 is the near-universal LLM default). It does not stop learning; it caps the size of a single bad step, which is the first line of defence against loss spikes. Log `grad_norm` every step — a rising or spiky norm is the earliest warning of instability.' },
    { t: 'callout', kind: 'warn', title: 'Two classic accumulation bugs', x: [
      'Forgetting to divide the loss by `accum_steps` makes the effective learning rate `accum_steps` times larger.',
      'Stepping the LR scheduler on every micro-batch makes warmup and decay finish `accum_steps` times too early. In DDP, also wrap the non-final micro-batches in `model.no_sync()` so gradients are all-reduced only once per step.',
    ]},

    { t: 'h2', x: 'Activation checkpointing' },
    { t: 'p', x: '[[activation-checkpointing|Activation checkpointing]] (a.k.a. gradient checkpointing) discards a block\'s internal activations after the forward pass and recomputes them during backward. Applied per transformer block, it cuts activation memory from ~34·s·b·h·L to ~2·s·b·h·L bytes at a cost of about one extra forward pass (~30–35% more compute per step). **Selective** checkpointing — recomputing only the cheap-but-large parts such as attention internals, as Megatron does with `--recompute-granularity selective` — gets most of the memory saving for a few percent of compute.' },
    { t: 'code', lang: 'python', title: 'Checkpointing in plain PyTorch and in Hugging Face', x: `from torch.utils.checkpoint import checkpoint

class Model(torch.nn.Module):
    def forward(self, x):
        for block in self.blocks:
            # store only the block input; recompute internals during backward
            x = checkpoint(block, x, use_reentrant=False)
        return self.head(self.norm(x))

# Hugging Face models: one call
hf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
hf_model.config.use_cache = False   # the KV cache is pointless during training and conflicts with it` },

    { t: 'h2', x: 'FlashAttention, SDPA and torch.compile' },
    { t: 'p', x: 'Naive attention materialises an s × s score matrix per head, reads and writes it to HBM several times, and stores it for backward. [[flash-attention|FlashAttention]] tiles Q, K and V into on-chip SRAM, computes softmax incrementally (the "online softmax" trick) and never writes the full matrix — making attention memory **linear** in sequence length and much faster because it is limited by HBM traffic, not FLOPs. It recomputes the attention tiles in backward instead of storing them. You rarely call it directly: PyTorch\'s `F.scaled_dot_product_attention` (SDPA) dispatches to a FlashAttention or memory-efficient kernel when dtype, shape and GPU allow, and Hugging Face models use SDPA by default (`attn_implementation="sdpa"`, or `"flash_attention_2"` if the `flash-attn` package is installed). FlashAttention-3 targets Hopper\'s asynchronous Tensor Core instructions and FP8.' },
    { t: 'code', lang: 'python', title: 'Causal attention with SDPA, then compile the model', x: `import torch.nn.functional as F

# q, k, v: (batch, heads, seq, head_dim) in bf16/fp16 on CUDA
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # fused kernel, no s×s matrix kept

# torch.compile traces the model with TorchDynamo and generates fused Triton kernels
# (elementwise ops, norms, activations) via TorchInductor. First steps are slow (compiling).
model = torch.compile(model)            # mode="max-autotune" searches harder, compiles longer` },
    { t: 'p', x: '[[torch-compile|torch.compile]] typically gives 1.2–2× on small and medium models by fusing the memory-bound elementwise operations around the matmuls (RMSNorm, SwiGLU, residual adds, [[rope|RoPE]]) so they read HBM once. Keep shapes static where possible — changing sequence lengths trigger recompiles — which is one more reason pretraining uses packed, fixed-length sequences.' },

    { t: 'h2', x: 'FP8 with Transformer Engine' },
    { t: 'p', x: 'Hopper (H100/H200) and Blackwell (B200/GB200) Tensor Cores execute [[fp8|FP8]] matmuls at roughly 2× bf16 throughput. FP8\'s range is tiny, so every tensor needs a scale factor. NVIDIA [[transformer-engine|Transformer Engine]] (TE) provides drop-in layers (`te.Linear`, `te.LayerNormLinear`, `te.TransformerLayer`) and a context manager that handles the scales. The common "hybrid" recipe uses **E4M3** (more precision) for weights and activations in the forward pass and **E5M2** (more range) for gradients. With *delayed scaling*, each tensor\'s scale comes from a history of recent absolute maxima (amax). Blackwell adds block-scaled formats (MXFP8, NVFP4) where small blocks of values share a scale.' },
    { t: 'code', lang: 'python', title: 'FP8 forward/backward with Transformer Engine (Hopper or newer)', x: `import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format

layer = te.Linear(4096, 4096, bias=False, params_dtype=torch.bfloat16).cuda()
x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)

recipe = DelayedScaling(fp8_format=Format.HYBRID,   # E4M3 forward, E5M2 backward
                        amax_history_len=16, amax_compute_algo="max")

with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = layer(x)          # GEMM runs in FP8 on Tensor Cores; output in bf16
y.float().pow(2).mean().backward()
# Master weights and optimizer states remain higher precision.
# In NeMo / Megatron-Core the same thing is a config switch, e.g. --fp8-format hybrid.` },
    { t: 'p', x: 'FP8 pays off when matmuls dominate step time — large hidden sizes, big models — and gives little on a 50M-parameter toy where kernel launches and memory-bound ops dominate. It also halves the memory of the FP8-cast tensors. Validate by comparing the loss curve against a bf16 baseline for the first few thousand steps.' },

    { t: 'h2', x: 'Which lever for which problem' },
    { t: 'table', head: ['Symptom', 'First lever', 'Cost'], rows: [
      ['Training is slow in fp32', 'bf16 autocast (or fp16 + GradScaler on pre-Ampere); TF32 for leftover fp32', 'None meaningful'],
      ['OOM from activations', 'Smaller micro-batch + gradient accumulation; activation checkpointing', 'Time per step'],
      ['OOM at long context', 'SDPA/FlashAttention (removes s² memory), then checkpointing, then context parallelism', 'Little / recompute'],
      ['GPU busy but many small kernels', 'torch.compile, fused optimizer (`fused=True`)', 'Compile time'],
      ['Matmul-bound on H100/B200', 'FP8 via Transformer Engine', 'Some numerical validation'],
      ['fp16 loss goes NaN', 'Switch to bf16; otherwise check loss scaler, LR, clipping', '—'],
    ]},
    { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
      'bf16 vs fp16: bf16 has fp32\'s range so needs **no loss scaling**; fp16 needs a dynamic loss scaler.',
      '"Global batch must stay 1M tokens but micro-batch must shrink" → **gradient accumulation**.',
      '"Reduce activation memory at the cost of extra compute" → **activation checkpointing**.',
      '"Attention memory grows quadratically with context" → **FlashAttention** makes it linear.',
      'FP8 on H100 → **Transformer Engine**, E4M3 forward / E5M2 gradients.',
    ]},

    { t: 'h2', x: 'Key takeaways' },
    { t: 'ul', items: [
      'Use bf16 autocast on Ampere and newer; fp16 needs GradScaler (dynamic [[loss-scaling|loss scaling]]); enable TF32 for the rest.',
      'Gradient accumulation = divide the loss by the number of micro-batches, step optimizer and scheduler once per accumulated batch.',
      'Clip gradients to norm 1.0 and log the grad norm — it is your early-warning signal.',
      'Activation checkpointing trades ~30% compute for a large activation-memory cut; selective checkpointing is cheaper.',
      'SDPA/FlashAttention removes the s² attention memory; torch.compile fuses memory-bound ops; FP8 with Transformer Engine doubles matmul throughput on Hopper/Blackwell.',
    ]},
  ],
}

/* ------------------------------------------------------------------ */
const pretrainSmallLlm = {
  id: 'pretrain-small-llm',
  title: 'Lab: pretrain a small LLM end to end',
  minutes: 40,
  outcomes: [
    'Build a Llama-style model from scratch with LlamaConfig and count its parameters',
    'Tokenize and pack a pretraining corpus into fixed-length blocks',
    'Train with AdamW, cosine learning-rate decay with warmup, mixed precision and gradient clipping',
    'Evaluate validation perplexity, save the checkpoint and sample text from it',
    'Size a pretraining run with the 6·N·D compute rule and the ~20 tokens-per-parameter heuristic',
  ],
  blocks: [
    { t: 'lede', x: 'Time to put the last two lessons to work. You will [[pretraining|pretrain]] a ~49M-parameter Llama-architecture model **from random weights** on TinyStories — a corpus of simple children\'s stories that is small enough to learn from in under an hour but rich enough that the model produces coherent English at the end. Every piece is the same as a frontier pretraining run, just smaller: config, tokenizer, [[packing|packing]], AdamW with warmup + cosine decay, [[mixed-precision|mixed precision]], clipping, evaluation, checkpointing and sampling.' },

    { t: 'lab', title: 'Pretrain a 49M-parameter Llama on TinyStories', x: 'Train a decoder-only Llama-style model from scratch with Hugging Face `transformers`, watch its loss fall from ~10.8 to around 2, measure validation perplexity, then generate stories from your own checkpoint.', gpu: 'Free Colab T4 (fp16) or 1× 24 GB GPU (bf16)', time: '60–90 min', steps: [
      'Install `torch`, `transformers`, `datasets` and `accelerate`; confirm the GPU with `nvidia-smi`.',
      'Load the TinyStories dataset and a pretrained tokenizer (we reuse SmolLM2\'s 49k-token BPE vocabulary rather than training one).',
      'Tokenize every story, append an end-of-text token, and pack the stream into 512-token blocks.',
      'Create a `LlamaConfig` (hidden 512, 8 layers, 8 heads, 4 KV heads) and instantiate `LlamaForCausalLM` from it — random weights, no `from_pretrained`.',
      'Train for 6,000 steps (~100M tokens) with AdamW (β₂ = 0.95, weight decay 0.1), peak LR 1e-3, 300 warmup steps, cosine decay, clipping at 1.0.',
      'Evaluate perplexity = exp(validation loss), save model + tokenizer, and sample a few stories at temperature 0.8.',
      'Stretch: double the layers or the steps and compare the final validation loss.',
    ]},
    { t: 'code', lang: 'bash', title: '1 · Environment', x: `pip install -U torch transformers datasets accelerate
nvidia-smi   # T4 = 16 GB (use fp16); A10/L4/RTX 3090/4090/A100 = bf16` },
    { t: 'code', lang: 'python', title: '2 · Data: tokenize and pack TinyStories', x: `import math, torch
from datasets import load_dataset
from transformers import AutoTokenizer

SEQ_LEN = 512
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")   # 49,152-token BPE vocab
EOS = tok.eos_token_id

# ~30% of the train split is plenty for ~100M training tokens; validation is separate text.
train_raw = load_dataset("roneneldan/TinyStories", split="train[:30%]")
val_raw = load_dataset("roneneldan/TinyStories", split="validation[:3000]")

def tokenize(batch):
    # Append EOS so the model learns where one story ends and the next begins.
    ids = tok(batch["text"], add_special_tokens=False)["input_ids"]
    return {"input_ids": [x + [EOS] for x in ids]}

def pack(batch):
    # Concatenate a batch of stories and cut into fixed SEQ_LEN blocks (drop the remainder).
    stream = [t for ids in batch["input_ids"] for t in ids]
    n = len(stream) // SEQ_LEN * SEQ_LEN
    blocks = [stream[i:i + SEQ_LEN] for i in range(0, n, SEQ_LEN)]
    # Labels = inputs; the model shifts them by one position internally.
    return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

def prepare(ds):
    ds = ds.map(tokenize, batched=True, num_proc=4, remove_columns=ds.column_names)
    return ds.map(pack, batched=True, batch_size=1000, num_proc=4, remove_columns=ds.column_names)

train_ds, val_ds = prepare(train_raw), prepare(val_raw)
print(f"train blocks: {len(train_ds):,}  (~{len(train_ds) * SEQ_LEN / 1e6:.0f}M tokens)")
print(f"val blocks:   {len(val_ds):,}")` },
    { t: 'p', x: '**Why pack?** Padding every story to 512 tokens would waste most of each batch on pad tokens (TinyStories averages a couple of hundred tokens per story). Packing concatenates documents separated by EOS and slices the stream into full blocks, so every position contributes to the loss and every batch has the same shape — which also keeps `torch.compile` happy. The small cost is that attention can see across a document boundary inside a block; large-scale recipes either accept that or use document-aware attention masks.' },
    { t: 'code', lang: 'python', title: '3 · Model: a Llama from scratch', x: `from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=len(tok),
    hidden_size=512,
    intermediate_size=1408,          # SwiGLU MLP width (~2.75x hidden)
    num_hidden_layers=8,
    num_attention_heads=8,           # head_dim = 64
    num_key_value_heads=4,           # grouped-query attention
    max_position_embeddings=SEQ_LEN,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,        # share embedding and LM head: saves 25M params here
    bos_token_id=tok.bos_token_id,
    eos_token_id=EOS,
)
model = LlamaForCausalLM(config)     # random init. SDPA attention is the default.
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params/1e6:.1f}M")   # ≈ 48.8M, of which ≈ 25.2M is the embedding matrix

# Sanity check: an untrained model should be near-uniform -> loss ≈ ln(vocab) ≈ 10.8
with torch.no_grad():
    sample = torch.tensor([train_ds[0]["input_ids"]])
    print("initial loss:", model(input_ids=sample, labels=sample).loss.item())` },
    { t: 'p', x: 'Look at where the parameters are: about half of this tiny model is the 49,152 × 512 embedding table. That is normal for small models with large vocabularies and is why small-model recipes tie the embedding and output head. In a 7B model the embeddings are only a few percent. The initial-loss check is worth making a habit: a fresh model should score close to ln(V) ≈ 10.8 — a much higher value means an initialisation or label bug.' },
    { t: 'code', lang: 'python', title: '4 · Train: AdamW, warmup + cosine, mixed precision, clipping', x: `from transformers import Trainer, TrainingArguments, default_data_collator

cc_major = torch.cuda.get_device_capability()[0]
use_bf16 = cc_major >= 8                     # Ampere+ -> bf16; T4 (7.5) -> fp16 + loss scaling
if use_bf16:
    torch.set_float32_matmul_precision("high")   # TF32 for the remaining fp32 matmuls

args = TrainingArguments(
    output_dir="tinystories-llama-49m",
    max_steps=6000,                          # 6000 x 32 x 512 ≈ 98M tokens
    per_device_train_batch_size=32,          # drop to 16 + gradient_accumulation_steps=2 if OOM
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,
    learning_rate=1e-3,                      # small models tolerate high LR; 7B-class use ~3e-4
    lr_scheduler_type="cosine",
    warmup_steps=300,
    weight_decay=0.1,
    adam_beta1=0.9, adam_beta2=0.95,         # the usual LLM betas
    max_grad_norm=1.0,                       # gradient clipping
    bf16=use_bf16, fp16=not use_bf16,
    eval_strategy="steps", eval_steps=500,
    logging_steps=50,
    save_steps=2000, save_total_limit=2,
    dataloader_num_workers=2,
    report_to="none",                        # or "tensorboard" / "wandb"
)

trainer = Trainer(model=model, args=args,
                  train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=default_data_collator)   # blocks already have labels
trainer.train()` },
    { t: 'p', x: 'Every hyperparameter here is a scaled-down version of a real pretraining recipe. The **warmup** keeps the first steps small while Adam\'s second-moment estimates are still noisy; **cosine decay** then lowers the [[learning-rate|learning rate]] smoothly to zero by `max_steps`, which is where most of the final loss improvement comes from. β₂ = 0.95 (instead of Adam\'s default 0.999) makes the optimizer react faster to gradient-scale changes and is a standard stability choice for LLMs.' },
    { t: 'code', lang: 'python', title: 'What lr_scheduler_type="cosine" computes (for a plain PyTorch loop)', x: `import math

def lr_at(step, peak=1e-3, warmup=300, total=6000, min_ratio=0.0):
    if step < warmup:                                   # linear warmup from 0
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup) # 0 -> 1 after warmup
    cosine = 0.5 * (1 + math.cos(math.pi * progress))   # 1 -> 0
    return peak * (min_ratio + (1 - min_ratio) * cosine)
# Many LLM recipes decay to ~10% of peak (min_ratio=0.1) instead of to zero.
# Equivalent: transformers.get_cosine_schedule_with_warmup(opt, 300, 6000) + LambdaLR` },
    { t: 'code', lang: 'python', title: '5 · Evaluate, save and sample', x: `metrics = trainer.evaluate()
print(f"val loss {metrics['eval_loss']:.3f}  perplexity {math.exp(metrics['eval_loss']):.2f}")

trainer.save_model("tinystories-llama-49m/final")      # weights + config
tok.save_pretrained("tinystories-llama-49m/final")     # always ship the tokenizer with the model

model.eval()
prompt = "Once upon a time, there was a little dog named"
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=120, do_sample=True,
                         temperature=0.8, top_p=0.95, pad_token_id=EOS)
print(tok.decode(out[0], skip_special_tokens=True))` },

    { t: 'h2', x: 'What to expect' },
    { t: 'table', head: ['Checkpoint', 'Train loss (approx.)', 'What the samples look like'], rows: [
      ['Step 0', '≈ 10.8 (= ln 49,152)', 'Random tokens'],
      ['~Step 300 (end of warmup)', '≈ 3.5–4.5', 'Common words, broken grammar'],
      ['~Step 2,000', '≈ 2.2–2.6', 'Sentences with characters and simple plots, some contradictions'],
      ['Step 6,000', '≈ 1.6–2.2 (val perplexity ≈ 5–9)', 'Coherent short stories with a beginning and an end'],
    ]},
    { t: 'specs', rows: [
      ['Tokens processed', '6,000 × 32 × 512 ≈ 98M'],
      ['Training compute', '6 · N · D ≈ 6 × 49M × 98M ≈ 2.9 × 10¹⁶ FLOPs'],
      ['24 GB Ampere/Ada GPU (bf16)', 'Roughly 20–40 min'],
      ['Colab T4 (fp16)', 'Roughly 1–2 h — halve `max_steps` if your session is short'],
      ['Peak memory', 'A few GB of states (49M × 16 B ≈ 0.8 GB) plus activations; batch 32 × 512 fits in 16 GB'],
    ]},
    { t: 'p', x: 'Exact numbers vary with GPU, library versions and data order — treat the table as a sanity range. If your loss stalls above ~4, check that labels equal inputs (not shifted twice), that the LR schedule actually warms up, and that you are not training on padding. If validation loss is much higher than training loss on TinyStories, you are probably evaluating on a different tokenizer or text format.' },
    { t: 'callout', kind: 'trap', title: 'Do not let the collator mask your EOS', x: '`DataCollatorForLanguageModeling(mlm=False)` sets labels to −100 wherever `input_ids == pad_token_id`. Many tokenizers have no pad token, and the common fix `tok.pad_token = tok.eos_token` then silently removes every EOS from the loss — the model never learns to end a story. With packed, pad-free blocks, build the labels yourself (as above) and use `default_data_collator`.' },

    { t: 'h2', x: 'From toy to real: sizing a pretraining run' },
    { t: 'p', x: 'Two rules of thumb let you size any pretraining job. Training compute is **C ≈ 6 · N · D** FLOPs, where N is parameters and D is training tokens: 2·N for the forward pass and 4·N for the backward pass, per token. And the Chinchilla scaling result (Hoffmann et al., 2022) says that for a fixed compute budget, loss is minimised with roughly **D ≈ 20 · N** tokens. Divide C by what your GPUs actually sustain (peak × [[mfu|MFU]]) to get wall-clock time.' },
    { t: 'table', head: ['Model N', 'Chinchilla tokens D ≈ 20N', 'Compute 6ND', '8× H100 at ~40% MFU (~400 TFLOP/s each)'], rows: [
      ['49M (this lab)', '~1B', '~3 × 10¹⁷', 'Under 2 minutes'],
      ['125M', '2.5B', '~1.9 × 10¹⁸', '~10 minutes'],
      ['1B', '20B', '~1.2 × 10²⁰', '~10 hours'],
      ['7B', '140B', '~5.9 × 10²¹', '~3 weeks'],
    ]},
    { t: 'p', x: 'Our lab uses ~2 tokens per parameter, so by Chinchilla standards it is undertrained — doubling or tripling `max_steps` keeps lowering the loss. Note also that modern open models deliberately train far past 20N (Llama 3 8B saw about 15T tokens, ~1,900 per parameter) because a smaller model trained longer is cheaper to *serve*. Chinchilla answers "best loss per training FLOP", not "best model for deployment".' },
    { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
      'Training FLOPs ≈ **6 × parameters × tokens**; compute-optimal tokens ≈ **20 × parameters**.',
      'Perplexity = exp(cross-entropy loss); a fresh model\'s loss ≈ ln(vocab size).',
      'Warmup + cosine decay, AdamW with weight decay ~0.1, β₂ ≈ 0.95, gradient clipping at 1.0 are the standard pretraining recipe.',
      'NVIDIA-scale equivalent: the same run in [[nemo|NeMo]] / [[megatron|Megatron-Core]] is a pretraining recipe with a Llama config, a data blend of pre-tokenized indexed datasets, and TP/PP/DP settings.',
    ]},

    { t: 'h2', x: 'Key takeaways' },
    { t: 'ul', items: [
      'Pretraining from scratch is `LlamaForCausalLM(config)` — not `from_pretrained` — plus tokenized, packed, fixed-length blocks with labels = inputs.',
      'Sanity-check the initial loss against ln(V); expect a fast drop during warmup, then a long slow decline.',
      'The standard recipe: AdamW (β₂ 0.95, wd 0.1), linear warmup, cosine decay, bf16 (fp16 + scaler on T4), clip at 1.0.',
      'Report validation [[perplexity|perplexity]], save the tokenizer with the model, and sample to check quality by eye.',
      'Size runs with C ≈ 6ND and D ≈ 20N, then divide by peak FLOP/s × MFU for wall-clock time.',
    ]},
  ],
}

/* ------------------------------------------------------------------ */
const distributed = {
  id: 'distributed',
  title: 'Distributed training: DDP, FSDP/ZeRO, tensor, pipeline and sequence parallelism',
  minutes: 25,
  outcomes: [
    'Launch data-parallel training with DDP and torchrun, and explain the gradient all-reduce',
    'Compare ZeRO stages 1–3 and FSDP by what they shard, how much memory they save and what they communicate',
    'Explain tensor, pipeline, sequence, context and expert parallelism and the bottleneck each one introduces',
    'Size a 3D-parallel layout (TP × PP × DP) for a model and cluster, placing TP inside an NVLink domain',
    'Map these ideas onto Megatron-Core/NeMo settings and Hugging Face Accelerate',
  ],
  blocks: [
    { t: 'lede', x: 'One GPU runs out of either memory or time. Distributed training splits the work along three axes: the **data** (every GPU sees different samples), the **model\'s state** (shard weights, gradients and optimizer states), and the **model\'s computation** (split each layer, or split the stack of layers). This lesson walks through each strategy, what it costs in communication, and how they combine into the "3D parallelism" that trains every large LLM.' },

    { t: 'h2', x: 'Collectives: the vocabulary of distributed training' },
    { t: 'p', x: 'All strategies are built from a handful of collective operations, implemented on NVIDIA GPUs by [[nccl|NCCL]] over [[nvlink|NVLink]] inside a node and [[infiniband|InfiniBand]] or RoCE between nodes.' },
    { t: 'table', head: ['Collective', 'What it does', 'Used by'], rows: [
      ['[[all-reduce|All-reduce]]', 'Every rank ends with the sum (or mean) of everyone\'s tensor', 'DDP gradient sync; tensor-parallel layer outputs'],
      ['[[reduce-scatter|Reduce-scatter]]', 'Sum across ranks, but each rank keeps only its 1/N slice of the result', 'ZeRO/FSDP gradient sharding; sequence parallelism'],
      ['[[all-gather|All-gather]]', 'Each rank contributes a slice; everyone ends with the full tensor', 'ZeRO-3/FSDP gathering parameters before a layer runs'],
      ['All-to-all', 'Each rank sends a different chunk to every other rank', 'Expert parallelism (routing tokens to experts); Ulysses-style sequence parallelism'],
      ['Send / recv (point-to-point)', 'One rank hands activations or gradients to the next', 'Pipeline parallelism between stages'],
      ['Broadcast', 'One rank\'s tensor copied to all', 'Initial weight sync'],
    ]},
    { t: 'p', x: 'A ring all-reduce is literally a reduce-scatter followed by an all-gather, and each rank sends and receives about 2 × (N−1)/N × the tensor size. Remember that — it is why ZeRO-1/2 cost the same bandwidth as plain data parallelism. Bandwidth figures to keep in mind: NVLink 4 on H100 gives 900 GB/s per GPU; NVLink 5 on Blackwell 1.8 TB/s, and an NVL72 rack puts 72 GPUs in one NVLink domain; a 400 Gb/s InfiniBand NDR link is ~50 GB/s per GPU. Inside a node is roughly an order of magnitude faster than between nodes.' },

    { t: 'h2', x: 'Data parallelism with DDP' },
    { t: 'p', x: 'With [[ddp|DistributedDataParallel]], every GPU holds a full replica of the model and optimizer, processes a different slice of the batch, and the gradients are averaged with an all-reduce during the backward pass (DDP buckets gradients and overlaps the all-reduce with the remaining backward computation). After the step all replicas are identical again. It scales throughput almost linearly — but it saves **no memory**: each GPU still needs the full 16 bytes per parameter.' },
    { t: 'code', lang: 'python', title: 'train_ddp.py — launched with torchrun', x: `import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

dist.init_process_group(backend="nccl")              # torchrun sets RANK, WORLD_SIZE, MASTER_ADDR...
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = build_model().cuda()
model = DDP(model, device_ids=[local_rank])         # registers all-reduce hooks on gradients
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)

sampler = DistributedSampler(train_ds, shuffle=True)  # each rank gets a disjoint shard
loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True)

for epoch in range(epochs):
    sampler.set_epoch(epoch)                          # different shuffle every epoch
    for batch in loader:
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
        loss.backward()                               # gradients all-reduced here
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), f"ckpt_{epoch}.pt")   # save once, from rank 0

dist.destroy_process_group()` },
    { t: 'code', lang: 'bash', title: 'Launching on one node and on two nodes', x: `# 8 GPUs on one machine
torchrun --nproc_per_node=8 train_ddp.py

# 2 nodes x 8 GPUs: run on every node (same rendezvous endpoint)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \\
         --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE_IP:29500 train_ddp.py
# Under Slurm, srun launches one torchrun per node with these variables filled in.` },

    { t: 'h2', x: 'Sharding the state: ZeRO and FSDP' },
    { t: 'p', x: 'Data parallelism stores N identical copies of the optimizer states, gradients and weights. [[zero|ZeRO]] (DeepSpeed) removes the redundancy in three stages, and PyTorch [[fsdp|FSDP]] implements the same idea natively. With N data-parallel GPUs:' },
    { t: 'table', head: ['Strategy', 'Sharded across ranks', 'Memory per GPU (Ψ params, 16 B/param)', '7B on 8 GPUs', 'Communication per step'], rows: [
      ['DDP', 'Nothing', '16Ψ', '112 GB — does not fit', 'All-reduce grads (≈2Ψ)'],
      ['ZeRO-1', 'Optimizer states', '4Ψ + 12Ψ/N', '38.5 GB', '≈2Ψ (same as DDP)'],
      ['ZeRO-2', '+ gradients', '2Ψ + 14Ψ/N', '26.3 GB', '≈2Ψ: reduce-scatter grads, all-gather updated params'],
      ['ZeRO-3 / FSDP full shard', '+ parameters', '16Ψ/N', '14 GB', '≈3Ψ: all-gather params in forward and again in backward, reduce-scatter grads'],
    ]},
    { t: 'p', x: 'ZeRO-3/FSDP never holds the whole model: before each layer (or "unit") runs, it all-gathers that layer\'s parameter shards, computes, and frees the full copy; in backward it gathers again, computes gradients, and reduce-scatters them so each rank keeps only its shard. The ~50% extra traffic is usually hidden by prefetching the next layer while the current one computes — as long as the interconnect is fast enough. **Hybrid sharding** (FSDP `HYBRID_SHARD`, ZeRO++ hpZ) shards within a node and replicates across nodes to keep the chatty all-gathers on NVLink. Megatron\'s *distributed optimizer* (`--use-distributed-optimizer`) is the ZeRO-1 idea inside Megatron-Core.' },
    { t: 'code', lang: 'python', title: 'FSDP2 (fully_shard) on a Hugging Face Llama', x: `import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy   # PyTorch 2.6+

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,    # compute in bf16
                          reduce_dtype=torch.float32)    # reduce-scatter grads in fp32

# Shard each transformer block as its own unit (gathered just-in-time), then the root.
for block in model.model.layers:
    fully_shard(block, mp_policy=mp)
fully_shard(model, mp_policy=mp)

# Build the optimizer AFTER sharding: it sees the local DTensor shards (fp32 = master weights).
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
# Training loop is unchanged; launch with torchrun exactly like DDP.` },
    { t: 'callout', kind: 'tip', title: 'Offload as a last resort', x: 'ZeRO-Offload / ZeRO-Infinity and FSDP CPU offload move optimizer states (or parameters) to CPU RAM or NVMe. It lets a big model train on few GPUs, but PCIe is far slower than HBM, so step time can grow several-fold. Prefer more GPUs, PEFT, or a smaller model when time matters.' },

    { t: 'h2', x: 'Model parallelism: tensor, pipeline, sequence, context, expert' },
    { t: 'p', x: 'Sharding state still requires each GPU to run whole layers on its own micro-batch, and every rank needs its own activations. When a single layer\'s activations or compute are too large, or the cluster is too big for pure data parallelism to stay efficient, you split the computation itself.' },
    { t: 'vs', cards: [
      { h: 'Tensor parallelism (TP)', items: [
        'Splits each weight matrix across GPUs (Megatron style): first MLP linear column-wise, second row-wise; attention split by heads',
        'Each GPU holds 1/TP of every layer\'s weights and does 1/TP of the math',
        '2 all-reduces per layer in forward + 2 in backward — on the critical path',
        'Needs NVLink bandwidth: keep TP ≤ GPUs per NVLink domain (typically 8)',
      ]},
      { h: 'Pipeline parallelism (PP)', items: [
        'Splits the layer stack into stages on different GPUs/nodes',
        'Only point-to-point activation transfers at stage boundaries — works across nodes',
        'Idle "bubble" while the pipeline fills and drains',
        'Fix: many micro-batches, 1F1B scheduling, interleaved (virtual) stages',
      ]},
      { h: 'Sequence & context parallelism', items: [
        '[[sequence-parallelism|Sequence parallelism]]: within a TP group, shard norm/dropout activations along the sequence; all-reduce becomes reduce-scatter + all-gather',
        '[[context-parallelism|Context parallelism]]: split the whole sequence across GPUs for all layers; attention exchanges K/V in a ring',
        'Used for long-context training (32k–1M tokens)',
      ]},
      { h: 'Expert parallelism (EP)', items: [
        'For Mixture-of-Experts: place different experts on different GPUs',
        'Tokens routed to their experts with all-to-all, then routed back',
        'All-to-all volume and load imbalance are the bottlenecks',
        'Combined with TP/PP/DP for large MoE models',
      ]},
    ]},
    { t: 'h3', x: 'Pipeline bubbles, micro-batches and 1F1B' },
    { t: 'p', x: 'With p pipeline stages and m micro-batches per step, the GPipe-style schedule leaves each stage idle for a fraction **(p − 1) / (m + p − 1)** of the step — the [[pipeline-bubble|pipeline bubble]]. With p = 4 and m = 4 that is 3/7 ≈ 43% idle; with m = 32 it is 3/35 ≈ 9%. The **1F1B** (one-forward-one-backward) schedule has the same bubble but starts backward passes as early as possible, so each stage holds activations for at most p micro-batches instead of m — a big memory win. **Interleaved 1F1B** gives each GPU v non-contiguous chunks of layers ("virtual stages") and shrinks the bubble by a factor of v at the price of more communication. Rule of thumb: keep m ≥ 4p.' },
    { t: 'flow', steps: [
      { b: 'Fill', s: 'Stage 1 runs forward on micro-batch 1, hands activations to stage 2; later stages are idle (bubble).' },
      { b: 'Steady state (1F1B)', s: 'Every stage alternates one forward and one backward, all busy.' },
      { b: 'Drain', s: 'The last backward passes flow back to stage 1; later stages fall idle (bubble).' },
      { b: 'Step', s: 'Gradients accumulated over all m micro-batches; optimizer steps once (synchronous).' },
    ]},

    { t: 'h2', x: '3D parallelism: putting it together' },
    { t: 'p', x: 'Large runs combine all three: **world size = TP × PP × DP** (× CP, with EP layered on for [[moe|MoE]]). The placement rule follows from the communication patterns: the chattiest axis gets the fastest link.' },
    { t: 'ol', items: [
      '**TP inside a node** (over NVLink), usually TP = 2, 4 or 8 — never spanning slower links.',
      '**PP across nodes** when the model still does not fit — point-to-point traffic tolerates InfiniBand.',
      '**DP (with a distributed optimizer / ZeRO-1) over everything left** — its all-reduce overlaps with backward.',
      'Add **CP** for very long sequences and **EP** for MoE; turn on **sequence parallelism** whenever TP > 1 (it is nearly free).',
    ]},
    { t: 'p', x: 'Worked example: a 70B dense model on 64 H100s (8 nodes × 8). With TP = 8 inside each node and PP = 2 across node pairs, each GPU holds 70B / 16 ≈ 4.4B parameters: 8.75 GB of bf16 weights + 8.75 GB of gradients, and with a distributed optimizer over DP = 64 / 16 = 4 replicas, 12 × 4.4B / 4 ≈ 13 GB of optimizer state — about 31 GB before activations, which leaves room for activations with selective recomputation on an 80 GB GPU.' },
    { t: 'code', lang: 'bash', title: 'The same layout as Megatron-LM (Megatron-Core) flags', x: `torchrun --nnodes=8 --nproc_per_node=8 ... pretrain_gpt.py \\
  --tensor-model-parallel-size 8 \\
  --pipeline-model-parallel-size 2 \\
  --sequence-parallel \\
  --use-distributed-optimizer \\
  --micro-batch-size 1 --global-batch-size 1024 \\
  --seq-length 4096 --bf16 \\
  --recompute-granularity selective \\
  --num-layers 80 --hidden-size 8192 --num-attention-heads 64 \\
  --group-query-attention --num-query-groups 8
# DP is implied: 64 / (8 * 2) = 4. Long context: --context-parallel-size 2.
# MoE: --num-experts 8 --expert-model-parallel-size 8. FP8: --fp8-format hybrid.` },
    { t: 'code', lang: 'python', title: 'NeMo 2.x recipe style (check names against your NeMo version)', x: `from nemo.collections import llm

recipe = llm.llama3_70b.pretrain_recipe(name="llama70b", num_nodes=8, num_gpus_per_node=8)
strategy = recipe.trainer.strategy               # a MegatronStrategy
strategy.tensor_model_parallel_size = 8
strategy.pipeline_model_parallel_size = 2
strategy.context_parallel_size = 1
strategy.sequence_parallel = True
recipe.data.micro_batch_size = 1
recipe.data.global_batch_size = 1024` },
    { t: 'p', x: 'The global batch must be divisible by micro-batch × DP; the number of gradient-accumulation steps (= micro-batches per pipeline step) is global / (micro × DP) = 1024 / 4 = 256 here, which also keeps the pipeline bubble small.' },

    { t: 'h2', x: 'Hugging Face Accelerate: one script, any strategy' },
    { t: 'p', x: '[[hf-accelerate|Accelerate]] wraps the same PyTorch machinery so one training script runs on a single GPU, DDP, FSDP or DeepSpeed depending on a config file. The Hugging Face `Trainer` uses it underneath, so the lab\'s script becomes multi-GPU with `accelerate launch` or `torchrun`.' },
    { t: 'code', lang: 'python', title: 'A training loop that runs anywhere', x: `from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
model, opt, loader, sched = accelerator.prepare(model, opt, loader, sched)

for batch in loader:
    with accelerator.accumulate(model):          # handles no_sync + loss scaling of accumulation
        loss = model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()

# accelerate config        -> choose DDP / FSDP / DeepSpeed ZeRO stage interactively
# accelerate launch train.py` },

    { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
      '"Model fits on one GPU, want faster training" → **DDP** (data parallelism).',
      '"Model states do not fit, but layers do" → **FSDP / ZeRO-3** (or ZeRO-1/2 if only optimizer/gradients overflow).',
      '"A single layer is too big / need to cut per-GPU compute" → **tensor parallelism**, inside the NVLink domain.',
      '"Scale across nodes over InfiniBand with a deep model" → **pipeline parallelism**; reduce bubbles with more micro-batches / interleaving.',
      '"Very long sequences" → **context parallelism**; "MoE" → **expert parallelism** with all-to-all.',
    ]},
    { t: 'callout', kind: 'trap', title: 'Common wrong answers', x: [
      'DDP does **not** reduce per-GPU memory — every rank holds the full model and optimizer.',
      'Tensor parallelism across nodes over InfiniBand is almost always the wrong placement: its all-reduces sit on the critical path of every layer.',
      'Pipeline parallelism with fewer micro-batches than stages leaves most GPUs idle.',
    ]},

    { t: 'h2', x: 'Key takeaways' },
    { t: 'ul', items: [
      'DDP replicates everything and all-reduces gradients: faster, not smaller.',
      'ZeRO-1/2/3 shard optimizer states → gradients → parameters; FSDP full-shard = ZeRO-3 at ~1.5× DDP traffic (all-gather + reduce-scatter).',
      'TP splits matrices (all-reduce per layer, needs NVLink); PP splits layers (point-to-point, bubble = (p−1)/(m+p−1)); SP/CP split the sequence; EP splits MoE experts (all-to-all).',
      '3D layout: TP within the node, PP across nodes, DP over the rest; world = TP × PP × DP.',
      'Megatron-Core/NeMo expose these as `tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`, `sequence_parallel`; Accelerate/Trainer give DDP/FSDP/DeepSpeed for Hugging Face code.',
    ]},
  ],
}

/* ------------------------------------------------------------------ */
const profiling = {
  id: 'profiling',
  title: 'Profiling and troubleshooting training: MFU, Nsight, OOMs and NaNs',
  minutes: 22,
  outcomes: [
    'Compute model FLOPs utilization (MFU) from throughput, model size and GPU peak',
    'Profile a training step with torch.profiler, Nsight Systems and Nsight Compute, and read the results',
    'Diagnose data-loader, communication and memory bottlenecks from GPU utilisation metrics',
    'Apply an ordered ladder of fixes to out-of-memory errors',
    'Debug NaN losses, loss spikes and NCCL timeouts systematically',
  ],
  blocks: [
    { t: 'lede', x: 'A training run that works is not the same as a training run that is efficient — or one that will still be working tomorrow. This lesson gives you one number to judge efficiency ([[mfu|MFU]]), the tools to find where time goes (torch.profiler, [[nsight-systems|Nsight Systems]], [[nsight-compute|Nsight Compute]], [[dcgm|DCGM]]), and playbooks for the three failures you will definitely meet: OOM, NaN and hangs.' },

    { t: 'h2', x: 'MFU: one number for training efficiency' },
    { t: 'p', x: '**Model FLOPs utilization** = the FLOPs your model mathematically needs per second ÷ the GPU\'s peak FLOPs. For a dense transformer, training needs about 6·N FLOPs per token (plus an attention term of roughly 12·L·h·s per token that matters at long context). So:' },
    { t: 'code', lang: 'text', title: 'MFU with real numbers', x: `MFU = (6 · N · tokens_per_second_per_gpu) / peak_flops_per_gpu

Example: 7B model on 8× H100 SXM, measured 80,000 tokens/s total
  tokens/s per GPU      = 10,000
  achieved FLOP/s       = 6 × 7e9 × 1e4 = 4.2e14  (420 TFLOP/s)
  H100 dense BF16 peak  ≈ 989 TFLOP/s
  MFU                   ≈ 420 / 989 ≈ 42%

Same model at 4,000 tokens/s per GPU → 168 TFLOP/s → MFU ≈ 17%  → something is wrong.
A100 dense BF16 peak ≈ 312 TFLOP/s. Use DENSE peaks (not the 2:4-sparsity figures).` },
    { t: 'p', x: 'Well-tuned dense LLM pretraining typically reaches **~35–55% MFU** on H100-class GPUs; small models and short sequences land lower because non-matmul work and kernel launches dominate. **HFU** (hardware FLOPs utilization) also counts recomputation from [[activation-checkpointing|activation checkpointing]], so HFU ≥ MFU — quote MFU when comparing setups, since it does not reward wasted work.' },

    { t: 'h2', x: 'Where does the time go? Profiling tools' },
    { t: 'table', head: ['Tool', 'Granularity', 'Answers', 'Use when'], rows: [
      ['`nvidia-smi` / `nvidia-smi dmon`', 'Whole GPU, ~1 s', 'Is a kernel running? Memory used, power, temperature', 'First 30 seconds of any investigation'],
      ['[[dcgm|DCGM]] (`dcgmi dmon`, dcgm-exporter)', 'Whole GPU, profiling counters', 'SM active, tensor-pipe active, DRAM active, NVLink/PCIe traffic', 'Fleet monitoring; "busy but slow" questions'],
      ['`torch.profiler`', 'PyTorch op + CUDA kernel', 'Which ops/kernels take time, CPU vs GPU, memory per op', 'Python-level hot spots, data loader gaps'],
      ['[[nsight-systems|Nsight Systems]] (`nsys`)', 'System timeline', 'CPU threads, CUDA API, kernels, NCCL, memcpy, NVTX ranges on one timeline', 'Gaps, overlap of compute and communication, multi-GPU'],
      ['[[nsight-compute|Nsight Compute]] (`ncu`)', 'Single kernel', 'Occupancy, memory throughput, roofline, warp stalls', 'Optimising one specific kernel'],
    ]},
    { t: 'code', lang: 'python', title: 'torch.profiler around a few training steps', x: `from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=3, repeat=1),  # skip warm-up / compile steps
    on_trace_ready=tensorboard_trace_handler("./prof"),       # view in TensorBoard or Perfetto
    record_shapes=True, profile_memory=True, with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()                       # advance the schedule
        if step >= 6:
            break

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

# Memory: record allocations, dump a snapshot, drag it into https://pytorch.org/memory_viz
torch.cuda.memory._record_memory_history(max_entries=100_000)
train_step(next(iter(loader)))
torch.cuda.memory._dump_snapshot("mem.pickle")` },
    { t: 'code', lang: 'bash', title: 'Nsight Systems and Nsight Compute from the command line', x: `# Timeline of steps 10-15: call torch.cuda.profiler.start()/stop() in the script around them,
# and wrap phases in torch.cuda.nvtx.range_push("forward") / range_pop() for readable labels.
nsys profile -t cuda,nvtx,osrt,cudnn,cublas \\
     --capture-range=cudaProfilerApi --capture-range-end=stop \\
     -o step_profile python train.py
nsys stats step_profile.nsys-rep          # summary tables; open the .nsys-rep in the GUI

# Deep-dive one kernel (slow: replays the kernel many times)
ncu --set full -k regex:gemm -c 3 -o gemm_report python train.py

# Live GPU counters
nvidia-smi dmon -s pucm                   # power, utilisation, clocks, memory
dcgmi dmon -e 1002,1004,1005              # SM active, tensor pipe active, DRAM active` },
    { t: 'p', x: 'In an Nsight Systems timeline, healthy training looks like a dense wall of kernels with [[nccl|NCCL]] kernels overlapping compute. **Gaps** on the GPU row with activity on CPU threads mean the GPU is waiting for Python, the data loader or a host-device sync (`.item()`, printing a tensor, `torch.cuda.synchronize()`). **Long NCCL kernels not overlapping compute** mean communication is exposed: check the interconnect, bucket sizes, or the parallelism layout.' },
    { t: 'callout', kind: 'trap', title: '"GPU utilization 100%" does not mean efficient', x: '`nvidia-smi`\'s utilization.gpu is the fraction of time *any* kernel was running — a single tiny kernel on one SM counts as 100%. Use DCGM\'s **SM active** and **tensor pipe active** (or MFU) to know whether the Tensor Cores are actually busy. Conversely, low utilization with high CPU usage is the classic signature of a data-loader bottleneck.' },

    { t: 'h3', x: 'Data-loader bottlenecks' },
    { t: 'p', x: 'Symptoms: GPU utilization saw-tooths, step time varies, and the profiler shows time in `DataLoader.__next__`. Fixes, in order: pre-tokenize offline (never tokenize in the training loop), set `num_workers` (4–8 per GPU) with `pin_memory=True` and `persistent_workers=True`, increase `prefetch_factor`, use memory-mapped or streaming formats (Arrow, Megatron indexed datasets), and move data to local NVMe instead of network storage.' },

    { t: 'h2', x: 'The OOM ladder' },
    { t: 'p', x: '`torch.OutOfMemoryError: CUDA out of memory` tells you how much was requested, how much is allocated by PyTorch and how much is *reserved but unallocated* (fragmentation). Use the memory-math lesson to decide whether states or activations are the problem, then climb this ladder — cheapest change first:' },
    { t: 'ol', items: [
      '**Mixed precision** (bf16) if you are still in fp32.',
      '**Smaller micro-batch + [[gradient-accumulation|gradient accumulation]]** to keep the global batch — fixes activation OOMs.',
      '**[[activation-checkpointing|Activation checkpointing]]** (selective first, then full).',
      '**SDPA/[[flash-attention|FlashAttention]]** and shorter sequences; context parallelism for long context.',
      '**Leaner optimizer**: fused AdamW, 8-bit Adam, or a smaller-state optimizer.',
      '**Shard states**: ZeRO-1 → ZeRO-2 → FSDP/ZeRO-3 across more GPUs; then tensor/pipeline parallelism.',
      '**Offload** optimizer states/parameters to CPU or NVMe (slow).',
      '**Change the problem**: LoRA/QLoRA instead of full fine-tuning, or a smaller model.',
    ]},
    { t: 'p', x: 'If the error shows lots of reserved-but-unallocated memory, try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (newer PyTorch releases also read the device-generic `PYTORCH_ALLOC_CONF`) to reduce fragmentation from variable-length batches. An OOM that appears only after hours usually means a leak: accumulating tensors that still hold the graph (e.g. `total_loss += loss` instead of `loss.item()`), or growing sequence lengths.' },

    { t: 'h2', x: 'NaNs and loss spikes' },
    { t: 'p', x: 'A loss that becomes NaN or jumps by several units in one step is a numerical or data problem. Work through it in this order:' },
    { t: 'table', head: ['Check', 'Signal', 'Fix'], rows: [
      ['Precision', 'fp16 run; scaler keeps skipping steps or scale collapses to tiny values', 'Switch to bf16; keep softmax/loss in fp32'],
      ['Learning rate', 'Spikes soon after warmup ends; grad norm climbing before the spike', 'Lower peak LR, longer warmup, confirm clipping at 1.0 is active'],
      ['Data', 'Spike reproduces at the same step; one batch has huge loss', 'Inspect that batch (empty docs, binary junk, extreme repetition); skip it'],
      ['Code / init', 'NaN at step 0 or initial loss ≫ ln(V)', '`torch.autograd.set_detect_anomaly(True)` for a few steps to find the op; check masks and labels'],
      ['Architecture at scale', 'Attention logits or output logits growing', 'QK-norm, z-loss on logits, lower Adam ε / β₂ = 0.95'],
    ]},
    { t: 'p', x: 'Large-scale practice when a spike will not recover: **roll back** to the last good checkpoint, **skip** the data batches around the spike (or reshuffle), optionally lower the LR, and resume. That only works if you checkpoint frequently and your data loader is deterministic and resumable — design for it from day one.' },

    { t: 'h2', x: 'Hangs and NCCL timeouts' },
    { t: 'p', x: 'A distributed job that freezes and later dies with a watchdog message such as "Watchdog caught collective operation timeout" means one or more ranks never reached a collective the others were waiting in. NCCL itself is rarely the root cause; it is where the problem becomes visible.' },
    { t: 'ul', items: [
      '**Rank divergence**: code paths that differ by rank (an `if rank == 0:` that calls a collective, uneven numbers of batches per rank, an exception on one rank only). Make every rank run the same sequence of collectives.',
      '**One slow rank**: data loading or checkpoint saving on one node takes longer than the timeout (default 10 min for NCCL in PyTorch). Raise it with `init_process_group(timeout=timedelta(minutes=30))` only after ruling out a real hang.',
      '**Network/hardware**: a down InfiniBand link, wrong interface, or a GPU with Xid errors. Check `dmesg`, DCGM health, and run `nccl-tests` (`all_reduce_perf -b 8 -e 4G -f 2 -g 8`) to measure bus bandwidth.',
      '**Configuration**: set `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` to the right interfaces in multi-NIC hosts.',
    ]},
    { t: 'code', lang: 'bash', title: 'First steps when a multi-GPU job hangs', x: `export NCCL_DEBUG=INFO                 # topology, transport (NVLink/IB/socket) and errors per rank
export NCCL_DEBUG_SUBSYS=INIT,NET      # narrow the output
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # checks that ranks call matching collectives (slower)
torchrun --nproc_per_node=8 train.py 2>&1 | tee run.log

# Look for: "via NET/Socket" where you expected IB (wrong NIC / missing drivers),
# a rank that never prints its init line, or Xid errors in dmesg on one node.
# py-spy dump --pid <pid>  shows where each Python rank is stuck.` },

    { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
      'MFU = 6·N·tokens/s ÷ peak FLOP/s — expect to compute it.',
      '"Timeline of CPU, CUDA kernels, NCCL and memcpy to find gaps" → **Nsight Systems**; "tune one kernel" → **Nsight Compute**.',
      '"GPU utilization low, CPU busy" → **data loader**: more workers, pre-tokenized data, pinned memory.',
      '"OOM with a fixed global batch" → smaller micro-batch + gradient accumulation, then checkpointing, then sharding.',
      '"Hang in multi-node training" → `NCCL_DEBUG=INFO`, check interfaces/IB, nccl-tests, rank divergence.',
    ]},

    { t: 'h2', x: 'Key takeaways' },
    { t: 'ul', items: [
      'MFU (≈35–55% is good for large dense models) is the honest efficiency metric; nvidia-smi utilization is not.',
      'Use torch.profiler for op-level hot spots, Nsight Systems for the timeline and overlap, Nsight Compute for single kernels, DCGM for fleet counters.',
      'Climb the OOM ladder: precision → micro-batch + accumulation → checkpointing → FlashAttention → optimizer → sharding → offload → PEFT.',
      'NaNs: precision, LR, data, code — and keep frequent, resumable checkpoints so you can roll back and skip bad batches.',
      'NCCL timeouts usually mean one rank diverged or stalled; `NCCL_DEBUG=INFO` and nccl-tests are the first tools.',
    ]},
  ],
}

/* ------------------------------------------------------------------ */
export default {
  id: 'gpu',
  n: 3,
  domain: 'gpu',
  weight: 14,
  title: 'GPU Acceleration and Optimization',
  blurb: 'Learn how LLMs are actually trained on GPUs: do the memory arithmetic, use mixed precision and checkpointing, pretrain a small Llama yourself, scale out with DDP, FSDP/ZeRO and 3D parallelism, and profile and debug runs with MFU, Nsight and NCCL tools.',
  hours: '6–8 h',
  lessons: [memoryMath, mixedPrecision, pretrainSmallLlm, distributed, profiling],
  quiz: [
    { id: 'g01', domain: 'gpu',
      scenario: 'A team wants to fully fine-tune a 13B-parameter model with AdamW in bf16 mixed precision. They plan to ignore activations for a first estimate.',
      q: 'Roughly how much GPU memory do the weights, gradients and optimizer states need in total?',
      opts: [
        'About 26 GB',
        'About 52 GB',
        'About 208 GB',
        'About 104 GB',
      ],
      a: 2,
      why: 'Mixed-precision AdamW costs **16 bytes per parameter**: 2 (bf16 weights) + 2 (bf16 grads) + 12 (fp32 master weights and two Adam moments). 13B × 16 B ≈ 208 GB — more than two 80 GB GPUs, so it needs [[fsdp|FSDP]]/[[zero|ZeRO]] sharding.',
      distract: '**26 GB** is the bf16 weights only (inference). **52 GB** is 4 bytes/param, e.g. fp32 weights alone. **104 GB** is 8 bytes/param — it forgets the master copy or one of the Adam moments.' },

    { id: 'g02', domain: 'gpu',
      scenario: 'A 3B model trains fine on one 80 GB GPU with micro-batch 8 at 2k context. Increasing the context to 8k with the same micro-batch causes an OOM. The team must keep the same global batch in tokens.',
      q: 'Which change addresses the cause most directly with the least loss of throughput?',
      opts: [
        'Switch from AdamW to plain SGD so that the optimizer states disappear entirely',
        'Use a smaller micro-batch with more gradient accumulation, plus activation checkpointing if needed',
        'Shard the optimizer states with ZeRO-1 across a second GPU added to the job',
        'Quantize the frozen weights to INT4 before training and keep the batch unchanged',
      ],
      a: 1,
      why: 'The model state did not change — only the sequence length grew, so **activation memory** (∝ batch × sequence) is what overflowed. A smaller micro-batch with more [[gradient-accumulation|gradient accumulation]] keeps the global batch, and [[activation-checkpointing|activation checkpointing]] cuts the rest.',
      distract: '**SGD** changes the training dynamics and does not touch activations. **ZeRO-1** shards optimizer states, which were already fitting. **INT4 weights** are an inference technique and do not fit the activation problem.' },

    { id: 'g03', domain: 'gpu',
      q: 'Why does fp16 mixed-precision training need a gradient scaler while bf16 normally does not?',
      opts: [
        'bf16 has fp32\'s exponent range, while fp16\'s narrow range lets small gradients underflow to zero',
        'bf16 has more mantissa bits than fp16, so its gradients are more precise and never round to zero',
        'fp16 Tensor Core kernels compute in fp32 and need the scaler to convert results back',
        'bf16 is only used for weights, so gradients are always kept in fp32 automatically',
      ],
      a: 0,
      why: '[[bf16|bf16]] keeps fp32\'s exponent width (range ~1e±38) with fewer mantissa bits. [[fp16|fp16]] has only 5 exponent bits, so tiny gradients underflow; [[loss-scaling|loss scaling]] shifts them into range before backward and unscales them before the optimizer step.',
      distract: 'bf16 has **fewer** mantissa bits (7) than fp16 (10). Tensor Cores accumulate in fp32 for both formats; that is not what the scaler is for. Autocast does not keep gradients in fp32 just because the format is bf16.' },

    { id: 'g04', domain: 'gpu',
      scenario: 'Training a 1B model at 32k context, the profiler shows that the attention score matrices dominate activation memory.',
      q: 'Which technique removes that quadratic memory term while also speeding attention up?',
      opts: [
        'Gradient accumulation with a micro-batch of 1',
        'ZeRO-3 parameter sharding',
        'FP8 training with Transformer Engine',
        'FlashAttention via PyTorch SDPA',
      ],
      a: 3,
      why: '[[flash-attention|FlashAttention]] tiles Q/K/V into on-chip SRAM, computes softmax incrementally and never stores the s × s matrix, so attention memory becomes linear in sequence length and HBM traffic drops. PyTorch\'s `scaled_dot_product_attention` dispatches to it.',
      distract: '**Micro-batch 1** still materialises an s × s matrix per head. **ZeRO-3** shards parameters, not activations. **FP8** shrinks matmul precision but the s² term remains.' },

    { id: 'g05', domain: 'gpu',
      scenario: 'A model fits comfortably on one GPU. The team has 8 GPUs in one node and wants to finish training faster, with minimal code changes.',
      q: 'Which approach fits best?',
      opts: [
        'Pipeline parallelism with 8 stages, one per GPU',
        'Tensor parallelism of size 8 within the node',
        'Data parallelism with DDP launched by torchrun',
        'ZeRO-3 sharding with CPU offload of optimizer states',
      ],
      a: 2,
      why: 'When the model fits, [[ddp|DDP]] is the simplest and most efficient scaling: each GPU processes a different shard of the data and gradients are [[all-reduce|all-reduced]], overlapped with backward. Throughput scales almost linearly.',
      distract: '**Pipeline** and **tensor parallelism** exist to split a model that does not fit and add bubbles or per-layer communication. **ZeRO-3 with offload** saves memory you do not need to save, at a large speed cost.' },

    { id: 'g06', domain: 'gpu',
      q: 'Compared with plain data parallelism, what does ZeRO stage 2 shard across data-parallel ranks?',
      opts: [
        'Only the optimizer states',
        'The optimizer states and the gradients',
        'The optimizer states, gradients and parameters',
        'The activations of each layer',
      ],
      a: 1,
      why: '[[zero|ZeRO]] stage 1 shards optimizer states; stage 2 adds **gradients** (reduce-scatter so each rank keeps only its slice); stage 3 also shards **parameters**, gathering them just-in-time. Stage 2 keeps communication at roughly DDP\'s volume.',
      distract: '**Optimizer states only** is ZeRO-1. **Plus parameters** is ZeRO-3 / FSDP full shard. ZeRO does not shard **activations** — that is the job of checkpointing or sequence/context parallelism.' },

    { id: 'g07', domain: 'gpu',
      scenario: 'A cluster has nodes of 8 H100s connected by NVLink, and nodes are linked by 400 Gb/s InfiniBand. A 175B model needs tensor parallelism of 8 and pipeline parallelism of 4.',
      q: 'How should the parallel groups be placed?',
      opts: [
        'Tensor-parallel groups inside each node, pipeline stages across nodes, data parallelism over the remaining replicas',
        'Pipeline stages inside each node, tensor-parallel groups spanning nodes over InfiniBand',
        'Data parallelism inside each node, tensor and pipeline groups spread round-robin across nodes',
        'Placement does not matter because NCCL picks the fastest route automatically',
      ],
      a: 0,
      why: '[[tensor-parallelism|Tensor parallelism]] all-reduces activations in every layer, on the critical path, so it must stay on [[nvlink|NVLink]]. [[pipeline-parallelism|Pipeline parallelism]] only sends activations point-to-point between stages, which InfiniBand handles well. DP fills the rest.',
      distract: 'TP across **InfiniBand** puts per-layer all-reduces on a ~10× slower link. Spreading TP **round-robin** has the same problem. NCCL picks the best transport for a given group, but it cannot fix a group whose members are on different nodes.' },

    { id: 'g08', domain: 'gpu',
      scenario: 'A model is split into 8 pipeline stages. Each global batch is divided into 8 micro-batches, and profiling shows GPUs idle almost half the time.',
      q: 'What is the most effective first change?',
      opts: [
        'Reduce the number of micro-batches to 4 to shorten each step',
        'Replace 1F1B scheduling with GPipe scheduling',
        'Raise the micro-batch size so each stage does more work per micro-batch',
        'Increase the number of micro-batches per step (e.g. to 32+) or use interleaved stages',
      ],
      a: 3,
      why: 'The [[pipeline-bubble|bubble]] fraction is (p − 1)/(m + p − 1): with p = m = 8 it is 7/15 ≈ 47%. Raising m to 32 brings it to 7/39 ≈ 18%; interleaved (virtual) stages divide it further.',
      distract: '**Fewer micro-batches** make the bubble larger. **GPipe** has the same bubble as 1F1B but needs more activation memory. A larger micro-batch does not change the bubble *fraction* and raises activation memory.' },

    { id: 'g09', domain: 'gpu',
      scenario: 'A 13B model trains on 16 H100 GPUs (dense BF16 peak ≈ 989 TFLOP/s each) at a measured 48,000 tokens/s in total.',
      q: 'What is the approximate model FLOPs utilization (MFU)?',
      opts: [
        'About 12%',
        'About 79%',
        'About 24%',
        'About 39%',
      ],
      a: 3,
      why: 'Per GPU: 48,000 / 16 = 3,000 tokens/s. [[mfu|MFU]] = 6 × 13e9 × 3,000 / 989e12 ≈ 2.34e14 / 9.89e14 ≈ **24%**.',
      distract: '**12%** uses 3·N instead of 6·N per token. **39%** divides by a wrong peak of ~600 TFLOP/s instead of the H100\'s dense 989. **79%** forgets to divide by the number of GPUs — and would be implausibly high anyway.' },

    { id: 'g10', domain: 'gpu',
      scenario: 'During training, `nvidia-smi` shows GPU utilization oscillating between 20% and 100%, while `top` shows the Python processes pegged at 100% CPU. Step time varies a lot.',
      q: 'What is the most likely bottleneck and fix?',
      opts: [
        'NCCL communication is slow; switch the transport from InfiniBand to TCP sockets',
        'The input pipeline; pre-tokenize the data and use more DataLoader workers with pinned memory',
        'Tensor Cores are underused; switch the training precision from bf16 to fp32',
        'Thermal throttling; lower the batch size so the GPUs draw less power and cool down',
      ],
      a: 1,
      why: 'Saw-tooth GPU utilization with busy CPUs is the classic **data-loader** signature — the GPU waits for batches. Pre-tokenize offline, raise `num_workers`, use `pin_memory` and persistent workers, and read from fast local storage.',
      distract: 'Sockets are **slower** than InfiniBand. **fp32** would make the GPU slower, not busier. Throttling shows as lower clocks and high temperatures, not busy CPUs.' },

    { id: 'g11', domain: 'gpu',
      q: 'You need a single timeline that shows CPU threads, CUDA API calls, GPU kernels, NCCL operations and memory copies for a few training steps across GPUs. Which tool fits?',
      opts: [
        'Nsight Compute',
        'nvidia-smi dmon',
        'Nsight Systems',
        'DCGM health checks',
      ],
      a: 2,
      why: '[[nsight-systems|Nsight Systems]] (`nsys profile`) records a system-wide timeline with CPU, CUDA API, kernels, NCCL, memcpy and NVTX ranges — ideal for finding gaps and checking compute/communication overlap.',
      distract: '[[nsight-compute|Nsight Compute]] analyses individual kernels in depth, not the whole timeline. **nvidia-smi dmon** gives coarse per-second counters. **DCGM health checks** test hardware, not program behaviour.' },

    { id: 'g12', domain: 'gpu',
      scenario: 'A 32-node training job repeatedly hangs at the same point and then fails with an NCCL watchdog timeout. Single-node runs work.',
      q: 'What is the best first diagnostic step?',
      opts: [
        'Rerun with NCCL_DEBUG=INFO, check transports and which rank stalls, and run nccl-tests between nodes',
        'Increase the NCCL timeout to two hours in init_process_group and restart from the checkpoint',
        'Switch the optimizer to SGD so that less gradient data is communicated each step',
        'Disable mixed precision and rerun in fp32 to rule out NaNs in the gradients',
      ],
      a: 0,
      why: '`NCCL_DEBUG=INFO` shows each rank\'s topology and transport (e.g. falling back to sockets instead of IB) and errors; combined with nccl-tests and checking for a rank that never reaches the collective (rank divergence, slow I/O), it pinpoints the cause.',
      distract: 'A longer **timeout** hides rather than diagnoses a hang. The **optimizer** does not change collective behaviour. **NaNs** do not cause hangs; the job would keep running with NaN losses.' },
  ],
}
