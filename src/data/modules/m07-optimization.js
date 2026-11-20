export default {
  id: 'optimization', n: 7, domain: 'optimization', weight: 17,
  title: 'Model Optimization',
  blurb: 'Make a trained model smaller, faster and cheaper to serve: quantize to FP8/INT8/INT4/FP4, prune and distill, understand exactly what an inference engine does with the KV cache and the batch, and build and tune TensorRT-LLM deployments behind Triton.',
  hours: '7–9 h',
  lessons: [
    /* ------------------------------------------------------------------ */
    {
      id: 'quantization',
      title: 'Lab: quantization — FP8, INT8, INT4, GPTQ, AWQ, SmoothQuant',
      minutes: 40,
      outcomes: [
        'Compare FP16, BF16, FP8 (E4M3/E5M2), INT8, INT4 and NVFP4 by range, precision, memory and hardware support',
        'Compute symmetric and asymmetric scales at per-tensor, per-channel and per-group granularity',
        'Choose between PTQ and QAT, and between weight-only (GPTQ, AWQ) and weight-and-activation (SmoothQuant, FP8) schemes',
        'Quantize a model with bitsandbytes and TensorRT Model Optimizer and measure memory and perplexity',
        'Estimate the memory and speed trade-offs of each scheme, including KV-cache quantization',
      ],
      blocks: [
        { t: 'lede', x: '[[quantization|Quantization]] is the highest-leverage optimisation you can apply to a trained LLM: halve or quarter the bytes per weight and you halve or quarter the memory traffic that dominates generation. It is also the largest topic in the largest exam domain. This lesson gives you the number formats, the maths of scales, the main algorithms, and a lab where you measure the trade-off yourself.' },

        { t: 'h2', x: 'Number formats' },
        { t: 'table', head: ['Format', 'Bits (sign-exp-mantissa)', 'Max magnitude / levels', 'Tensor Core support', 'Typical LLM use'], rows: [
          ['FP32', '1-8-23', '≈3.4×10³⁸', 'TF32 path on Ampere+', 'Master weights, optimizer states'],
          ['[[bf16|BF16]]', '1-8-7', '≈3.4×10³⁸ (FP32 range, ~3 significant digits)', 'Ampere+', 'Training and baseline inference'],
          ['[[fp16|FP16]]', '1-5-10', '65,504', 'Volta+', 'Inference baseline; training needs loss scaling'],
          ['[[fp8|FP8 E4M3]]', '1-4-3', '±448', 'Ada, Hopper, Blackwell', 'Weights and activations for inference (W8A8), KV cache'],
          ['FP8 E5M2', '1-5-2', '±57,344', 'Ada, Hopper, Blackwell', 'Gradients in FP8 training (needs range)'],
          ['[[int8|INT8]]', 'integer', '256 levels (−128…127)', 'Turing+', 'W8A16 weight-only, W8A8 SmoothQuant, KV cache'],
          ['[[int4|INT4]]', 'integer', '16 levels (−8…7)', 'Weight-only kernels on Ampere+', 'W4A16 with GPTQ / AWQ, group size 64–128'],
          ['[[fp4|FP4 E2M1 / NVFP4]]', '1-2-1 + block scales', '±6 (values 0, 0.5, 1, 1.5, 2, 3, 4, 6)', 'Blackwell', 'W4A4: FP8 scale per 16 values + FP32 per-tensor scale'],
        ]},
        { t: 'p', x: 'Two rules make the table memorable. **Exponent bits buy range, mantissa bits buy precision** — that is why E4M3 (more precision) is used for forward-pass tensors and E5M2 (more range) for gradients. And **integers are uniform while floats are logarithmic**: INT8 spends its 256 levels evenly across the range, FP8 concentrates them near zero, where most weights and activations live, which is one reason FP8 PTQ is usually easier than INT8 W8A8.' },

        { t: 'h2', x: 'Scales, zero points and granularity' },
        { t: 'p', x: 'Integer quantization maps a real tensor onto integers with a **scale** s (and optionally a **zero point** z). Dequantization multiplies back. All the cleverness of quantization algorithms is in choosing s well.' },
        { t: 'specs', rows: [
          ['Symmetric (INT8)', 's = max|x| / 127 · q = clamp(round(x / s), −128, 127) · x̂ = s·q. Zero maps exactly to 0. Standard for weights.'],
          ['Asymmetric (UINT8)', 's = (max − min) / 255 · z = round(−min / s) · q = clamp(round(x / s) + z, 0, 255) · x̂ = s·(q − z). Uses the full range for skewed data (e.g. post-ReLU/GELU activations).'],
          ['Per-tensor', 'One scale for the whole tensor. Cheapest, but a single outlier stretches the scale and wastes levels for everything else.'],
          ['Per-channel / per-token', 'One scale per output channel of a weight (row), or per token row of an activation. Standard for INT8 weights.'],
          ['Per-group / block', 'One scale per 16–128 consecutive values. Required for 4-bit: group 128 with an FP16 scale adds 16/128 = 0.125 bits per weight (≈4.125 bits total, more with zero points).'],
          ['Static vs dynamic', 'Static activation scales are fixed from calibration data (fast); dynamic scales are computed per batch/token at runtime (more accurate, a little overhead).'],
        ]},
        { t: 'code', lang: 'python', title: 'quant_math.py — see granularity and outliers in action (CPU is fine)', x: `import torch
torch.manual_seed(0)

W = torch.randn(1024, 1024) * 0.02             # a weight matrix (out_features, in_features)
idx = torch.randint(0, W.numel(), (64,))
W.view(-1)[idx] *= 40                          # a few large outliers, like real LLM weights

def quant_sym(x, bits, dim=None):
    qmax = (1 << (bits - 1)) - 1               # 127 for INT8, 7 for INT4
    amax = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    scale = amax.clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale                           # dequantized ("fake-quant") tensor

def quant_group(x, bits, group=128):
    o, i = x.shape
    return quant_sym(x.reshape(o, i // group, group), bits, dim=-1).reshape(o, i)

def quant_fp8(x):                              # per-tensor scaled FP8 E4M3 (PyTorch >= 2.1)
    scale = x.abs().amax() / 448.0
    return (x / scale).to(torch.float8_e4m3fn).float() * scale

def rel_err(a, b):
    return ((a - b).norm() / a.norm()).item()

print(f"INT8 per-tensor   {rel_err(W, quant_sym(W, 8)):.4f}")
print(f"INT8 per-channel  {rel_err(W, quant_sym(W, 8, dim=1)):.4f}")
print(f"FP8  per-tensor   {rel_err(W, quant_fp8(W)):.4f}")
print(f"INT4 per-channel  {rel_err(W, quant_sym(W, 4, dim=1)):.4f}")
print(f"INT4 group=128    {rel_err(W, quant_group(W, 4, 128)):.4f}")
# Expect: finer granularity -> lower error; INT4 per-channel is poor, per-group recovers much of it;
# FP8's log spacing copes with the outliers better than INT8 per-tensor.` },

        { t: 'h2', x: 'PTQ vs QAT, weight-only vs weight-and-activation' },
        { t: 'vs', cards: [
          { h: '[[ptq|PTQ]] — post-training quantization', items: [
            'No training: weights + a small [[calibration]] set (128–512 samples)',
            'Minutes to a few hours, even for 70B models',
            'FP8 and INT8 W8A8: usually near-lossless',
            'INT4/FP4: small but measurable loss; algorithm and calibration matter',
            'The default first step',
          ]},
          { h: '[[qat|QAT]] — quantization-aware training', items: [
            'Fine-tune with fake-quant in the forward pass, straight-through gradients',
            'Weights learn to live on the quantized grid',
            'Recovers accuracy at 4 bits and below; often combined with distillation (QAD)',
            'Costs a training run and training data',
            'Use when PTQ misses your accuracy target',
          ]},
        ]},
        { t: 'p', x: 'The second axis is **what** you quantize. **Weight-only** schemes (W8A16, W4A16) store weights in low precision and dequantize them inside the matmul kernel, computing in FP16/BF16. They shrink memory and speed up memory-bound decode, but do not speed up compute-bound work. **Weight-and-activation** schemes (W8A8 INT8, FP8, NVFP4 W4A4) run the matmul itself in low precision on Tensor Cores, speeding up compute-bound prefill and large-batch decode too — but activations have outliers, which is the hard part.' },
        { t: 'table', head: ['Algorithm', 'Scheme', 'Key idea', 'Notes'], rows: [
          ['[[gptq|GPTQ]]', 'W4A16 / W3A16 (weight-only)', 'Quantize columns one at a time; use approximate Hessian from calibration activations to adjust remaining weights and absorb the error', 'Strong accuracy; slower to apply; can overfit calibration data'],
          ['[[awq|AWQ]]', 'W4A16 (weight-only)', '~1% of channels are salient — find them by activation magnitude and scale them up before quantizing so they keep more precision', 'Fast, no backprop/reconstruction, generalises well; `INT4_AWQ` in ModelOpt'],
          ['[[smoothquant|SmoothQuant]]', 'W8A8 INT8', 'Divide activations by a per-channel factor s and multiply weights by s, moving outliers from activations to weights; α ≈ 0.5 balances them', 'Enables INT8 GEMMs on Ampere-class GPUs'],
          ['FP8 PTQ', 'W8A8 FP8', 'Per-tensor (or finer) scales from calibration amax; E4M3 for weights and activations', 'Near-lossless and simple on Hopper/Ada/Blackwell — usually the first choice there'],
          ['NVFP4 PTQ / QAT', 'W4A4 FP4', 'E2M1 values with FP8 scale per 16-value block', 'Blackwell; ~2× FP8 math throughput; QAT/QAD closes remaining gaps'],
          ['LLM.int8() / NF4 (bitsandbytes)', 'W8 mixed / W4 weight-only', 'INT8 with outlier features kept in FP16; NF4 = normal-float 4-bit data type (QLoRA)', 'Great for memory (fine-tuning, experiments); not a high-throughput serving path'],
        ]},
        { t: 'callout', kind: 'warn', title: 'Activation outliers are the whole story for W8A8', x: 'LLM activations develop a few channels with magnitudes 10–100× larger than the rest (from around 6.7B parameters upward). With per-tensor INT8, those channels set the scale and every other channel collapses onto a few levels. Fixes: keep outliers in FP16 (LLM.int8()), migrate them into weights (SmoothQuant), use finer scales (per-token), or use FP8, whose logarithmic spacing tolerates them far better.' },

        { t: 'h2', x: 'KV-cache quantization and calibration' },
        { t: 'p', x: 'The [[kv-cache|KV cache]] is quantized independently of the weights. Storing it in FP8 (or INT8) halves its memory and the bandwidth spent reading it every decode step — in long-context, high-concurrency serving this often matters more than weight precision. See [[kv-cache-quantization|KV-cache quantization]]; you will size the cache in the inference-engine lesson.' },
        { t: 'p', x: 'Every static scheme needs [[calibration]]: run a few hundred representative samples, record activation statistics (max, percentiles, or MSE-optimal clipping), derive scales. Calibrate on data that looks like production — chat-formatted prompts for a chat model, code for a code model. Calibrating a code assistant on news articles is a classic source of mysterious PTQ accuracy loss.' },

        { t: 'lab', title: 'Quantize a model and measure memory vs perplexity', x: 'Load Qwen2.5-0.5B in FP16, 8-bit (LLM.int8()) and 4-bit NF4 with bitsandbytes, and compare memory footprint, perplexity and generation speed. Then, on a Hopper/Ada/Blackwell GPU, produce FP8 and INT4-AWQ checkpoints with TensorRT Model Optimizer that TensorRT-LLM can serve.', gpu: 'Free Colab T4 for bitsandbytes; 1× Ada/Hopper GPU (e.g. L4, L40S, H100) for FP8 with ModelOpt', time: '45 min', steps: [
          'Install: `pip install transformers accelerate bitsandbytes datasets`.',
          'Run `bnb_compare.py`: it loads each variant, prints `get_memory_footprint()`, perplexity on 32k WikiText-2 tokens and tokens/s for a short generation.',
          'Record the results in a table; note that perplexity barely moves at 8-bit and moves a little at 4-bit, while 8-bit is often slower than FP16.',
          'On an FP8-capable GPU: `pip install "nvidia-modelopt[hf]"` and run `modelopt_ptq.py` with `FP8_DEFAULT_CFG`, then again with `INT4_AWQ_CFG`.',
          'Load the exported checkpoint with the TensorRT-LLM LLM API (next lessons) and compare its perplexity/benchmarks with the BF16 baseline.',
        ]},
        { t: 'code', lang: 'python', title: 'bnb_compare.py — FP16 vs 8-bit vs 4-bit NF4', x: `import math, time, torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(model_id)
text = "\\n\\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
ids = tok(text, return_tensors="pt").input_ids[:, :32768]       # 32k tokens keeps this quick

@torch.no_grad()
def perplexity(model, window=1024):
    nll, n = 0.0, 0
    for i in range(0, ids.size(1) - 1, window):                  # disjoint windows: fine for A/B comparison
        x = ids[:, i:i + window].to(model.device)
        logits = model(input_ids=x).logits.float()
        nll += F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
        n += x.size(1) - 1
    return math.exp(nll / n)

@torch.no_grad()
def tokens_per_s(model, new=128):
    x = tok("The history of the GPU", return_tensors="pt").input_ids.to(model.device)
    torch.cuda.synchronize(); t = time.time()
    model.generate(x, max_new_tokens=new, min_new_tokens=new, do_sample=False)
    torch.cuda.synchronize()
    return new / (time.time() - t)

nf4 = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,           # quantize the scales too (~0.4 bit/param saved)
    bnb_4bit_compute_dtype=torch.float16)     # T4 has no bf16; use bfloat16 on Ampere+
variants = {                                  # name -> quantization_config (None = plain fp16)
    "fp16": None,
    "int8 (LLM.int8)": BitsAndBytesConfig(load_in_8bit=True),
    "nf4 (4-bit)": nf4,
}
for name, qcfg in variants.items():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.float16, quantization_config=qcfg).eval()
    print(f"{name:16s} mem={model.get_memory_footprint() / 1e9:.2f} GB  "
          f"ppl={perplexity(model):.2f}  speed={tokens_per_s(model):.0f} tok/s")
    del model; torch.cuda.empty_cache()
# Note: bitsandbytes only quantizes nn.Linear layers; the large embedding table of a small
# model stays in fp16, so a 0.5B model shrinks much less than 4x.` },
        { t: 'code', lang: 'python', title: 'modelopt_ptq.py — FP8 / INT4-AWQ with TensorRT Model Optimizer', x: `# pip install "nvidia-modelopt[hf]"   (FP8 needs an Ada, Hopper or Blackwell GPU)
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")

# Calibration set: a few hundred samples that look like production traffic.
# (WikiText here for convenience; use your own chat-formatted prompts in practice.)
calib = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
         if len(t) > 200][:256]

def forward_loop(m):
    for text in calib:
        x = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(m.device)
        m(x)                                   # ModelOpt observers record activation amax

cfg = mtq.FP8_DEFAULT_CFG                      # alternatives: mtq.INT4_AWQ_CFG, mtq.INT8_SMOOTHQUANT_CFG,
                                               #               mtq.NVFP4_DEFAULT_CFG (Blackwell)
model = mtq.quantize(model, cfg, forward_loop) # inserts quantizers + calibrates in place
mtq.print_quant_summary(model)

with torch.inference_mode():
    export_hf_checkpoint(model, export_dir="qwen2.5-0.5b-instruct-fp8")   # HF-style quantized checkpoint
tok.save_pretrained("qwen2.5-0.5b-instruct-fp8")
# Serve it: from tensorrt_llm import LLM; LLM(model="qwen2.5-0.5b-instruct-fp8")  (also loadable by vLLM)` },
        { t: 'p', x: 'This is the NVIDIA production path: [[model-optimizer|TensorRT Model Optimizer]] does PTQ/QAT (and pruning, distillation and speculative-head training), and [[tensorrt-llm|TensorRT-LLM]] runs the resulting checkpoint with fused low-precision kernels. NVIDIA also publishes ready-made FP8/NVFP4 checkpoints of popular models on Hugging Face.' },

        { t: 'h2', x: 'The trade-off table' },
        { t: 'p', x: 'Worked memory example — weights only, 8B-parameter model: BF16 = 8×10⁹ × 2 B = **16 GB**; FP8/INT8 = **8 GB**; INT4 group-128 ≈ 8×10⁹ × 4.125 / 8 ≈ **4.1 GB** (≈5 GB in practice, since embeddings and lm_head often stay 16-bit). For a 70B model: 140 GB → 70 GB → ~38 GB, i.e. from two 80 GB GPUs down to one.' },
        { t: 'table', head: ['Scheme', 'Weight memory vs BF16', 'Typical speed effect (H100-class)', 'Typical accuracy effect', 'Hardware'], rows: [
          ['FP8 W8A8 (+ FP8 KV)', '0.5×', '~1.3–1.6× throughput; faster prefill and decode', 'Usually < 1 point on benchmarks', 'Ada / Hopper / Blackwell'],
          ['INT8 SmoothQuant W8A8', '0.5×', '~1.2–1.5× throughput', 'Small with good α and calibration', 'Ampere+'],
          ['INT8 weight-only W8A16', '0.5×', 'Faster small-batch decode; no prefill gain', 'Negligible', 'Ampere+'],
          ['INT4 AWQ / GPTQ W4A16', '~0.28×', 'Up to ~2–3× at batch 1; gain fades at large batch (dequant + compute-bound)', '~1–3 points; task-dependent', 'Ampere+'],
          ['NVFP4 W4A4', '~0.28×', '~2× FP8 Tensor Core throughput', 'Small with calibration; QAT closes the gap', 'Blackwell'],
          ['FP8 KV cache only', 'KV 0.5×', '~2× concurrent tokens; less KV bandwidth', 'Usually negligible', 'Hopper+'],
        ]},
        { t: 'callout', kind: 'trap', title: '"Fewer bits" does not always mean "faster"', x: [
          'bitsandbytes LLM.int8() is frequently **slower** than FP16 because of its outlier decomposition — it is a memory tool. Production speedups come from fused kernels in TensorRT-LLM, vLLM or similar.',
          'INT4 weight-only speeds up memory-bound decode, but at large batch the GEMMs become compute-bound and still run in FP16 — FP8 W8A8 can then beat INT4 on throughput.',
          'Always measure perplexity/benchmarks **and** latency/throughput at your real batch size.',
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Format facts: E4M3 vs E5M2, FP8 needs Hopper/Ada (Blackwell for FP4), BF16 has FP32 range.',
          'Method matching: "INT8 activations break because of outliers" → SmoothQuant; "4-bit weights, no retraining, robust" → AWQ; "Hessian-based error compensation" → GPTQ; "PTQ accuracy insufficient" → QAT.',
          'Memory arithmetic: bytes per parameter × parameters, plus group-scale overhead. Weight-only vs W8A8 and which phase (prefill/decode) each speeds up.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Exponent bits give range, mantissa bits give precision; FP8 E4M3 for inference tensors, E5M2 for gradients; NVFP4 uses 16-value blocks with FP8 scales.',
          'Scale granularity is the main accuracy lever: per-tensor < per-channel < per-group; INT4 needs per-group scales (≈0.125 extra bits at group 128).',
          'PTQ first (FP8 is near-lossless on Hopper+); QAT when 4-bit PTQ is not good enough.',
          'Weight-only (GPTQ, AWQ) helps memory-bound decode; W8A8 (SmoothQuant, FP8) and W4A4 (NVFP4) also accelerate compute-bound work.',
          'Quantize the KV cache separately; calibrate with production-like data; validate both accuracy and speed.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'pruning-distillation',
      title: 'Pruning and knowledge distillation (Minitron)',
      minutes: 22,
      outcomes: [
        'Distinguish unstructured, 2:4 semi-structured and structured (width/depth) pruning and their real speedups',
        'Implement a knowledge-distillation loss with temperature-scaled soft targets and KL divergence',
        'Describe the Minitron prune-then-distill recipe and its results',
        'Choose between quantization, pruning, distillation and their combinations for a given constraint',
      ],
      blocks: [
        { t: 'lede', x: 'Quantization keeps every parameter and stores it in fewer bits. Pruning and distillation take the other route: fewer parameters. Done naively, pruning wrecks a model; done the NVIDIA way — structured pruning followed by distillation from the original — it produces small models that beat same-size models trained from scratch, at a fraction of the training cost.' },

        { t: 'h2', x: 'Three kinds of pruning' },
        { t: 'table', head: ['Type', 'What is removed', 'Real GPU speedup?', 'Examples'], rows: [
          ['Unstructured', 'Individual weights anywhere (e.g. smallest |w|, or |w|·‖x‖)', 'Rarely — dense GEMMs do the same work on zeros', 'Magnitude pruning, Wanda, SparseGPT'],
          ['Semi-structured [[structured-sparsity|2:4]]', '2 of every 4 consecutive weights', 'Yes on Ampere+ Sparse Tensor Cores — up to 2× on those GEMMs, less end-to-end', 'ASP, SparseGPT 2:4, ModelOpt sparsity'],
          ['Structured — width', 'Attention heads, MLP neurons, embedding/hidden channels', 'Yes — result is a smaller dense model', 'Minitron width pruning'],
          ['Structured — depth', 'Whole transformer layers', 'Yes — fewer sequential layers, good for latency', 'Minitron depth pruning, layer dropping'],
        ]},
        { t: 'p', x: '[[pruning|Pruning]] needs an **importance** score. For unstructured pruning, Wanda scores each weight by |W_ij| × ‖X_j‖ (weight magnitude times input activation norm) on calibration data and needs no retraining at 50% sparsity on large models. For structured pruning, Minitron computes activation-based importance of heads, neurons and channels from forward passes on ~1k calibration samples, and layer importance from how much the LM loss (or the hidden state) changes when a layer is removed.' },
        { t: 'code', lang: 'python', title: 'sparsity_2_4.py — build a 2:4 mask and use semi-structured sparse kernels', x: `import torch
from torch.sparse import to_sparse_semi_structured   # PyTorch >= 2.1, Ampere+ GPU, fp16/bf16

def mask_2_4(W: torch.Tensor) -> torch.Tensor:
    """Keep the 2 largest-magnitude weights in every group of 4 along the input dim."""
    groups = W.abs().reshape(-1, 4)
    keep = groups.topk(2, dim=1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(1, keep, True)
    return mask.reshape(W.shape)

linear = torch.nn.Linear(4096, 4096, bias=False).half().cuda()   # dims must be multiples of 64
with torch.no_grad():
    linear.weight.mul_(mask_2_4(linear.weight))                  # 50% zeros in a 2:4 pattern
dense_w = linear.weight.detach().clone()
linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))  # compressed + metadata

x = torch.randn(4096, 4096, dtype=torch.half, device="cuda")
print(torch.allclose(linear(x), x @ dense_w.t(), atol=1e-1, rtol=1e-2))       # same math, sparse kernel
# In practice: prune with a smarter score (SparseGPT/Wanda), then fine-tune to recover accuracy.` },
        { t: 'callout', kind: 'trap', title: '50% sparse ≠ 2× faster', x: '50% unstructured sparsity leaves the dense matmul doing exactly the same work, and sparse kernels only win at very high sparsity. Even 2:4 gives up to 2× only on the sparse GEMMs; attention, norms and memory-bound decode are unaffected, so end-to-end gains are typically far smaller. If you need a guaranteed speedup, prune structurally or quantize.' },

        { t: 'h2', x: 'Knowledge distillation' },
        { t: 'p', x: 'In [[distillation|knowledge distillation]] a small student learns from a large teacher\'s full output distribution. A one-hot label says "the next token is *cat*"; the teacher\'s distribution says "*cat* 0.6, *kitten* 0.25, *dog* 0.05…" — far more information per token. A **temperature** τ > 1 softens both distributions (logits ÷ τ) so the small probabilities ("dark knowledge") contribute to the gradient.' },
        { t: 'specs', rows: [
          ['Loss', 'L = α · CE(student, labels) + (1 − α) · τ² · KL(p_teacher^τ ‖ p_student^τ)'],
          ['Why τ²', 'Gradients of the softened KL scale as 1/τ²; multiplying by τ² keeps the soft-target term\'s magnitude comparable to CE as τ changes'],
          ['KL direction', 'Forward KL(teacher ‖ student) is mean-seeking (cover all teacher modes) — the standard; reverse KL is mode-seeking (used by some generative-distillation methods)'],
          ['Typical values', 'τ = 1–4; α from 0 (pure distillation, as in Minitron) to 0.5'],
          ['Requirement', 'Logit distillation needs teacher and student to share the tokenizer/vocabulary'],
          ['Other variants', 'Intermediate-state distillation (match hidden states via a projection); sequence-level distillation (SFT on teacher-generated outputs — what most "synthetic data" recipes are)'],
        ]},
        { t: 'code', lang: 'python', title: 'distill.py — logit distillation, Qwen2.5-1.5B teacher → 0.5B student', x: `import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")          # same tokenizer family as the teacher
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16).cuda().eval()
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).cuda()
opt = torch.optim.AdamW(student.parameters(), lr=1e-5)

def kd_loss(s_logits, t_logits, labels, T=2.0, alpha=0.5):
    # shift: position t predicts token t+1
    s, t, y = s_logits[:, :-1], t_logits[:, :-1], labels[:, 1:]
    mask = y != -100
    s, t, y = s[mask].float(), t[mask].float(), y[mask]           # (N, V) over real tokens only
    kl = F.kl_div(F.log_softmax(s / T, dim=-1),                   # input: student log-probs
                  F.log_softmax(t / T, dim=-1),                   # target: teacher log-probs
                  log_target=True, reduction="batchmean") * T * T # KL(teacher || student) * T^2
    ce = F.cross_entropy(s, y)
    return alpha * ce + (1 - alpha) * kl

batch = tok(["The mitochondria is the powerhouse of the cell because"], return_tensors="pt").to("cuda")
labels = batch.input_ids.masked_fill(batch.attention_mask == 0, -100)

with torch.no_grad():
    t_logits = teacher(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
s_logits = student(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
loss = kd_loss(s_logits, t_logits, labels)
loss.backward(); opt.step(); opt.zero_grad()
print(f"kd loss {loss.item():.3f}")
# Real runs: loop over a large corpus; cache teacher top-k logits if the teacher is expensive.` },
        { t: 'p', x: 'Note the vocabulary check: Qwen2.5-0.5B and 1.5B share both tokenizer and embedding size, so logits line up. At NVIDIA scale the same loss runs in [[nemo|NeMo]] / [[megatron|Megatron]] (and `modelopt.torch.distill`), with teacher and student sharded across GPUs.' },

        { t: 'h2', x: 'Minitron: prune, then distill' },
        { t: 'flow', steps: [
          { b: 'Teacher correction', s: 'Lightly fine-tune the original model on the distillation dataset (fixes distribution shift when the original pretraining data is not available)' },
          { b: 'Importance', s: 'Forward ~1k calibration samples; score heads, MLP neurons, embedding channels and layers by activations / loss impact' },
          { b: 'Prune', s: 'Width: trim neurons, heads, hidden size. Depth: drop the least important layers' },
          { b: 'Distill', s: 'Retrain the pruned student with logit KL against the (corrected) teacher — tens of billions of tokens, not trillions' },
          { b: 'Align', s: 'Instruction-tune / align the student as usual' },
        ]},
        { t: 'p', x: '[[minitron|Minitron]] results that come up repeatedly: Minitron 8B and 4B were derived from Nemotron-4 15B using up to ~40× fewer training tokens per model than training from scratch; Llama-3.1-Minitron 4B came from Llama 3.1 8B using roughly 94B distillation tokens; Mistral-NeMo-Minitron 8B came from Mistral NeMo 12B. The findings: **width pruning generally recovers accuracy better than depth pruning** at the same size, while **depth pruning gives larger latency gains** (fewer sequential layers); distillation beats conventional fine-tuning for recovery; and pruning a trained large model then distilling beats training the small model from scratch.' },

        { t: 'h2', x: 'Which technique when?' },
        { t: 'table', head: ['Constraint / goal', 'First choice', 'Why'], rows: [
          ['Same model, cut memory and latency now', 'PTQ (FP8; INT4-AWQ if memory-bound)', 'Hours, no training data, near-lossless at 8 bits'],
          ['Need a smaller model family for edge / cheaper tiers', 'Structured pruning + distillation (Minitron)', 'Real dense speedup; far cheaper than pretraining'],
          ['Task-specific small model from a big general one', 'Distillation (logit or sequence-level on synthetic data)', 'Student learns the teacher\'s behaviour on your distribution'],
          ['Ampere+ GPU, want extra GEMM throughput', '2:4 sparsity + fine-tune', 'Hardware-supported pattern; combine with quantization'],
          ['Maximum compression', 'Prune + distill, then quantize (FP8/INT4/NVFP4)', 'Techniques compose: fewer params × fewer bits'],
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Which pruning gives real GPU speedups (structured, 2:4 on Ampere+) vs which mostly does not (unstructured).',
          'Distillation mechanics: soft targets, temperature, KL(teacher ‖ student), the τ² factor, shared vocabulary.',
          'Minitron: prune (width or depth) then distill; width = better accuracy, depth = better latency; orders of magnitude fewer tokens than from-scratch training.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Unstructured sparsity rarely speeds up GPUs; 2:4 sparsity uses Sparse Tensor Cores (Ampere+); structured pruning yields a smaller dense model.',
          'Distillation trains a student on teacher soft targets: α·CE + (1 − α)·τ²·KL(teacher ‖ student).',
          'Minitron = importance-based structured pruning + distillation; width prunes recover better, depth prunes run faster.',
          'Techniques compose: prune and distill for a smaller architecture, then quantize for fewer bits.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'inference-engine',
      title: 'Inside the inference engine: KV cache, paged attention, in-flight batching, speculative decoding',
      minutes: 25,
      outcomes: [
        'Explain why prefill is compute-bound and decode is memory-bound using arithmetic intensity',
        'Calculate KV-cache size per token, per sequence and per GPU, and the resulting concurrency',
        'Describe paged attention, in-flight batching, chunked prefill and prefix caching and what each improves',
        'Estimate the speedup from speculative decoding and compare draft models, Medusa and EAGLE',
        'Explain how FlashAttention/FlashDecoding, CUDA Graphs and tensor parallelism reduce inference latency',
      ],
      blocks: [
        { t: 'lede', x: 'Every modern serving engine — [[tensorrt-llm|TensorRT-LLM]], [[vllm|vLLM]], SGLang — is built from the same handful of ideas. Each one exists because of two facts: generation has a compute-bound phase and a memory-bound phase, and the [[kv-cache|KV cache]] is big. Understand those and every knob in the next lesson makes sense.' },

        { t: 'h2', x: 'Prefill vs decode' },
        { t: 'vs', cards: [
          { h: '[[prefill|Prefill]] (context phase)', items: [
            'All prompt tokens processed in one parallel forward pass',
            'Matrix–matrix multiplies (prompt length × hidden)',
            'Compute-bound; writes the KV cache for the prompt',
            'Determines time to first token (TTFT)',
            'Cost grows with prompt length (attention quadratically)',
          ]},
          { h: '[[decode|Decode]] (generation phase)', items: [
            'One token per sequence per step, strictly sequential',
            'Each step reads all weights + that sequence\'s KV cache',
            'Memory-bandwidth-bound at small/medium batch',
            'Determines time per output token (TPOT) / inter-token latency',
            'Gets cheaper per token with batching, quantization, speculation',
          ]},
        ]},
        { t: 'p', x: '[[arithmetic-intensity|Arithmetic intensity]] makes this precise. In decode at batch 1, each weight (2 bytes in BF16) is used for one multiply-add (2 FLOPs): **~1 FLOP per byte**. An H100 SXM offers ~989 TFLOPS dense BF16 and ~3.35 TB/s of [[hbm|HBM]] bandwidth — a ridge point of ~295 FLOPs/byte. Decode runs ~300× below it: the Tensor Cores sit idle waiting for memory.' },
        { t: 'specs', rows: [
          ['Batch-1 decode bound, 8B BF16', '16 GB of weights per step ÷ 3.35 TB/s ≈ 4.8 ms → at most ~210 tokens/s per sequence'],
          ['Same model, INT4 weights (~4.5 GB)', '≈1.3 ms per step → ceiling ~700 tokens/s — why weight-only quantization helps decode'],
          ['Batch B decode', 'Weight bytes are shared by all B sequences → intensity ≈ B FLOPs/byte for the GEMMs; throughput grows almost linearly with B until compute- or KV-bound'],
          ['But attention in decode', 'Each sequence reads its own KV cache, so attention stays memory-bound at any batch — hence KV quantization, GQA and FlashDecoding'],
          ['Prefill of 2,048 tokens', 'Each weight reused 2,048× → intensity ~2,000 FLOPs/byte → compute-bound'],
        ]},

        { t: 'h2', x: 'KV-cache math (worked example)' },
        { t: 'p', x: 'Per token, every layer stores one key and one value vector per KV head: **bytes/token = 2 × layers × kv_heads × head_dim × bytes_per_element**. Multiply by tokens in flight to get the cache size.' },
        { t: 'table', head: ['Model (BF16 KV)', 'Layers · KV heads · head dim', 'KV per token', '8k-token sequence', '128k-token sequence'], rows: [
          ['Llama 3.1 8B (GQA)', '32 · 8 · 128', '2·32·8·128·2 B = **128 KiB**', '1 GiB', '16 GiB'],
          ['Same, without GQA (32 KV heads)', '32 · 32 · 128', '512 KiB', '4 GiB', '64 GiB'],
          ['Llama 3.1 70B (GQA)', '80 · 8 · 128', '**320 KiB**', '2.5 GiB', '40 GiB'],
          ['Llama 3.1 8B, FP8 KV', '32 · 8 · 128', '64 KiB', '0.5 GiB', '8 GiB'],
        ]},
        { t: 'p', x: 'Capacity on one 80 GB H100 serving Llama 3.1 8B in BF16: weights 16 GB, reserve ~8 GB for activations and runtime → ~56 GB for KV ≈ 56 GiB ÷ 128 KiB ≈ **450k tokens** in flight. At an average of 4k tokens per request that is ~110 concurrent sequences; an FP8 KV cache doubles it to ~220. The KV cache, not the weights, is what limits concurrency — and concurrency is what drives throughput in the memory-bound decode phase.' },

        { t: 'h2', x: 'Paged attention and in-flight batching' },
        { t: 'p', x: 'Early servers reserved a contiguous KV buffer of max_seq_len for every request. Most requests finish far shorter, so 60–80% of KV memory was wasted to reservation and fragmentation. [[paged-attention|Paged attention]] splits the cache into fixed-size blocks (e.g. 16–64 tokens) allocated on demand from a pool and tracked by a per-sequence block table — waste drops to the last partial block (<4% in the vLLM paper), so 2–4× more sequences fit.' },
        { t: 'p', x: 'Static batching waits for the slowest request in a batch: if one answer is 1,000 tokens and the rest are 50, the batch runs 1,000 steps with mostly-empty slots. [[inflight-batching|In-flight (continuous) batching]] schedules per iteration: finished sequences leave immediately, waiting requests join at the next step, and prefill and decode tokens of different requests share the same forward pass. Paged KV makes this cheap, because joining requests just grab free blocks.' },
        { t: 'flow', steps: [
          { b: 'Step t', s: 'Batch = decode tokens of A, B, C + a prefill chunk of new request D' },
          { b: 'A finishes', s: 'A emits EOS; its KV blocks return to the pool' },
          { b: 'Step t+1', s: 'E joins (prefill) in A\'s place; B, C, D decode — no idle slots' },
          { b: 'Memory pressure', s: 'If the pool runs out, the scheduler pauses or evicts a request (policy-dependent)' },
        ]},
        { t: 'ul', items: [
          '**[[chunked-prefill|Chunked prefill]]**: a 32k-token prompt is split into chunks (bounded by a per-step token budget) interleaved with decode steps, so it does not freeze everyone else\'s streams. Trades a little TTFT for that request for stable TPOT for all.',
          '**[[prefix-caching|Prefix caching / KV reuse]]**: requests that share a prefix (system prompt, few-shot examples, previous chat turns, RAG templates) reuse its cached KV blocks and skip that prefill — big TTFT wins for chat and agents.',
          '**Disaggregated serving**: run prefill and decode on separate GPU pools and ship the KV cache between them, so each phase is batched and scaled for its own bottleneck (NVIDIA Dynamo orchestrates this).',
        ]},

        { t: 'h2', x: 'Speculative decoding' },
        { t: 'p', x: 'Decode leaves compute idle, so spend it on guessing. In [[speculative-decoding|speculative decoding]] a cheap drafter proposes γ tokens; the target model scores all γ (plus one) positions in **one** forward pass — about the cost of one decode step, since decode is memory-bound — and accepts the longest prefix consistent with its own distribution, plus one corrected token. With rejection sampling, the output distribution is exactly the target model\'s: it is lossless.' },
        { t: 'specs', rows: [
          ['Expected tokens per target pass', '(1 − α^(γ+1)) / (1 − α), with per-token acceptance rate α and draft length γ'],
          ['Example', 'α = 0.8, γ = 4 → (1 − 0.8⁵)/0.2 = (1 − 0.328)/0.2 ≈ **3.36 tokens per pass**'],
          ['Net speedup', '≈ 3.36 / (1 + γ·c), where c = draft cost per token relative to target: c = 0.05 → 3.36/1.2 ≈ **2.8×**'],
          ['Draft sources', 'Small model of the same family (shared tokenizer) · [[eagle-medusa|Medusa]] extra heads with tree attention · [[eagle-medusa|EAGLE]] feature-level draft head (higher α) · n-gram / prompt lookup (great for editing and RAG, where output copies input)'],
          ['When it fades', 'Large batches — decode becomes compute-bound, verification work is no longer free; low acceptance on creative/high-temperature sampling'],
        ]},

        { t: 'h2', x: 'Kernels, launch overhead and parallelism' },
        { t: 'ul', items: [
          '**FlashAttention** (fused, tiled attention that never materialises the T×T score matrix) speeds up prefill and cuts memory. **FlashDecoding** splits the KV sequence across thread blocks and reduces the partial results, so a single long sequence at small batch still fills the GPU during decode. TensorRT-LLM ships its own fused attention kernels for both phases.',
          '**Kernel fusion** (e.g. bias + activation, residual + norm, dequantize-inside-GEMM) cuts memory round-trips — most of a memory-bound step is moving data.',
          '**[[cuda-graphs|CUDA Graphs]]** capture a decode step\'s hundreds of kernel launches and replay them with one launch; at small batch on small models CPU launch overhead can otherwise dominate.',
          '**Tensor parallelism (TP)** splits each weight matrix across GPUs: per-GPU weight and KV memory drop by the TP degree and each GPU reads fewer bytes per step, lowering latency — at the cost of two all-reduces per layer, so keep TP inside an NVLink domain. A 70B BF16 model (140 GB) needs at least TP2 on 80 GB GPUs, and usually TP4–8 for KV headroom and latency. Pipeline parallelism adds capacity across nodes but not single-request latency.',
        ]},
        { t: 'callout', kind: 'trap', title: 'Throughput and latency pull in opposite directions', x: 'Bigger batches raise tokens/s per GPU but raise each user\'s TPOT; more TP lowers latency but costs communication and GPUs per replica; speculation helps latency at low load but can reduce peak throughput. Always ask: optimise for TTFT, TPOT, or total throughput — under which SLA?' },
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'KV-cache arithmetic: 2 × layers × kv_heads × head_dim × bytes × tokens × batch. Expect GQA vs MHA and FP8 vs BF16 comparisons.',
          'Phase reasoning: prefill → compute-bound → TTFT; decode → memory-bound → TPOT; which technique targets which.',
          'Technique matching: fragmentation → paged attention; idle slots from variable lengths → in-flight batching; long prompts stalling streams → chunked prefill; shared system prompts → prefix caching; low-batch latency → speculative decoding.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Prefill is compute-bound (drives TTFT); decode is memory-bound at ~1 FLOP/byte per sequence (drives TPOT).',
          'KV bytes/token = 2·L·H_kv·d·bytes — 128 KiB for Llama 3.1 8B in BF16; the KV cache bounds concurrency.',
          'Paged KV removes fragmentation; in-flight batching removes idle slots; chunked prefill and prefix caching manage prefill cost.',
          'Speculative decoding is lossless; tokens/pass = (1 − α^(γ+1))/(1 − α); best at low batch.',
          'Fused attention kernels, CUDA Graphs and NVLink-local tensor parallelism cut per-step latency.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'tensorrt-llm',
      title: 'TensorRT-LLM and Triton: building and tuning optimised engines',
      minutes: 25,
      outcomes: [
        'Run a model with the TensorRT-LLM LLM API and configure KV cache, batch and token limits',
        'Describe the checkpoint-conversion and trtllm-build workflow and the meaning of its key limits',
        'Lay out a Triton TensorRT-LLM model repository and set the key config.pbtxt parameters',
        'Benchmark with trtllm-bench and tune knobs against TTFT, TPOT and throughput targets',
        'Compare TensorRT-LLM with vLLM and choose between them',
      ],
      blocks: [
        { t: 'lede', x: '[[tensorrt-llm|TensorRT-LLM]] is NVIDIA\'s open-source library for running LLMs as fast as the GPU allows: fused and quantized kernels, paged [[kv-cache|KV cache]], [[inflight-batching|in-flight batching]], [[speculative-decoding|speculative decoding]] and multi-GPU parallelism, behind a simple Python LLM API. [[triton|Triton Inference Server]] puts it behind a production endpoint, and [[nim|NIM]] packages both. This lesson is about using it and, above all, tuning it.' },

        { t: 'h2', x: 'The workflow at a glance' },
        { t: 'flow', steps: [
          { b: 'HF checkpoint', s: 'BF16 weights from Hugging Face or your fine-tune (merged LoRA)' },
          { b: 'Optimise', s: 'Optional ModelOpt PTQ (FP8, INT4-AWQ, NVFP4) / pruning / spec-decoding heads' },
          { b: 'Load / build', s: 'LLM API loads it directly (PyTorch backend), or convert + trtllm-build a TensorRT engine' },
          { b: 'Benchmark', s: 'trtllm-bench on your real input/output length distribution' },
          { b: 'Serve', s: 'trtllm-serve (OpenAI-compatible), Triton TensorRT-LLM backend, or a NIM' },
        ]},
        { t: 'p', x: 'TensorRT-LLM has two backends. The **PyTorch backend** (the default for the LLM API in current releases) loads Hugging Face-style checkpoints directly and uses TensorRT-LLM\'s optimised kernels and runtime without a separate compile step. The original **TensorRT backend** compiles an engine ahead of time with [[trtllm-build]]. Both share the batch manager (in-flight batching, paged KV, scheduling) — and the same tuning concepts.' },

        { t: 'h2', x: 'The LLM API' },
        { t: 'code', lang: 'python', title: 'trtllm_quickstart.py — pip install tensorrt-llm (or use the NGC container)', x: `from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",       # HF id, local HF dir, or a ModelOpt-quantized export
    tensor_parallel_size=1,                   # >1 shards weights + KV across GPUs (keep within NVLink)
    max_batch_size=64,                        # max sequences scheduled concurrently
    max_num_tokens=8192,                      # per-iteration token budget (prefill chunks + decode tokens)
    max_seq_len=4096,                         # input + output limit per request
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.85,        # share of free memory (after weights) given to the KV pool
        enable_block_reuse=True,              # prefix caching: reuse KV blocks of shared prefixes
    ),
)

params = SamplingParams(max_tokens=128, temperature=0.7, top_p=0.9)
prompts = ["Explain paged attention in two sentences.", "What does max_num_tokens control?"]
for out in llm.generate(prompts, params):    # requests are in-flight batched internally
    print(out.prompt, "->", out.outputs[0].text)` },
        { t: 'code', lang: 'bash', title: 'Serve and query an OpenAI-compatible endpoint', x: `trtllm-serve Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000

curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "messages": [{"role": "user", "content": "Give me one tip for faster LLM inference."}],
  "max_tokens": 64
}'` },

        { t: 'h2', x: 'The engine-build workflow (TensorRT backend)' },
        { t: 'p', x: 'You will still meet the explicit build path in Triton deployments, older docs and the exam. It has three steps: convert (or quantize) the HF weights into a TensorRT-LLM checkpoint with the target TP/PP layout and dtype, compile with `trtllm-build`, then run.' },
        { t: 'code', lang: 'bash', title: 'Convert / quantize → build (run from the TensorRT-LLM examples)', x: `# 1a) plain BF16 checkpoint, split for 2-way tensor parallelism
#     (each model family has a convert_checkpoint.py in its examples folder)
python convert_checkpoint.py --model_dir ./Llama-3.1-8B-Instruct \\
    --output_dir ./ckpt_bf16_tp2 --dtype bfloat16 --tp_size 2

# 1b) or: quantize with ModelOpt into a TensorRT-LLM checkpoint (FP8 weights/activations + FP8 KV cache)
python examples/quantization/quantize.py --model_dir ./Llama-3.1-8B-Instruct \\
    --qformat fp8 --kv_cache_dtype fp8 --calib_size 512 --output_dir ./ckpt_fp8

# 2) compile the engine — the limits below are baked in
trtllm-build --checkpoint_dir ./ckpt_fp8 --output_dir ./engine_fp8 \\
    --gemm_plugin auto \\
    --max_batch_size 64 \\
    --max_input_len 4096 \\
    --max_seq_len 8192 \\
    --max_num_tokens 8192 \\
    --use_paged_context_fmha enable      # needed for chunked context and KV-cache reuse` },
        { t: 'table', head: ['Build option', 'Meaning', 'Trade-off'], rows: [
          ['`--max_batch_size`', 'Max concurrent sequences the runtime can schedule', 'Higher → more throughput potential, more activation/workspace memory, higher TPOT under load'],
          ['`--max_num_tokens`', 'Max tokens (all prefill + decode tokens of the batch) processed in one iteration', 'Higher → faster prefill of long prompts, more activation memory, bigger TPOT spikes; lower → smoother latency with chunked context'],
          ['`--max_input_len` / `--max_seq_len`', 'Longest prompt / longest prompt + output per request', 'Requests beyond the limit are rejected — size to your traffic, rebuild to change'],
          ['`--gemm_plugin`, attention plugins', 'Use TensorRT-LLM\'s hand-tuned kernels for GEMMs and attention', '`auto` picks the dtype; paged context FMHA enables chunked prefill and KV reuse'],
          ['Quantization (from the checkpoint)', 'FP8, INT8 SmoothQuant, INT4 AWQ, W4A8, NVFP4; plus KV-cache dtype', 'See the quantization lesson: memory vs accuracy vs which phase speeds up'],
          ['TP / PP size (from the checkpoint)', 'How weights are split across GPUs', 'TP lowers latency (NVLink needed); PP adds capacity across nodes'],
        ]},
        { t: 'callout', kind: 'warn', title: 'Engines are not portable', x: 'A TensorRT engine is compiled for one GPU architecture (an H100 engine will not run on an L40S), one TP/PP layout and one set of max limits, and is tied to the TensorRT-LLM version that built it. Rebuild for each target; keep the checkpoint and build command in version control rather than the engine binary.' },

        { t: 'h2', x: 'Serving with Triton: the TensorRT-LLM backend' },
        { t: 'p', x: 'Triton\'s TensorRT-LLM backend serves engines with the same in-flight batch manager. The reference model repository (`all_models/inflight_batcher_llm` in the tensorrtllm_backend repo) is an **ensemble** of tokenizer, engine and detokenizer models:' },
        { t: 'code', lang: 'text', title: 'Model repository layout', x: `triton_model_repo/
├── ensemble/            # config.pbtxt only: wires preprocessing -> tensorrt_llm -> postprocessing
│   ├── 1/
│   └── config.pbtxt
├── preprocessing/       # Python backend: tokenizes text_input into input_ids
│   ├── 1/model.py
│   └── config.pbtxt
├── tensorrt_llm/        # the TensorRT-LLM backend: runs the engine with in-flight batching
│   ├── 1/
│   └── config.pbtxt
├── postprocessing/      # Python backend: detokenizes output_ids
│   ├── 1/model.py
│   └── config.pbtxt
└── tensorrt_llm_bls/    # optional Business Logic Scripting model (e.g. for speculative decoding flows)` },
        { t: 'code', lang: 'text', title: 'tensorrt_llm/config.pbtxt — the parameters that matter', x: `backend: "tensorrtllm"
max_batch_size: 64                                   # Triton-side cap; align with the engine
model_transaction_policy { decoupled: true }         # stream tokens back as they are generated

parameters { key: "gpt_model_path"                 value: { string_value: "/engines/llama8b_fp8" } }
parameters { key: "batching_strategy"              value: { string_value: "inflight_fused_batching" } }
parameters { key: "batch_scheduler_policy"         value: { string_value: "max_utilization" } }  # or guaranteed_no_evict
parameters { key: "kv_cache_free_gpu_mem_fraction" value: { string_value: "0.9" } }
parameters { key: "enable_kv_cache_reuse"          value: { string_value: "true" } }   # prefix caching
parameters { key: "enable_chunked_context"         value: { string_value: "true" } }   # chunked prefill
parameters { key: "exclude_input_in_output"        value: { string_value: "true" } }` },
        { t: 'code', lang: 'bash', title: 'Fill the templates, launch, query', x: `# the repo ships config.pbtxt templates; fill_template.py substitutes the placeholders
python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt \\
  triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:True,engine_dir:/engines/llama8b_fp8,\\
batching_strategy:inflight_fused_batching,kv_cache_free_gpu_mem_fraction:0.9,enable_kv_cache_reuse:True,\\
batch_scheduler_policy:max_utilization,enable_chunked_context:True,max_queue_delay_microseconds:0,\\
exclude_input_in_output:True
# (repeat for preprocessing/postprocessing with tokenizer_dir, and for ensemble)

python3 scripts/launch_triton_server.py --world_size 1 --model_repo triton_model_repo

curl -s -X POST localhost:8000/v2/models/ensemble/generate \\
  -d '{"text_input": "What is in-flight batching?", "max_tokens": 64}'` },
        { t: 'callout', kind: 'tip', title: 'Scheduler policies', x: [
          '`guaranteed_no_evict`: admit a request only if KV blocks for its full max length can be guaranteed — no request is ever paused, but fewer run concurrently.',
          '`max_utilization`: admit aggressively and pause/evict requests if the KV pool runs out — higher throughput, occasional latency hiccups. Choose by SLA.',
        ]},

        { t: 'h2', x: 'Benchmark, then tune' },
        { t: 'p', x: 'Tune on a dataset that matches production input/output lengths — a 128-in/128-out benchmark tells you little about a [[rag|RAG]] workload with 3k-token prompts. [[trtllm-bench]] runs offline throughput and latency benchmarks directly on TensorRT-LLM; genai-perf (Module 8) measures a running server end to end.' },
        { t: 'code', lang: 'bash', title: 'trtllm-bench', x: `# dataset: JSONL of requests with your real input/output lengths
# (the repo provides a prepare_dataset script to synthesise token-length distributions)
trtllm-bench --model meta-llama/Llama-3.1-8B-Instruct throughput --dataset ./requests.jsonl
trtllm-bench --model meta-llama/Llama-3.1-8B-Instruct latency    --dataset ./requests.jsonl
# compare: output tokens/s, requests/s, TTFT and per-token latency percentiles across configurations` },
        { t: 'table', head: ['Symptom', 'Knob to turn', 'Cost'], rows: [
          ['Throughput low, GPU memory has headroom', '↑ max_batch_size, ↑ KV free-memory fraction, FP8 KV cache', 'Higher TPOT per user'],
          ['Requests queue, KV pool full', 'FP8/INT8 KV cache, quantize weights (frees KV room), add TP or replicas', 'Accuracy check needed; more GPUs'],
          ['TPOT spikes when long prompts arrive', 'Enable chunked context, lower max_num_tokens', 'Slightly higher TTFT for long prompts'],
          ['High TTFT with a shared system prompt', 'Enable KV-cache block reuse (prefix caching)', 'Pool memory held by cached blocks'],
          ['Single-user latency too high', 'More TP inside NVLink, speculative decoding (EAGLE/Medusa/draft), INT4/FP8 weights', 'More GPUs per replica; spec gains fade at high load'],
          ['Prefill-heavy (RAG, summarisation)', 'FP8 W8A8 (compute-bound gains), larger max_num_tokens', 'Activation memory'],
          ['Small model, tiny batch, CPU-bound', 'CUDA graphs, fewer Python hops (LLM API / C++ runtime)', 'Graph capture memory and warm-up'],
        ]},

        { t: 'h2', x: 'TensorRT-LLM vs vLLM' },
        { t: 'vs', cards: [
          { h: 'TensorRT-LLM', items: [
            'NVIDIA-optimised kernels; earliest support for new NVIDIA formats (FP8, NVFP4) and GPUs',
            'Tight integration with ModelOpt, Triton, NIM and Dynamo',
            'Typically the highest throughput / lowest latency on NVIDIA GPUs, especially with FP8/FP4',
            'More configuration surface; TensorRT backend needs per-GPU engine builds',
          ]},
          { h: 'vLLM', items: [
            'Open-source, PyTorch-based; originated paged attention',
            'Very broad and fast Hugging Face model coverage; simple `vllm serve`',
            'Runs on NVIDIA and other accelerators',
            'Competitive performance; also loads ModelOpt-quantized checkpoints',
          ]},
        ]},
        { t: 'p', x: 'Both implement the same core ideas — paged KV, continuous batching, chunked prefill, prefix caching, speculative decoding, [[tensor-parallelism|tensor parallelism]], FP8. Choose TensorRT-LLM (or a NIM, which typically wraps it) when you need peak performance on NVIDIA hardware and the NVIDIA serving stack; choose [[vllm|vLLM]] for fastest access to new architectures and portability. Benchmark both on your workload.' },
        { t: 'callout', kind: 'trap', title: 'Mismatched limits are a common outage', x: 'If Triton\'s `max_batch_size`, the engine\'s build-time limits and the client\'s `max_tokens` disagree, requests are rejected or silently capped. A request with a 6k-token prompt against an engine built with `--max_input_len 4096` fails no matter how much memory is free.' },
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Knob semantics: max_batch_size (concurrency), max_num_tokens (per-iteration token budget), max_input_len / max_seq_len (request limits), kv_cache_free_gpu_mem_fraction (KV pool size).',
          'Triton backend: `batching_strategy: inflight_fused_batching`, `decoupled` for streaming, `enable_kv_cache_reuse`, scheduler policy guaranteed_no_evict vs max_utilization, ensemble = preprocessing → tensorrt_llm → postprocessing.',
          'Workflow order: (quantize) → convert → trtllm-build → serve; engines are GPU-specific; benchmark on realistic lengths.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'The LLM API (`from tensorrt_llm import LLM`) is the simplest entry point; `trtllm-serve` exposes an OpenAI-compatible server.',
          'The TensorRT backend compiles GPU-specific engines whose max batch, token and length limits are fixed at build time.',
          'Triton\'s TensorRT-LLM backend uses an ensemble repo; key parameters are inflight_fused_batching, KV fraction, KV reuse, chunked context, scheduler policy and decoupled streaming.',
          'Tune with trtllm-bench on real length distributions, trading TTFT, TPOT and throughput explicitly.',
          'TensorRT-LLM and vLLM share the same core techniques; TensorRT-LLM leads on NVIDIA-specific performance and integration.',
        ]},
      ],
    },
  ],

  quiz: [
    { id: 'o01', domain: 'optimization',
      q: 'Why is FP8 E4M3, rather than E5M2, used for weights and activations in FP8 inference?',
      opts: ['E4M3 has a larger dynamic range, needed for activation outliers', 'E4M3 has one more mantissa bit, giving more precision; forward tensors need precision more than the extra range of E5M2', 'E5M2 is not supported by Hopper Tensor Cores', 'E4M3 is an integer format, so it is faster on Tensor Cores'],
      a: 1,
      why: 'E4M3 (max ±448) trades range for an extra mantissa bit. Weights and activations with per-tensor scales fit that range and benefit from precision; E5M2 (max ±57,344) is used for gradients, which need range. See [[fp8]].',
      distract: 'E5M2 has the larger range, not E4M3. Hopper supports both FP8 formats. Both are floating-point formats.' },
    { id: 'o02', domain: 'optimization',
      scenario: 'You quantize a 13B model to INT8 W8A8 with per-tensor activation scales and accuracy drops sharply. Weight-only INT8 of the same model is fine.',
      q: 'What is the most likely cause and fix?',
      opts: ['The weights have too few bits; switch to INT4 AWQ', 'The KV cache overflowed; increase kv_cache_free_gpu_mem_fraction', 'The calibration set was too large; use fewer samples', 'Activation outlier channels dominate the per-tensor scale; apply SmoothQuant or switch to FP8'],
      a: 3,
      why: 'Weight-only works, so activations are the problem. A few outlier channels stretch the per-tensor scale. [[smoothquant|SmoothQuant]] migrates those outliers into the weights; FP8\'s logarithmic spacing also handles them better.',
      distract: 'INT4 weights are a harder problem than INT8, not a fix. KV-pool size affects capacity, not accuracy. More calibration data does not cause outlier damage.' },
    { id: 'o03', domain: 'optimization',
      q: 'An INT4 weight-only scheme uses group size 128 with one FP16 scale per group and no zero point. What is the effective storage per weight?',
      opts: ['4.125 bits', '4.0 bits', '4.5 bits', '5.0 bits'],
      a: 0,
      why: 'Each group of 128 weights adds 16 bits of scale: 16/128 = 0.125 bits per weight, so 4 + 0.125 = 4.125 bits. See [[quant-granularity|quantization granularity]].',
      distract: '4.0 ignores the scales. 4.5 is the NVFP4 figure (FP8 scale per 16 values: 8/16 = 0.5). 5.0 would correspond to a 16-bit scale per 16 weights.' },
    { id: 'o04', domain: 'optimization',
      q: 'Which statement about 2:4 structured sparsity is correct?',
      opts: ['It removes 2 of every 4 transformer layers', 'It gives a 2× speedup on any GPU because half the weights are zero', 'In every group of 4 consecutive weights at most 2 are non-zero, and Ampere+ Sparse Tensor Cores can skip the zeros for up to 2× GEMM throughput', 'It requires converting the model to INT4 first'],
      a: 2,
      why: '[[structured-sparsity|2:4 sparsity]] is a fine-grained pattern that Sparse Tensor Cores (Ampere and later) accelerate, storing half the values plus small indices.',
      distract: 'Removing layers is depth pruning. Speedups need Sparse Tensor Core hardware and apply only to the sparse GEMMs. 2:4 works with FP16/BF16, INT8 and FP8.' },
    { id: 'o05', domain: 'optimization',
      q: 'In a distillation loss α·CE + (1 − α)·τ²·KL(p_teacher^τ ‖ p_student^τ), what is the role of τ > 1?',
      opts: ['It speeds up training by reducing the number of tokens', 'It sharpens the teacher distribution so only the top token matters', 'It is the learning rate for the KL term', 'It softens both distributions so the teacher\'s small probabilities (dark knowledge) carry useful gradient signal'],
      a: 3,
      why: 'Dividing logits by τ > 1 flattens the softmax, exposing the relative probabilities of non-top tokens. The τ² factor rescales gradients to stay comparable with CE. See [[distillation]].',
      distract: 'Temperature does not change token counts. Sharpening corresponds to τ < 1. τ is not a learning rate.' },
    { id: 'o06', domain: 'optimization',
      q: 'NVIDIA\'s Minitron approach to building smaller models from a large one consists of:',
      opts: ['Structured pruning of width (heads, neurons, channels) or depth (layers) by importance, then knowledge distillation from the original model', 'Unstructured magnitude pruning to 90% sparsity with no retraining', 'Training a small model from scratch on the large model\'s pretraining data', 'Quantizing the large model to INT4 and renaming it'],
      a: 0,
      why: '[[minitron|Minitron]] prunes structurally using activation-based importance and recovers accuracy with distillation, needing far fewer tokens than training from scratch.',
      distract: 'Unstructured 90% sparsity without retraining destroys accuracy and gives no dense speedup. Training from scratch is what Minitron avoids. Quantization is a separate, complementary technique.' },
    { id: 'o07', domain: 'optimization',
      q: 'For Llama 3.1 8B (32 layers, 8 KV heads, head dim 128) with an FP8 KV cache, how much KV cache does one 32k-token sequence need?',
      opts: ['1 GiB', '8 GiB', '4 GiB', '2 GiB'],
      a: 3,
      why: 'Per token: 2 (K and V) × 32 layers × 8 KV heads × 128 dims × 1 byte = 65,536 B = 64 KiB. × 32,768 tokens = 2 GiB. The same sequence in BF16 would need 4 GiB. See [[kv-cache]].',
      distract: '1 GiB is an 8k-token BF16 sequence (or 16k in FP8). 4 GiB is the BF16 figure for 32k tokens. 8 GiB assumes 32 KV heads (no GQA).' },
    { id: 'o08', domain: 'optimization',
      q: 'Why is the decode phase of LLM inference memory-bandwidth-bound at small batch sizes?',
      opts: ['Because attention is quadratic in sequence length', 'Because each step reads every weight (and the KV cache) from HBM to perform only about one multiply-add per weight per sequence', 'Because decode uses FP32 while prefill uses BF16', 'Because the tokenizer runs on the CPU'],
      a: 1,
      why: 'At batch 1 the arithmetic intensity is ~1 FLOP per byte, far below the GPU\'s ridge point (~300 FLOPs/byte on H100). See [[arithmetic-intensity|arithmetic intensity]] and [[decode]].',
      distract: 'Quadratic attention cost is mainly a prefill/long-context compute issue. Both phases use the same precision. Tokenization is negligible compared with the GPU step.' },
    { id: 'o09', domain: 'optimization',
      q: 'With speculative decoding, the per-token acceptance rate is 0.8 and the draft length is 4. About how many tokens does each target-model forward pass produce on average?',
      opts: ['3.4', '4.0', '5.0', '0.8'],
      a: 0,
      why: 'Expected tokens = (1 − α^(γ+1))/(1 − α) = (1 − 0.8⁵)/0.2 = (1 − 0.328)/0.2 ≈ 3.36. See [[speculative-decoding]].',
      distract: '4.0 assumes every draft token is always accepted (and forgets the bonus token). 5.0 is the maximum possible (γ + 1), reached only with α = 1. 0.8 confuses acceptance rate with output.' },
    { id: 'o10', domain: 'optimization',
      scenario: 'A chat service shows smooth token streaming, but whenever a user pastes a 30k-token document every other user\'s stream freezes for over a second.',
      q: 'Which feature addresses this most directly?',
      opts: ['Prefix caching', 'Larger max_batch_size', 'Chunked prefill (chunked context) with a moderate max_num_tokens', 'Speculative decoding'],
      a: 2,
      why: 'The long prompt\'s prefill monopolises iterations. [[chunked-prefill|Chunked prefill]] splits it into chunks scheduled alongside decode steps, keeping TPOT stable.',
      distract: 'Prefix caching only helps when prefixes repeat. A larger batch does not stop one huge prefill from dominating a step. Speculative decoding speeds up decode, not a blocking prefill.' },
    { id: 'o11', domain: 'optimization',
      q: 'In Triton\'s TensorRT-LLM backend, which setting enables iteration-level scheduling so new requests join while others are still generating?',
      opts: ['dynamic_batching { max_queue_delay_microseconds: 100 }', 'batching_strategy: inflight_fused_batching', 'decoupled: true', 'instance_group { count: 4 }'],
      a: 1,
      why: '[[inflight-batching|In-flight batching]] is enabled by `batching_strategy: inflight_fused_batching` in the tensorrt_llm model\'s config.pbtxt.',
      distract: 'Triton\'s dynamic batcher groups whole requests before execution — request-level, not iteration-level. `decoupled` enables streaming responses. Multiple instances add parallel model copies, not continuous batching.' },
    { id: 'o12', domain: 'optimization',
      q: 'A TensorRT engine for Llama 3.1 8B was built on an H100 with --max_input_len 4096. Which deployment will fail without a rebuild?',
      opts: ['Serving it on the same H100 with more KV-cache memory fraction', 'Serving it with prefix caching enabled at runtime', 'Running more concurrent requests than before, up to max_batch_size', 'Running it on an L40S GPU, or with 8k-token prompts'],
      a: 3,
      why: 'Engines are compiled for a specific GPU architecture and fixed build limits. A different GPU or prompts above max_input_len require a rebuild ([[trtllm-build]]).',
      distract: 'KV fraction is a runtime setting. KV-cache reuse is a runtime option (with paged context FMHA built in). Concurrency up to the built max_batch_size is supported.' },
  ],
}
