/**
 * Module 1 glossary — LLM Architecture.
 * Shared terms (transformer, self-attention, embedding, decoder-only, kv-cache …) live in core.js.
 */
export default [
  { id: 'multi-head-attention', term: 'Multi-head attention (MHA)', cat: 'Architecture',
    def: 'Attention run in H parallel heads: the hidden size C is split into H slices of d_head = C/H, each head computes its own softmax(QKᵀ/√d_head)V, and the results are concatenated and mixed by an output projection. Lets different heads track different relationships at the same total cost.',
    exam: 'Know the reshape (B, T, C) → (B, H, T, d_head), that attention has ~4·C² parameters per layer, and that in MHA every query head has its own K/V head — the baseline that GQA and MQA shrink.' },

  { id: 'causal-mask', term: 'Causal mask', cat: 'Architecture',
    def: 'A triangular mask that sets attention scores for future positions to −∞ before the softmax, so token t attends only to tokens ≤ t. It lets a decoder train on all T next-token predictions of a sequence in one forward pass without seeing the answers.',
    exam: 'The defining difference between decoder (causal) and encoder (bidirectional) attention.' },

  { id: 'positional-encoding', term: 'Positional encoding', cat: 'Architecture',
    def: 'Information about token order added to a transformer, which is otherwise permutation-invariant. Variants: fixed sinusoidal vectors (original transformer), learned absolute embeddings (GPT-2, BERT), relative schemes such as RoPE and ALiBi (modern LLMs).',
    exam: 'Absolute learned positions cannot extrapolate past the trained length; relative schemes (RoPE with scaling) are what enable long-context extension.' },

  { id: 'rope', term: 'RoPE (rotary position embedding)', cat: 'Architecture',
    def: 'Encodes position by rotating each pair of dimensions of the query and key vectors by an angle proportional to the position, with frequencies set by a base `rope_theta`. The q·k dot product then depends on relative distance. Used by Llama, Qwen, Mistral and most current LLMs.',
    exam: 'Context-extension methods (position interpolation, NTK-aware scaling, YaRN, larger rope_theta) all work by rescaling RoPE frequencies; look for `rope_scaling` in a config.' },

  { id: 'layernorm', term: 'LayerNorm', cat: 'Architecture',
    def: 'Normalises each token’s activation vector to zero mean and unit variance, then applies a learned gain and bias. Used in GPT-2 and BERT; largely replaced by RMSNorm in modern LLMs.',
    exam: 'Pre-norm (norm before each sub-layer) is the stable modern placement; post-norm was the 2017 original.' },

  { id: 'rmsnorm', term: 'RMSNorm', cat: 'Architecture',
    def: 'A simplified LayerNorm: x / sqrt(mean(x²) + ε) · g. No mean subtraction and no bias, so it is cheaper with similar quality. Standard in Llama-family models, in pre-norm position.',
    exam: 'Appears as `rms_norm_eps` in configs. Cheaper than LayerNorm — a small but real throughput gain.' },

  { id: 'residual', term: 'Residual connection', cat: 'Architecture',
    def: 'Adding a sub-layer’s input to its output: x = x + f(x). In transformers this forms the "residual stream" that every block reads from and writes to, and gives gradients a direct path through deep stacks.',
    exam: 'Residuals plus pre-norm are why transformers with 100+ layers train stably.' },

  { id: 'ffn', term: 'Feed-forward network (MLP)', cat: 'Architecture',
    def: 'The per-position two-layer (or gated three-matrix) MLP in each transformer block, typically expanding the hidden size ~4× then projecting back. Holds about two-thirds of a dense model’s non-embedding parameters.',
    exam: 'MLP ≈ 8·d² params per layer (4× expansion) vs 4·d² for attention; MoE replaces this sub-layer with multiple experts.' },

  { id: 'swiglu', term: 'SwiGLU', cat: 'Architecture',
    def: 'A gated MLP: down(SiLU(gate(x)) ⊙ up(x)). Three weight matrices instead of two, so the hidden size is reduced to ≈ 8/3·d (often rounded up) to keep parameters similar. Used by Llama, Qwen, Mistral and others.',
    exam: 'Explains `intermediate_size` values like 14,336 for d = 4096 and the gate_proj/up_proj/down_proj weight names LoRA often targets.' },

  { id: 'gqa', term: 'GQA (grouped-query attention)', cat: 'Architecture',
    def: 'Attention where the H query heads are split into G groups and each group shares one key/value head. Reduces KV-cache size and memory bandwidth by H/G with quality close to MHA. Configured by `num_key_value_heads` (HF) or `num_query_groups` (NeMo/Megatron).',
    exam: 'KV bytes/token = 2 × layers × kv_heads × head_dim × bytes. Llama-3-8B: 32 query / 8 KV heads → 128 KiB/token in BF16, 4× less than MHA.' },

  { id: 'mqa', term: 'MQA (multi-query attention)', cat: 'Architecture',
    def: 'The extreme case of GQA: all query heads share a single key/value head. Smallest KV cache (1/H of MHA) but some quality loss, so most recent models choose GQA instead.',
    exam: 'If a question asks for the maximum KV-cache reduction regardless of quality, the answer is MQA; for the best trade-off, GQA.' },

  { id: 'moe', term: 'Mixture of experts (MoE)', cat: 'Architecture',
    def: 'A layer with many expert MLPs and a learned router that sends each token to its top-k experts. Total parameters (capacity, memory) are much larger than active parameters per token (compute). Examples: Mixtral 8×7B (≈47B total, ≈13B active), DeepSeek-V3 (671B total, 37B active).',
    exam: 'Memory ∝ total params, FLOPs ∝ active params. Needs load balancing and expert parallelism (all-to-all communication).' },

  { id: 'router', term: 'Router (gating network)', cat: 'Architecture',
    def: 'The small linear layer in an MoE block that scores each token against every expert, selects the top-k, and weights their outputs. Trained with an auxiliary load-balancing loss (or a balancing bias) so tokens spread across experts.',
    exam: 'A router collapse — most tokens going to a few experts — wastes capacity and overloads the GPUs holding those experts.' },

  { id: 'active-parameters', term: 'Active vs total parameters', cat: 'Architecture',
    def: 'In an MoE model, total parameters count every expert; active parameters count only what one token actually uses (shared layers + its top-k experts). Dense models have active = total.',
    exam: 'Size GPU memory by total, FLOPs and per-token latency by active.' },

  { id: 'scaling-laws', term: 'Scaling laws', cat: 'Architecture',
    def: 'Empirical power laws relating loss to parameters N, training tokens D and compute C. Training compute ≈ 6·N·D FLOPs (2ND forward + 4ND backward). Used to choose model size and data budget before an expensive run.',
    exam: 'Remember C ≈ 6ND and be able to rearrange it for time-to-train given GPU FLOPS and utilisation.' },

  { id: 'chinchilla', term: 'Chinchilla scaling', cat: 'Architecture',
    def: 'The 2022 DeepMind result (Hoffmann et al.) that compute-optimal training scales parameters and tokens together — about 20 tokens per parameter (Chinchilla: 70B params, 1.4T tokens). Modern small models are deliberately trained far beyond this because inference cost depends on size.',
    exam: 'Fixed training budget → ~20 tokens/param. Inference-heavy deployment → smaller model, more tokens ("over-training").' },

  { id: 'weight-tying', term: 'Weight tying', cat: 'Architecture',
    def: 'Using the same V×d matrix for the input token embedding and the output LM head. Saves V·d parameters; common in small models (GPT-2, Qwen2.5-0.5B, SmolLM2), usually off in large ones. Shown as `tie_word_embeddings` in configs.',
    exam: 'In small models the embedding table can be a quarter or more of all parameters, so tying matters for size estimates.' },

  { id: 'encoder-only', term: 'Encoder-only model', cat: 'Architecture',
    def: 'A transformer with bidirectional attention (no causal mask), trained with objectives such as masked-language modelling. BERT, RoBERTa and DeBERTa are examples. Produces contextual representations for classification, NER, embeddings and reranking; cannot generate autoregressively.',
    exam: 'Pick it for classification/embedding workloads where generation is not needed and latency/cost matter.' },

  { id: 'encoder-decoder', term: 'Encoder-decoder model', cat: 'Architecture',
    def: 'A transformer with a bidirectional encoder over the input and a causal decoder that generates the output while attending to the encoder via cross-attention. T5, BART and Whisper are examples. Natural for translation, summarisation and speech recognition.',
    exam: 'Contrast with decoder-only: separate input and output stacks, cross-attention, and a fit for strict input→output transformation tasks.' },

  { id: 'pre-norm', term: 'Pre-norm', cat: 'Architecture',
    def: 'Placing the normalisation before each sub-layer inside the residual branch: x = x + f(Norm(x)). Keeps the residual stream an identity path and trains stably at depth; requires a final norm before the LM head.',
    exam: 'The default in GPT-2 onwards and all current LLMs; post-norm needs careful warmup and is less stable when deep.' },

  { id: 'sdpa', term: 'Scaled dot-product attention (SDPA)', cat: 'Architecture',
    def: 'The core attention operation softmax(QKᵀ/√d_k + mask)·V. In PyTorch, `torch.nn.functional.scaled_dot_product_attention` dispatches it to fused, memory-efficient kernels (FlashAttention-style on supported NVIDIA GPUs).',
    exam: 'The √d_k factor keeps score variance near 1 so softmax does not saturate.' },
]
