// Exam-bank questions for LLM Architecture (6%). Not reused in lesson checks or the module quiz.
export default [
  { id: 'xa01', domain: 'architecture',
    scenario: 'An inference team serves a 32-layer model (head_dim 128, BF16 KV cache) on an 80 GB GPU. With 32 KV heads it fits only a small batch at 8k context. A retrained variant with identical quality targets uses 4 KV heads.',
    q: 'By what factor does the KV-cache memory per sequence change, and what is the per-token cache size of the new variant?',
    opts: [
      'It halves; 256 KiB per token',
      'It drops 8×; 64 KiB per token',
      'It drops 8×; 32 KiB per token',
      'It is unchanged, because KV-cache size depends on query heads',
    ],
    a: 1,
    why: 'Cache scales with KV heads: 32 → 4 is 8× smaller. Per token: 2 × 32 × 4 × 128 × 2 bytes = 65,536 B = 64 KiB (vs 512 KiB with 32 KV heads). This is the core reason for [[gqa|GQA]].',
    distract: 'Halving would be 32 → 16 heads. 32 KiB forgets either the K-and-V factor of 2 or the 2 bytes per value. The cache stores keys and values, so it depends on **KV** heads, not query heads.' },

  { id: 'xa02', domain: 'architecture',
    q: 'A research group has a fixed budget of about 1.2 × 10²¹ training FLOPs and wants the lowest possible loss for that budget, with no concern for inference cost. Which plan follows Chinchilla scaling?',
    opts: [
      'A 20B-parameter model trained on 10B tokens',
      'A 1B-parameter model trained on 200B tokens',
      'A 500M-parameter model trained on 400B tokens',
      'A 3B-parameter model trained on about 65B tokens',
    ],
    a: 3,
    why: 'Compute-optimal means D ≈ 20·N and C ≈ 6·N·D = 120·N². N = √(1.2×10²¹ / 120) ≈ 3.2B, D ≈ 64B tokens. A 3B model on ~65B tokens matches (6 × 3×10⁹ × 6.5×10¹⁰ ≈ 1.17×10²¹).',
    distract: '20B on 10B tokens is badly under-trained (0.5 tokens/param). 1B on 200B tokens and 500M on 400B tokens spend the same compute but over-train small models — sensible for inference-heavy use, not for minimum loss at fixed compute.' },

  { id: 'xa03', domain: 'architecture',
    q: 'Which set of choices describes the block design of current open LLMs such as Llama 3 and Qwen2.5?',
    opts: [
      'Pre-norm RMSNorm, RoPE on queries and keys, SwiGLU MLP, grouped-query attention',
      'Post-norm LayerNorm, learned absolute positions, GELU MLP, multi-head attention',
      'Pre-norm LayerNorm, sinusoidal positions added at each layer, ReLU MLP, multi-query attention',
      'Bidirectional attention, RoPE, SwiGLU MLP, cross-attention to an encoder',
    ],
    a: 0,
    why: 'The modern decoder recipe is pre-[[rmsnorm|RMSNorm]] + [[rope|RoPE]] + [[swiglu|SwiGLU]] + [[gqa|GQA]], visible in their configs (`rms_norm_eps`, `rope_theta`, `hidden_act: silu`, `num_key_value_heads`).',
    distract: 'Post-norm + learned positions + GELU + MHA is the 2017–2019 (BERT/GPT-2 era) design. The third option mixes outdated components. Bidirectional attention and cross-attention describe encoders and encoder-decoders, not these decoder-only models.' },

  { id: 'xa04', domain: 'architecture',
    scenario: 'A company compares two models for an on-premises chatbot: a dense 14B model and an MoE model with 30B total and 3B active parameters. Both reach similar benchmark quality. GPU memory is the binding constraint; per-token compute is not.',
    q: 'Which statement is correct?',
    opts: [
      'The MoE model needs less GPU memory because only 3B parameters are active per token',
      'Both need the same memory because quality is similar',
      'The MoE model needs more memory for weights (≈60 GB vs ≈28 GB in BF16), even though it uses far less compute per token',
      'The dense model needs more memory because dense models cannot share weights',
    ],
    a: 2,
    why: 'In an [[moe|MoE]] every expert must be resident, so weight memory follows **total** parameters: 30B × 2 B ≈ 60 GB vs 14B × 2 B ≈ 28 GB. MoE wins on FLOPs per token, not on memory.',
    distract: 'Sizing memory by active parameters is the classic MoE mistake. Similar quality says nothing about memory. Dense models are smaller in total here, so they need less memory.' },
]
