// End-of-lesson comprehension checks for Module 1: LLM Architecture. Keyed by lesson id.
export default {
  'pytorch-foundations': [
    {
      q: 'In a language-model forward pass, the token ids have shape (B, T). What shape are the logits?',
      opts: [
        '(B, T, V), one score per vocabulary entry at every position',
        '(B, V), one distribution per sequence',
        '(B, T, C), the hidden size at every position',
        '(T, V), shared across the batch',
      ],
      a: 0,
      why: 'The LM head maps each position’s C-dimensional vector to V [[logits]], so every one of the B·T positions gets a full next-token distribution.',
      distract: '(B, V) would predict only one token per sequence. (B, T, C) is the hidden state before the LM head. Logits are not shared across the batch — each sequence has its own.',
    },
    {
      q: 'Why does a PyTorch training loop call `optimizer.zero_grad()` every step?',
      opts: [
        'To reset the learning-rate schedule',
        'To free the activations stored during the forward pass',
        'Because `.grad` accumulates across backward calls, so old gradients must be cleared',
        'To re-initialise AdamW’s moment estimates',
      ],
      a: 2,
      why: '[[autograd|Autograd]] adds new gradients to whatever is already in `.grad`. Without clearing, each step would apply the sum of all previous gradients.',
      distract: 'The LR schedule is separate. Activations are freed by `backward()` itself. AdamW’s moments are optimizer state and must persist across steps — resetting them would break the optimizer.',
    },
    {
      q: 'For next-token prediction on a sequence x, how are the targets y built?',
      opts: [
        'y is x reversed',
        'y is x shifted one position to the left, so y[t] = x[t+1]',
        'y is a copy of x; the model learns to reproduce its input',
        'y is a random sample of tokens from the vocabulary',
      ],
      a: 1,
      why: 'Each position predicts the following token, so targets are the inputs offset by one. One sequence of T+1 tokens yields T training examples.',
      distract: 'Reversing or copying would teach a different (trivial) task. Random targets carry no signal.',
    },
    {
      q: 'After a few thousand steps a model’s training loss keeps falling while validation loss has started to rise. What is happening?',
      opts: [
        'The learning rate is too low',
        'The validation set is too easy',
        'The model is underfitting',
        'The model is overfitting — memorising the training data',
      ],
      a: 3,
      why: 'A widening gap with rising validation loss is the textbook sign of [[overfitting]]. Fixes: early stopping, dropout, more data, a smaller model.',
      distract: 'A too-low LR slows both curves. An easy validation set would give **lower** val loss. Underfitting means both losses stay high.',
    },
    {
      q: 'What should you wrap validation in, and why?',
      opts: [
        '`torch.cuda.amp.autocast()`, to make validation numerically exact',
        '`model.train()`, so dropout gives a realistic estimate',
        '`model.eval()` and `torch.no_grad()`, to disable dropout and skip building the autograd graph',
        '`torch.compile()`, because uncompiled models cannot be evaluated',
      ],
      a: 2,
      why: '`eval()` turns off dropout (and other train-only behaviour); `no_grad()` avoids storing activations for a backward pass that will never happen, saving memory and time.',
      distract: 'Autocast reduces precision, it does not make results exact. Dropout on at eval adds noise. Compilation is optional.',
    },
  ],

  attention: [
    {
      q: 'In self-attention, what role does the key vector play?',
      opts: [
        'It is the information a token passes on when attended to',
        'It is what other tokens’ queries are matched against to compute attention scores',
        'It stores the position of the token in the sequence',
        'It is the output of the attention layer after the projection',
      ],
      a: 1,
      why: 'Scores are q·k: each query is compared with every visible key. The **value** is what gets mixed into the output.',
      distract: 'Passing information on is the value’s job. Positions come from positional encodings such as RoPE. The output is the weighted sum of values after the output projection.',
    },
    {
      q: 'What does the causal mask do in a decoder-only transformer?',
      opts: [
        'Hides padding tokens from the loss',
        'Randomly drops attention weights during training',
        'Restricts attention to a fixed window of recent tokens',
        'Sets scores for future positions to −∞ so each token attends only to itself and earlier tokens',
      ],
      a: 3,
      why: 'The [[causal-mask]] prevents a position from seeing the token it must predict, which lets one forward pass train all T positions in parallel.',
      distract: 'Padding is handled by an attention/padding mask and ignored labels. Random dropping is dropout. A fixed window is sliding-window attention.',
    },
    {
      q: 'A model has hidden size 4096 and 32 attention heads. What is the head dimension, and what shape is the attention score tensor for a batch?',
      opts: [
        'd_head = 128; scores are (B, 32, T, T)',
        'd_head = 4096; scores are (B, T, T)',
        'd_head = 32; scores are (B, 128, T, T)',
        'd_head = 128; scores are (B, T, 4096)',
      ],
      a: 0,
      why: '[[multi-head-attention|Multi-head attention]] splits C into H heads: 4096 / 32 = 128. Each head computes its own T×T score matrix, giving (B, H, T, T).',
      distract: 'd_head = 4096 ignores the split. Swapping heads and head_dim gives 32. (B, T, 4096) is the shape of the layer’s output, not its scores.',
    },
    {
      q: 'Which layout do modern LLMs such as Llama and Qwen use inside each block?',
      opts: [
        'x = Norm(x + Attn(x)); x = Norm(x + MLP(x))',
        'x = Attn(MLP(x)) with no residuals',
        'x = x + Attn(Norm(x)); x = x + MLP(Norm(x))',
        'x = Norm(Attn(x)) + Norm(MLP(x))',
      ],
      a: 2,
      why: 'Pre-norm with [[residual]] connections keeps the residual stream an identity path, which makes very deep stacks train stably.',
      distract: 'The first option is the original post-norm layout. Without residuals deep networks are hard to train. The last option runs attention and MLP in parallel without a residual and is not what these models use.',
    },
    {
      q: 'With a standard 4× MLP, roughly what share of a transformer block’s parameters are in the MLP?',
      opts: [
        'About one-third',
        'About two-thirds',
        'About 10%',
        'Exactly half',
      ],
      a: 1,
      why: 'Attention has 4d² parameters (Q, K, V, output); the [[ffn|MLP]] has 2 × d × 4d = 8d². That is 8/12 ≈ two-thirds.',
      distract: 'One-third is the attention share. 10% and half do not match 4d² vs 8d².',
    },
  ],

  'mini-gpt': [
    {
      q: 'What does weight tying do in the mini-GPT?',
      opts: [
        'Shares weights between all transformer blocks',
        'Freezes the embedding layer during training',
        'Ties the query and key projections together',
        'Reuses the token-embedding matrix as the LM-head output projection',
      ],
      a: 3,
      why: '[[weight-tying|Weight tying]] sets `lm_head.weight = tok_emb.weight`, saving V·C parameters — significant in small models with large vocabularies.',
      distract: 'Sharing across blocks is a different technique (used in ALBERT). Tying does not freeze anything. Q and K stay separate projections.',
    },
    {
      q: 'Using the 12·L·d² rule, about how many non-embedding parameters does a GPT with 12 layers and d = 768 have?',
      opts: [
        'About 85M',
        'About 7M',
        'About 1.1B',
        'About 340M',
      ],
      a: 0,
      why: '12 × 12 × 768² = 144 × 589,824 ≈ 85M. Add ~39M of embeddings to get GPT-2 small’s 124M.',
      distract: '7M forgets to square d properly; 1.1B and 340M overshoot by large factors.',
    },
    {
      q: 'You sample from your trained model with temperature 0.5 instead of 1.0. What changes?',
      opts: [
        'Generation becomes twice as fast',
        'The model forgets half its context',
        'The distribution is sharpened, so output is more conservative and repetitive',
        'The distribution is flattened, so output is more random',
      ],
      a: 2,
      why: 'Logits are divided by T before softmax. T < 1 enlarges the gaps between logits, concentrating probability on the most likely tokens.',
      distract: 'Temperature does not change speed or context. Flattening happens with T > 1.',
    },
    {
      q: 'A character-level model reaches validation loss 1.5 and a BPE-token model reaches 3.0 on the same text. What can you conclude?',
      opts: [
        'The character model is clearly better',
        'Nothing directly — losses per prediction are not comparable across different tokenizers',
        'The BPE model is clearly better because its loss is higher',
        'Both models are overfitting',
      ],
      a: 1,
      why: 'Loss is averaged per predicted unit. A character model makes several predictions per BPE token, so its per-step loss is naturally lower. Compare using a tokenizer-independent measure such as bits per byte.',
      distract: 'Neither raw number proves superiority across tokenizers. Higher loss is never better by itself. Overfitting needs a train/val comparison.',
    },
    {
      q: 'Why does the mini-GPT have a final LayerNorm before the LM head?',
      opts: [
        'With pre-norm blocks the residual stream itself is never normalised, so it is normalised once at the end',
        'To apply the causal mask',
        'To convert logits into probabilities',
        'Because weight tying requires it',
      ],
      a: 0,
      why: 'Pre-norm normalises only the input to each sub-layer; the accumulated residual stream grows unnormalised, so a final [[layernorm]] stabilises the input to the LM head.',
      distract: 'Masking happens inside attention. Softmax, not LayerNorm, produces probabilities. Weight tying works with or without a final norm.',
    },
  ],

  'modern-architectures': [
    {
      q: 'What is the main practical benefit of grouped-query attention over standard multi-head attention?',
      opts: [
        'Higher accuracy on long documents',
        'Fewer parameters in the MLP layers',
        'A much smaller KV cache, allowing larger batches or longer contexts at inference',
        'It removes the need for positional encodings',
      ],
      a: 2,
      why: '[[gqa|GQA]] shares each key/value head among a group of query heads, cutting the [[kv-cache]] by H/G with quality close to MHA.',
      distract: 'GQA trades a little quality, it does not add accuracy. It does not touch the MLP. Positional encodings are still needed.',
    },
    {
      q: 'How does SwiGLU differ from the GPT-2 MLP?',
      opts: [
        'It removes the nonlinearity entirely',
        'It uses a gated design with three projections (gate, up, down) and a SiLU gate',
        'It replaces the MLP with an attention layer',
        'It runs the MLP only on every other token',
      ],
      a: 1,
      why: '[[swiglu|SwiGLU]] computes down(silu(gate(x)) ⊙ up(x)). With three matrices, the hidden size is reduced (≈ 8/3·d) to keep the parameter count similar.',
      distract: 'SwiGLU is still nonlinear (SiLU). It is an MLP, not attention. Skipping tokens describes nothing in SwiGLU.',
    },
    {
      q: 'Under Chinchilla scaling, how many tokens should a 500M-parameter model be trained on to be compute-optimal?',
      opts: [
        'About 500M tokens',
        'About 1T tokens',
        'About 2.5B tokens',
        'About 10B tokens',
      ],
      a: 3,
      why: '[[chinchilla|Chinchilla]]: ≈ 20 tokens per parameter → 20 × 0.5B = 10B tokens.',
      distract: '500M is 1 token/param, 2.5B is 5 tokens/param, and 1T is heavy over-training (2,000 tokens/param).',
    },
    {
      q: 'Which statement about RMSNorm is correct?',
      opts: [
        'It rescales by the root-mean-square of the activations, without subtracting the mean, and has a learned gain',
        'It subtracts the batch mean and divides by the batch standard deviation',
        'It normalises the weights, not the activations',
        'It is applied only once, after the final layer',
      ],
      a: 0,
      why: '[[rmsnorm|RMSNorm]] = x / rms(x) · g. Dropping the mean and bias makes it cheaper than LayerNorm with similar quality; Llama-family models use it in pre-norm position in every block.',
      distract: 'Batch statistics describe BatchNorm. Weight normalisation is a different technique. RMSNorm appears before every sub-layer plus once at the end.',
    },
    {
      q: 'A config.json shows num_attention_heads = 14 and num_key_value_heads = 2. What does this mean?',
      opts: [
        'The model uses MQA',
        'The model has 2 layers with 14 heads each',
        'Only 2 heads are trained; the other 12 are frozen',
        'It uses GQA: 7 query heads share each key/value head',
      ],
      a: 3,
      why: '14 query heads / 2 KV heads = groups of 7. That is [[gqa|GQA]]; MQA would be num_key_value_heads = 1.',
      distract: 'MQA has exactly one KV head. Layer count is num_hidden_layers. No heads are frozen.',
    },
  ],
}
