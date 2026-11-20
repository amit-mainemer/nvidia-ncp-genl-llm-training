export default {
  id: 'architecture',
  n: 1,
  domain: 'architecture',
  weight: 6,
  title: 'LLM Architecture',
  blurb: 'Go from "what is a tensor" to a GPT you wrote and trained yourself, then learn to read any modern model config — RoPE, RMSNorm, SwiGLU, GQA, MoE — and size a model and its training budget with scaling laws.',
  hours: '4–6 h',
  lessons: [
    /* ------------------------------------------------------------------ */
    {
      id: 'pytorch-foundations',
      title: 'Tensors, autograd and your first training loop',
      minutes: 20,
      outcomes: [
        'Create and reshape tensors and reason about (B, T, C) shapes',
        'Explain how autograd turns a forward pass into gradients',
        'Write the zero_grad → backward → step training loop from memory',
        'Train a bigram character model on tiny Shakespeare and read its train/val loss',
      ],
      blocks: [
        { t: 'lede', x: 'Every LLM — from a 1M-parameter toy to a 400B frontier model — is trained by the same five-line loop: get a batch, run the model, compute a [[loss]], backpropagate, update the weights. This lesson teaches the PyTorch you need to write that loop, and ends with a working (if not very smart) language model that you train in under a minute on a laptop.' },

        { t: 'h2', x: 'Tensors: arrays with a shape, a dtype and a device' },
        { t: 'p', x: 'A [[tensor]] is PyTorch’s n-dimensional array. Three attributes matter all the time: `shape` (the size of each dimension), `dtype` (fp32, [[bf16]], int64…) and `device` (CPU, `cuda`, or `mps` on Apple Silicon). Most bugs you will hit in this course are shape bugs, so get comfortable reading shapes out loud.' },
        { t: 'p', x: 'Language-model activations follow one convention: **(B, T, C)** — batch size, time (sequence position) and channels (the hidden or embedding size). Token ids enter as an integer tensor of shape (B, T); the model turns them into vectors (B, T, C); the final layer produces [[logits]] of shape (B, T, V) where V is the vocabulary size.' },
        { t: 'code', lang: 'python', title: 'Tensor basics you will use constantly', x: `import torch

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.randn(4, 8, 32, device=device)   # (B=4, T=8, C=32) random activations
print(x.shape, x.dtype, x.device)          # torch.Size([4, 8, 32]) torch.float32 ...

# Reshape without copying: split C=32 into 4 heads of 8 dims
h = x.view(4, 8, 4, 8).transpose(1, 2)     # (B, n_head, T, head_dim) = (4, 4, 8, 8)

# Matrix multiply broadcasts over leading dims: (B, T, C) @ (C, V) -> (B, T, V)
W = torch.randn(32, 65, device=device)
logits = x @ W
print(logits.shape)                         # torch.Size([4, 8, 65])

# Integer token ids are int64 ("long"); they index into embedding tables
ids = torch.randint(0, 65, (4, 8))
print(ids.dtype)                            # torch.int64

# Precision: bf16 halves memory per element (2 bytes vs 4)
print(x.element_size(), x.bfloat16().element_size())   # 4 2` },
        { t: 'callout', kind: 'tip', title: 'view vs reshape vs transpose', x: '`view` reinterprets the same memory and requires it to be contiguous; `transpose` swaps two dimensions without moving data, which makes the tensor non-contiguous. After a transpose, use `.contiguous().view(...)` or simply `.reshape(...)`. You will see exactly this pattern in every attention implementation.' },

        { t: 'h2', x: 'Autograd: gradients for free' },
        { t: 'p', x: 'Training needs the derivative of the loss with respect to every weight. You never write those derivatives by hand: [[autograd]] records each operation of the forward pass in a graph, and `loss.backward()` walks that graph in reverse applying the chain rule ([[backprop]]). The result lands in each parameter’s `.grad`.' },
        { t: 'code', lang: 'python', title: 'Autograd on a two-variable function', x: `import torch

x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)

y = w * x ** 2          # forward: y = 2 * 9 = 18; autograd records mul and pow
y.backward()            # reverse pass fills .grad

print(x.grad)           # dy/dx = 2*w*x = 12
print(w.grad)           # dy/dw = x^2    = 9

# Gradients ACCUMULATE: calling backward again adds to .grad
y = w * x ** 2
y.backward()
print(w.grad)           # 18, not 9 -> this is why the loop calls zero_grad()` },
        { t: 'p', x: 'Two consequences matter later. First, autograd must keep the intermediate activations of the forward pass until backward runs — that is where "activation memory" comes from when you scale up. Second, gradients accumulate by default, which is a bug if you forget to clear them and a feature ([[gradient-accumulation|gradient accumulation]]) when you do it on purpose.' },

        { t: 'h2', x: 'The training loop' },
        { t: 'p', x: 'Every PyTorch training loop is the same skeleton. Memorise the order: forward, loss, **zero_grad**, **backward**, **step**. The optimizer ([[adamw]] in practice) owns the update rule; the [[learning-rate]] sets its step size.' },
        { t: 'flow', steps: [
          { b: 'Sample a batch', s: 'Inputs x of shape (B, T) and targets y — for language modelling, y is x shifted one token to the left.' },
          { b: 'Forward', s: '`logits, loss = model(x, y)`: logits (B, T, V), loss = mean [[cross-entropy]] over all B·T positions.' },
          { b: 'zero_grad', s: '`opt.zero_grad(set_to_none=True)` clears last step’s gradients so they do not accumulate.' },
          { b: 'backward', s: '`loss.backward()` fills `.grad` on every parameter via autograd.' },
          { b: 'step', s: '`opt.step()` applies the AdamW update: θ ← θ − lr · (adaptive gradient) − lr · wd · θ.' },
          { b: 'Evaluate sometimes', s: 'Every N steps, measure loss on held-out validation data under `torch.no_grad()` and `model.eval()`.' },
        ]},
        { t: 'callout', kind: 'trap', title: 'The three classic loop bugs', x: [
          '**Forgetting zero_grad** — gradients from previous steps pile up and the effective learning rate explodes.',
          '**Evaluating with autograd on** — validation inside the loop without `torch.no_grad()` wastes memory storing a graph you will never use.',
          '**Leaving dropout on at eval time** — call `model.eval()` before measuring validation loss and `model.train()` afterwards.',
        ]},

        { t: 'h2', x: 'Your first language model: a bigram on tiny Shakespeare' },
        { t: 'p', x: 'A **bigram** model predicts the next character from the current character only. It is just a V×V lookup table: row i holds the logits for "what follows character i". It cannot learn anything beyond pairs of letters, but it exercises the full pipeline — data, batching, loss, loop, sampling — that the GPT in lesson 3 reuses unchanged.' },
        { t: 'p', x: 'The dataset is tiny Shakespeare: about 1.1 million characters with a vocabulary of 65 distinct characters. We use a character-level [[tokenizer]] (each character is a [[token]]) so there is nothing to install. Notice the **train/val split**: the last 10% of the text is never trained on, so its loss tells you whether the model generalises or merely memorises ([[overfitting]]).' },
        { t: 'code', lang: 'python', title: 'bigram.py — runs on CPU in well under a minute', x: `import os, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(url, "input.txt")
text = open("input.txt", encoding="utf-8").read()

# --- character-level tokenizer ---
chars = sorted(set(text))
V = len(chars)                                   # 65
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# --- train / validation split: last 10% is held out ---
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size, batch_size = 8, 32
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])          # (B, T)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])  # targets = x shifted by one
    return x, y

class Bigram(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.table = nn.Embedding(V, V)          # row i = logits for the char after char i
    def forward(self, idx, targets=None):
        logits = self.table(idx)                 # (B, T, V)
        loss = None
        if targets is not None:
            # cross_entropy wants (N, V) logits and (N,) targets, so flatten B and T
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss

model = Bigram(V)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(3001):
    x, y = get_batch("train")
    _, loss = model(x, y)              # forward
    opt.zero_grad(set_to_none=True)    # clear old grads
    loss.backward()                    # compute new grads
    opt.step()                         # update weights
    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20
        model.train()
        print(f"step {step:4d}  train {loss.item():.3f}  val {val_loss:.3f}")

# --- sample 300 characters ---
idx = torch.zeros((1, 1), dtype=torch.long)       # start from char 0 (a newline)
with torch.no_grad():
    for _ in range(300):
        logits, _ = model(idx[:, -1:])            # bigram only needs the last char
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
print("".join(chars[i] for i in idx[0].tolist()))` },
        { t: 'h3', x: 'Reading the numbers' },
        { t: 'table', head: ['Moment', 'Expected loss', 'Why'], rows: [
          ['Perfectly uniform guess', 'ln(65) ≈ 4.17', 'Cross-entropy of a uniform distribution over V classes is ln V'],
          ['Step 0 (random init)', '≈ 4.5–5', 'Random N(0,1) logits are worse than uniform: confidently wrong'],
          ['After ~3,000 steps', '≈ 2.45–2.5 train and val', 'The best a pairs-only model can do on this text'],
          ['Train ≪ val', '—', 'Overfitting; with 4,225 parameters and 1M chars the bigram cannot overfit'],
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Expect to reason from loss values: initial loss ≈ ln(V), [[perplexity]] = exp(loss), and "training loss falls while validation loss rises" = overfitting.',
          'Know the loop order and why `zero_grad` exists; know that a training step costs forward + backward ≈ 3× the forward compute.',
        ]},

        { t: 'h2', x: 'From laptop to cluster' },
        { t: 'p', x: 'Nothing about this loop changes at scale. [[nemo]] and [[megatron]] wrap it in distributed data loaders, [[mixed-precision|mixed precision]] and model parallelism, but each step is still forward, loss, backward, optimizer step. When you later read a NeMo training log, "reduced_train_loss" and "val_loss" mean exactly the two numbers you just printed.' },

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Think in shapes: ids (B, T) → activations (B, T, C) → logits (B, T, V).',
          'Autograd records the forward pass and `backward()` fills `.grad`; gradients accumulate unless cleared.',
          'The loop is forward → loss → zero_grad → backward → step, with periodic no-grad validation.',
          'Language-model targets are the inputs shifted by one token; the loss is mean cross-entropy over every position.',
          'Start-of-training loss ≈ ln(V); a held-out validation split is how you detect overfitting.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'attention',
      title: 'Self-attention and the transformer block, from scratch',
      minutes: 24,
      outcomes: [
        'Explain queries, keys and values intuitively and as matrix maths',
        'Implement scaled dot-product attention with a causal mask',
        'Split attention into multiple heads and reason about the reshapes',
        'Assemble a pre-norm transformer block with residuals and an MLP',
      ],
      blocks: [
        { t: 'lede', x: 'The bigram model sees one character. To predict well, a token must gather information from everything before it — and decide **which** earlier tokens matter. [[self-attention|Self-attention]] is the mechanism that does this, and a [[transformer]] is just attention plus a small MLP, stacked many times with residual connections. By the end of this lesson you will have written both.' },

        { t: 'h2', x: 'Intuition: queries, keys and values' },
        { t: 'p', x: 'Every token produces three vectors from its own embedding through three learned linear layers:' },
        { t: 'ul', items: [
          '**Query (q)** — "what am I looking for?" (e.g. a verb looking for its subject).',
          '**Key (k)** — "what do I contain?" — the label other tokens match their queries against.',
          '**Value (v)** — "what will I hand over if you attend to me?"',
        ]},
        { t: 'p', x: 'Token t compares its query with the key of every token it is allowed to see (a dot product — large when they point the same way), turns those scores into weights with [[softmax]], and outputs the weighted average of the values. It is a soft, learned, differentiable dictionary lookup. Because q, k and v are different projections, "what I search for" can differ from "what I advertise" and "what I give".' },

        { t: 'h2', x: 'The maths: scaled dot-product attention' },
        { t: 'p', x: 'Stack all tokens of a sequence into matrices Q, K, V of shape (T, d). Then:' },
        { t: 'code', lang: 'text', title: 'Attention in one line', x: `Attention(Q, K, V) = softmax( Q Kᵀ / √d_k  +  M ) V

Q Kᵀ        (T, T)  score of every query against every key
/ √d_k              keeps scores at unit variance so softmax does not saturate
M           (T, T)  causal mask: 0 on/below the diagonal, −∞ above it
softmax             row-wise → each row is a probability distribution over positions
· V         (T, d)  weighted average of value vectors` },
        { t: 'p', x: 'Why divide by √d_k? If q and k have unit-variance components, their dot product has variance d_k. With d_k = 64 the raw scores would have a standard deviation of 8; softmax of such large numbers is almost one-hot, so gradients vanish. Scaling restores unit variance.' },
        { t: 'h3', x: 'The causal mask' },
        { t: 'p', x: 'A language model is trained to predict token t+1 from tokens ≤ t. If position t could attend to t+1 it would simply copy the answer. The [[causal-mask]] sets the scores above the diagonal to −∞ before the softmax, so their weights become exactly 0. One forward pass therefore trains T predictions at once — position 0 predicts token 1, position 1 predicts token 2, and so on — without any leakage. This is what makes [[decoder-only]] training so efficient.' },
        { t: 'code', lang: 'text', title: 'Causal attention weights for T = 4 (rows = queries)', x: `          k0    k1    k2    k3
q0      1.00  0     0     0
q1      0.38  0.62  0     0
q2      0.10  0.55  0.35  0
q3      0.21  0.09  0.30  0.40     <- each row sums to 1, nothing above the diagonal` },

        { t: 'h2', x: 'Multi-head attention' },
        { t: 'p', x: 'One attention pattern per layer is limiting: a token may need its subject, the previous word and a matching bracket simultaneously. [[multi-head-attention|Multi-head attention]] splits the C channels into H heads of size d_head = C / H, runs attention independently in each, concatenates the results and mixes them with an output projection. The cost is the same as one big head — you just slice the channels.' },
        { t: 'specs', rows: [
          ['Input', 'x: (B, T, C)'],
          ['QKV projection', 'one Linear C → 3C, split into q, k, v each (B, T, C)'],
          ['Split heads', 'view (B, T, H, d_head) then transpose → (B, H, T, d_head)'],
          ['Scores', 'q @ kᵀ → (B, H, T, T), scale by 1/√d_head, mask, softmax'],
          ['Weighted values', 'weights @ v → (B, H, T, d_head)'],
          ['Merge heads', 'transpose → (B, T, H, d_head), reshape → (B, T, C), output Linear C → C'],
          ['Parameters', '4·C² (Q, K, V and output projections, no bias)'],
        ]},
        { t: 'code', lang: 'python', title: 'CausalSelfAttention, written out explicitly', x: `import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.d_head = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)   # Q, K, V in one matmul
        self.proj = nn.Linear(n_embd, n_embd, bias=False)      # mixes the heads back together
        self.drop = nn.Dropout(dropout)
        # lower-triangular ones: 1 = may attend, 0 = future position
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                  # each (B, T, C)
        # split channels into heads: (B, T, C) -> (B, H, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)        # (B, H, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))        # causal mask
        att = self.drop(F.softmax(att, dim=-1))
        y = att @ v                                                      # (B, H, T, d_head)

        y = y.transpose(1, 2).contiguous().view(B, T, C)                 # merge heads
        return self.drop(self.proj(y))

# quick shape test
attn = CausalSelfAttention(n_embd=64, n_head=4, block_size=16)
print(attn(torch.randn(2, 16, 64)).shape)   # torch.Size([2, 16, 64])` },
        { t: 'callout', kind: 'tip', title: 'Use the fused kernel in real code', x: 'The four lines from `att = …` to `y = att @ v` can be replaced by `F.scaled_dot_product_attention(q, k, v, is_causal=True)`. PyTorch then dispatches to a fused, memory-efficient kernel (FlashAttention-style on supported NVIDIA GPUs) that never materialises the (T, T) matrix. The explicit version is for understanding; the fused one is what you train with.' },

        { t: 'h2', x: 'The transformer block: attention + MLP + residuals + norms' },
        { t: 'p', x: 'Attention moves information **between** positions. The [[ffn|feed-forward network]] (MLP) then processes each position **independently**: Linear C → 4C, a nonlinearity (GELU), Linear 4C → C. Roughly two-thirds of a dense transformer’s parameters live in these MLPs (8·C² vs 4·C² for attention).' },
        { t: 'p', x: 'Two ingredients make deep stacks trainable. [[residual|Residual connections]] add each sub-layer’s output to its input (x = x + f(x)), creating a "residual stream" that gradients can flow down unimpeded. [[layernorm|Layer normalisation]] rescales activations to zero mean and unit variance per token so magnitudes stay stable. Modern models apply the norm **before** each sub-layer (pre-norm), which trains more stably than the original post-norm layout.' },
        { t: 'vs', cards: [
          { h: 'Post-norm (original 2017 transformer)', items: [
            'x = Norm(x + Attn(x)); x = Norm(x + MLP(x))',
            'Norm sits on the residual path itself',
            'Needs careful learning-rate warmup; unstable when deep',
            'Still seen in BERT-era encoders',
          ]},
          { h: 'Pre-norm (GPT-2 onwards, all modern LLMs)', items: [
            'x = x + Attn(Norm(x)); x = x + MLP(Norm(x))',
            'Residual stream is a clean identity path',
            'Stable at 100+ layers; needs a final norm before the LM head',
            'Llama, Qwen, Mistral, Nemotron use pre-norm (with RMSNorm)',
          ]},
        ]},
        { t: 'code', lang: 'python', title: 'MLP and pre-norm Block (uses CausalSelfAttention above)', x: `class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)     # expand
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)   # contract
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # communicate: tokens read from earlier tokens
        x = x + self.mlp(self.ln2(x))    # compute: each token thinks on its own
        return x

blk = Block(n_embd=64, n_head=4, block_size=16)
print(blk(torch.randn(2, 16, 64)).shape)                      # (2, 16, 64): shape preserved
print(sum(p.numel() for p in blk.parameters()))               # 49,408 ≈ 12·64² + LayerNorm params` },
        { t: 'callout', kind: 'warn', title: 'Attention has no idea of order', x: 'Without extra information, attention is permutation-invariant: shuffle the tokens and each output is the same weighted sum. Models therefore add [[positional-encoding|positional information]] — learned position embeddings in GPT-2 and in the next lesson’s mini-GPT, [[rope|RoPE]] in modern LLMs. Forgetting it gives a model that cannot tell "dog bites man" from "man bites dog".' },

        { t: 'h2', x: 'Cost: why context length is expensive' },
        { t: 'p', x: 'The score matrix is T×T per head. Doubling the [[context-window]] quadruples attention compute and (in a naive implementation) its memory. For short contexts the MLP dominates FLOPs; past a few thousand tokens attention takes over. That quadratic term is why fused attention kernels, the [[kv-cache]], [[gqa|GQA]] and sliding-window variants exist.' },
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Know the formula softmax(QKᵀ/√d_k)V, why the √d_k scaling exists and what the causal mask prevents.',
          'Shape questions: with C = 4096 and 32 heads, d_head = 128; the score tensor is (B, H, T, T).',
          'Know that attention parameters ≈ 4C² per layer, MLP ≈ 8C² (with 4× expansion), and that pre-norm + residuals are what make deep stacks stable.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Attention = softmax(QKᵀ/√d_k + mask)·V: a learned, soft lookup over earlier tokens.',
          'The causal mask lets one forward pass train every position in parallel without seeing the future.',
          'Multi-head attention slices C into H heads of d_head = C/H; the reshape is view → transpose and back.',
          'A block is x + Attn(Norm(x)) then x + MLP(Norm(x)); ~12·C² parameters per layer.',
          'Attention is order-blind (needs positional info) and quadratic in sequence length.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'mini-gpt',
      title: 'Lab: build and train a tiny GPT',
      minutes: 35,
      outcomes: [
        'Build a complete GPT: embeddings, block stack, final norm and tied LM head',
        'Train it on tiny Shakespeare on CPU, Apple MPS or a free Colab GPU',
        'Generate text and control it with temperature and top-k',
        'Count parameters with the 12·L·d² rule and predict the loss you should reach',
      ],
      blocks: [
        { t: 'lede', x: 'Time to put the pieces together. In about 150 lines you will build a nanoGPT-style model — the same architecture family as GPT-2 — train it on tiny Shakespeare, and watch it go from random characters to passable pseudo-Elizabethan verse. Everything later in the course ([[mixed-precision|mixed precision]], distributed training, [[lora|LoRA]], quantization) is an optimisation of what you build here.' },

        { t: 'h2', x: 'Architecture at a glance' },
        { t: 'flow', steps: [
          { b: 'Token embedding', s: 'ids (B, T) → (B, T, C) via a V×C table ([[embedding]]).' },
          { b: '+ position embedding', s: 'a learned T_max×C table indexed by position 0…T−1, added to the token vectors.' },
          { b: 'L × Block', s: 'pre-norm attention and MLP with residuals (the lesson 2 code).' },
          { b: 'Final LayerNorm', s: 'needed with pre-norm because the residual stream is never normalised otherwise.' },
          { b: 'LM head', s: 'Linear C → V producing logits; its weight is **tied** to the token embedding.' },
          { b: 'Loss / sampling', s: 'cross-entropy against shifted targets in training; softmax + sample at inference.' },
        ]},
        { t: 'p', x: '[[weight-tying|Weight tying]] reuses the V×C embedding matrix as the output projection. It saves V·C parameters (38M for GPT-2’s 50k×768 vocabulary) and usually helps small models. Large models such as Llama 3 8B leave it off; small ones such as Qwen2.5-0.5B and SmolLM2 keep it on.' },

        { t: 'lab', title: 'Train a 3M-parameter GPT on tiny Shakespeare', x: 'Run one self-contained script that downloads the data, builds a 4-layer GPT, trains it for 3,000 steps, reports train/val loss and samples text at two temperatures. Then change one hyperparameter at a time and observe the effect.', gpu: 'CPU / Apple Silicon OK (free Colab T4 is faster)', time: '30–45 min', steps: [
          'Create a fresh environment with Python 3.10+ and `pip install torch` (any PyTorch 2.x).',
          'Save the script below as `minigpt.py` and run `python minigpt.py`. It picks CUDA, then MPS, then CPU automatically.',
          'Watch the log: loss should start near 4.2 and val loss should fall below 2.0 within ~1,000 steps.',
          'Read the two samples: temperature 0.8 vs 1.2. Note the difference in coherence vs variety.',
          'Experiment: set `n_layer = 6, n_embd = 384` (≈10.7M params) and compare final val loss and time per step.',
          'Experiment: set `dropout = 0.0` and `max_iters = 8000` and watch for train loss pulling away from val loss.',
        ]},
        { t: 'code', lang: 'python', title: 'minigpt.py — complete model + training loop', x: `import math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- hyperparameters ----------------
batch_size = 32       # sequences per step
block_size = 128      # context length (T)
n_layer, n_head, n_embd = 4, 4, 256
dropout = 0.1
max_iters, eval_every, eval_iters = 3000, 300, 50
lr, min_lr, warmup = 1e-3, 1e-4, 100
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)

# ---------------- data: chars of tiny Shakespeare ----------------
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(url, "input.txt")
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- model ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))
        # fused kernel: same maths as the explicit version in lesson 2
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(n_embd), CausalSelfAttention()
        self.ln2, self.mlp = nn.LayerNorm(n_embd), MLP()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))                     # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]                     # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature             # last position only
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")     # keep only the top k
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx

model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters())          # tied weights counted once
print(f"device={device}  params={n_params/1e6:.2f}M  (12*L*d^2 = {12*n_layer*n_embd*n_embd/1e6:.2f}M)")

# ---------------- training ----------------
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

def lr_at(it):                                   # linear warmup, then cosine decay to min_lr
    if it < warmup:
        return lr * (it + 1) / warmup
    progress = (it - warmup) / max(1, max_iters - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = torch.stack([model(*get_batch(split))[1] for _ in range(eval_iters)]).mean().item()
    model.train()
    return out

t0 = time.time()
for it in range(max_iters + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    if it % eval_every == 0:
        l = estimate_loss()
        print(f"step {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time()-t0:.0f}s")
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # guards against loss spikes
    opt.step()

# ---------------- sampling ----------------
start = torch.tensor([encode("ROMEO:")], device=device)
for temp in (0.8, 1.2):
    print(f"\\n--- temperature {temp} ---")
    print(decode(model.generate(start, 300, temperature=temp, top_k=50)[0].tolist()))
torch.save(model.state_dict(), "minigpt.pt")` },
        { t: 'callout', kind: 'tip', title: 'Running on Colab', x: 'In a Colab notebook, select Runtime → Change runtime type → T4 GPU, paste the script into a cell and run it. On a T4 the default config takes a couple of minutes; on a laptop CPU budget roughly 10–20 minutes, and Apple Silicon (MPS) sits in between. Times vary a lot with hardware, so treat these as rough.' },

        { t: 'h2', x: 'Counting parameters: the 12·L·d² rule' },
        { t: 'p', x: 'Each block has 4d² parameters in attention (Q, K, V, output) and 8d² in a 4×-expansion MLP: **12·d² per layer**, so the non-embedding parameters are ≈ 12·L·d². Add V·d for the token embedding (counted once when tied) and T_max·d for learned positions. Norm and bias parameters are negligible.' },
        { t: 'table', head: ['Model', 'L', 'd', '12·L·d²', 'Embeddings', 'Total (reported)'], rows: [
          ['This lab (default)', '4', '256', '3.15M', '65·256 + 128·256 ≈ 0.05M', '≈ 3.2M'],
          ['This lab (bigger)', '6', '384', '10.6M', '≈ 0.07M', '≈ 10.7M'],
          ['GPT-2 small', '12', '768', '85M', '50,257·768 + 1,024·768 ≈ 39M', '124M'],
          ['GPT-2 XL', '48', '1600', '1.47B', '≈ 82M', '1.5B'],
        ]},
        { t: 'p', x: 'Note how embeddings dominate tiny models with big vocabularies — a third of GPT-2 small is its embedding table — while the 12·L·d² term dominates large ones. Modern LLMs change the constant a little (SwiGLU MLPs, [[gqa|GQA]]), which you will see in the next lesson, but the d² scaling holds.' },

        { t: 'h2', x: 'What loss to expect' },
        { t: 'table', head: ['Checkpoint', 'Val loss (char-level)', 'What samples look like'], rows: [
          ['Step 0', '≈ 4.2 (= ln 65)', 'Uniform noise over 65 characters'],
          ['Bigram ceiling', '≈ 2.5', 'Pronounceable gibberish, no real words'],
          ['3M GPT, 3k steps', '≈ 1.6–1.8', 'Real words, speaker names, line breaks, broken grammar'],
          ['10M GPT, 5k steps (nanoGPT reference)', '≈ 1.5', 'Convincing-looking Shakespeare, little meaning'],
          ['Train ≪ val (e.g. 1.0 vs 1.7)', '—', 'Memorising: add dropout, stop earlier or get more data'],
        ]},
        { t: 'callout', kind: 'warn', title: 'Character loss is not comparable to token loss', x: 'Loss is per prediction. A character model predicts ~4× more often per word than a BPE model, so its per-step loss is lower than a subword model’s on the same text. Only compare losses (and [[perplexity]]) between models that share a tokenizer, or convert to bits-per-byte.' },

        { t: 'h2', x: 'Sampling and temperature' },
        { t: 'p', x: 'Generation is autoregressive: feed the context, take the logits of the last position, divide by a temperature, optionally keep only the top-k, sample one token, append, repeat. Temperature T rescales logits before softmax: T < 1 sharpens the distribution (safer, more repetitive), T > 1 flattens it (more varied, more errors), and T → 0 approaches greedy argmax.' },
        { t: 'vs', cards: [
          { h: 'Temperature 0.8, top-k 50', items: [
            'Mostly real words and plausible names',
            'Repeats common phrases ("my lord", "the king")',
            'Good default for a small model',
          ]},
          { h: 'Temperature 1.2, top-k 50', items: [
            'More invented words and odd punctuation',
            'More variety, less coherence',
            'Shows how much probability mass sits on mistakes',
          ]},
        ]},
        { t: 'p', x: 'Notice that generation recomputes the whole context at every step. Real inference engines cache each layer’s keys and values instead — the [[kv-cache]] — so each new token only computes one new row of attention. That optimisation, and the memory it costs, is a major theme of the inference modules.' },

        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Parameter arithmetic: non-embedding params ≈ 12·L·d²; with tying, embedding counted once; 2 bytes/param in BF16 → a 124M model is ~250 MB of weights.',
          'Decoding knobs: lower temperature = more deterministic; top-k restricts sampling to the k most likely tokens.',
          'Reading curves: initial loss ≈ ln(V); a widening train/val gap means overfitting.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'A GPT is embeddings + L pre-norm blocks + final norm + (tied) LM head; ~150 lines of PyTorch.',
          'Train with AdamW, warmup + cosine decay, gradient clipping; evaluate on held-out data.',
          'Non-embedding parameters ≈ 12·L·d²; embeddings matter a lot for small models with big vocabularies.',
          'Temperature and top-k shape the output distribution at sampling time without retraining.',
          'Compare losses only between models that share a tokenizer.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'modern-architectures',
      title: 'Modern LLM design: RoPE, GQA, SwiGLU, MoE and scaling laws',
      minutes: 25,
      outcomes: [
        'Explain what RoPE, RMSNorm and SwiGLU change relative to GPT-2 and why',
        'Calculate KV-cache memory for MHA, GQA and MQA configurations',
        'Distinguish total and active parameters in a mixture-of-experts model',
        'Choose between encoder-only, decoder-only and encoder-decoder for a task',
        'Apply Chinchilla scaling (≈20 tokens/param, C ≈ 6ND) and read a config.json',
      ],
      blocks: [
        { t: 'lede', x: 'Your mini-GPT is architecturally a 2019 model. A 2026 open model — Llama, Qwen, Mistral, Nemotron — is the same skeleton with five well-understood upgrades: rotary positions, RMSNorm, SwiGLU MLPs, grouped-query attention and (for the largest ones) mixture-of-experts. This lesson explains each, the numbers behind it, and how to read them straight out of a `config.json`.' },

        { t: 'h2', x: 'Positions: from learned tables to RoPE' },
        { t: 'p', x: 'GPT-2 adds a learned vector per absolute position, which cannot extrapolate past the trained length. [[rope|Rotary position embeddings (RoPE)]] instead **rotate** each pair of dimensions of q and k by an angle proportional to the position. Because the dot product of two rotated vectors depends only on the difference of their angles, attention scores depend on **relative** distance, and nothing is added to the residual stream.' },
        { t: 'code', lang: 'python', title: 'RoPE in a few lines (applied to q and k, not v)', x: `import torch

def rope_cache(T, d_head, theta=10000.0):
    # one frequency per pair of dims: theta^(-2i/d)
    inv_freq = 1.0 / torch.pow(theta, torch.arange(0, d_head, 2).float() / d_head)
    angles = torch.outer(torch.arange(T).float(), inv_freq)     # (T, d_head/2)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):            # x: (B, H, T, d_head)
    x1, x2 = x[..., 0::2], x[..., 1::2]  # split into pairs
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)               # rotate each pair, re-interleave

cos, sin = rope_cache(T=16, d_head=64)
q = torch.randn(2, 4, 16, 64)
print(apply_rope(q, cos, sin).shape)     # (2, 4, 16, 64) — norms unchanged, only rotated` },
        { t: 'p', x: 'The base `rope_theta` sets how slowly the low-frequency pairs rotate. Long-context models raise it (Llama 3 uses 500,000; Qwen2.5 uses 1,000,000) and context-extension methods such as position interpolation, NTK-aware scaling and YaRN rescale these frequencies so a model trained on 8k tokens can be adapted to 32k–128k with modest extra training. You will see this as a `rope_scaling` entry in configs.' },

        { t: 'h2', x: 'RMSNorm and SwiGLU' },
        { t: 'p', x: '[[rmsnorm|RMSNorm]] drops LayerNorm’s mean subtraction and bias: it divides by the root-mean-square of the vector and multiplies by a learned gain. It is cheaper and empirically just as good, so every Llama-family model uses it, in pre-norm position.' },
        { t: 'p', x: '[[swiglu|SwiGLU]] replaces the GELU MLP with a **gated** one: two parallel up-projections, one passed through SiLU acting as a gate on the other, then a down-projection. Three matrices instead of two, so to keep parameters constant the hidden size shrinks from 4d to about (8/3)·d; Llama 3 8B uses d = 4096 and an intermediate size of 14,336 (3.5d).' },
        { t: 'code', lang: 'python', title: 'RMSNorm and a SwiGLU MLP', x: `import torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)   # names match HF Llama/Qwen weights
        self.up_proj   = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

x = torch.randn(2, 8, 512)
print(SwiGLU(512, 1408)(RMSNorm(512)(x)).shape)          # (2, 8, 512)` },

        { t: 'h2', x: 'GQA and MQA: shrinking the KV cache' },
        { t: 'p', x: 'During generation every past token’s keys and values are cached for every layer. With standard multi-head attention (MHA) each of the H query heads has its own K and V head. [[mqa|Multi-query attention (MQA)]] shares **one** K/V head across all query heads; [[gqa|grouped-query attention (GQA)]] is the middle ground — G groups of query heads each share one K/V head. Quality stays close to MHA, while cache size drops by H/G.' },
        { t: 'code', lang: 'text', title: 'KV-cache size per token', x: `bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_value

Llama-3-8B shape: 32 layers, 32 query heads, head_dim 128, BF16 (2 bytes)
  MHA (32 KV heads):  2 × 32 × 32 × 128 × 2 = 524,288 B = 512 KiB / token
  GQA  (8 KV heads):  2 × 32 ×  8 × 128 × 2 = 131,072 B = 128 KiB / token   <- the real config
  MQA  (1 KV head):   2 × 32 ×  1 × 128 × 2 =  16,384 B =  16 KiB / token

One 8,192-token sequence:   MHA 4 GiB   |   GQA 1 GiB   |   MQA 128 MiB
32 such sequences in a batch: MHA 128 GiB (does not fit an 80 GB GPU) | GQA 32 GiB` },
        { t: 'table', head: ['Variant', 'K/V heads', 'KV cache vs MHA', 'Quality', 'Used by'], rows: [
          ['MHA', 'H (one per query head)', '1×', 'Baseline', 'GPT-2, original Llama 1/2-7B'],
          ['GQA', 'G, with 1 < G < H', 'G/H (e.g. 8/32 = ¼)', 'Near MHA', 'Llama 2-70B, Llama 3, Qwen2.5, Mistral, most 2024–26 models'],
          ['MQA', '1', '1/H', 'Some degradation', 'PaLM, Falcon-7B, some small/edge models'],
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'GQA/MQA questions are almost always about **inference memory and throughput**: fewer KV heads → smaller [[kv-cache]] → larger batch or longer context on the same GPU. They do not reduce weight memory meaningfully.',
          'Be ready to compute bytes/token = 2 × layers × kv_heads × head_dim × bytes, and to scale it by sequence length and batch.',
        ]},

        { t: 'h2', x: 'Mixture of experts: big capacity, small compute' },
        { t: 'p', x: 'A [[moe|mixture-of-experts (MoE)]] layer replaces the single MLP with E expert MLPs and a small **router** that, for each token, picks the top-k experts (typically k = 1–8) and mixes their outputs by the router’s weights. Only the chosen experts run, so compute per token tracks the **active** parameters while knowledge capacity tracks the **total** parameters.' },
        { t: 'table', head: ['Model', 'Experts (routed)', 'Top-k', 'Total params', 'Active per token'], rows: [
          ['Mixtral 8×7B', '8', '2', '≈ 47B', '≈ 13B'],
          ['DeepSeek-V3', '256 + 1 shared', '8', '671B', '37B'],
          ['Dense 13B (for contrast)', '—', '—', '13B', '13B'],
        ]},
        { t: 'ul', items: [
          '**Memory** follows total parameters: Mixtral must hold all ~47B weights (~94 GB in BF16) even though it computes like a ~13B model.',
          '**Load balancing**: routers collapse onto a few favourite experts unless trained with an auxiliary balancing loss (or a bias-based balancing scheme, as in DeepSeek-V3).',
          '**Parallelism**: experts are spread across GPUs (expert parallelism), which adds all-to-all communication — covered in the GPU module.',
        ]},
        { t: 'callout', kind: 'trap', title: '"8×7B" is not 56B, and not 7B', x: 'Attention layers, embeddings and norms are shared, so Mixtral 8×7B totals ≈ 47B, not 8 × 7 = 56B. And it does not cost what a 7B costs to serve: you need memory for all 47B weights. Exam answers that size MoE serving by active parameters alone are wrong.' },

        { t: 'h2', x: 'Encoder, decoder, or both?' },
        { t: 'vs', cards: [
          { h: '[[encoder-only|Encoder-only]] (BERT, RoBERTa)', items: [
            'Bidirectional attention: every token sees the whole input',
            'Trained with masked-language modelling',
            'Best for classification, NER, embeddings, reranking',
            'Cannot generate text autoregressively',
          ]},
          { h: '[[decoder-only|Decoder-only]] (GPT, Llama, Qwen, Nemotron)', items: [
            'Causal attention; next-token prediction',
            'One architecture for generation, chat, code, reasoning',
            'Scales simply; the default for LLMs since ~2020',
            'Uses a KV cache at inference',
          ]},
          { h: '[[encoder-decoder|Encoder-decoder]] (T5, BART, Whisper)', items: [
            'Encoder reads the input bidirectionally; decoder generates with cross-attention to it',
            'Natural fit for translation, summarisation, speech-to-text',
            'Two stacks, more moving parts to serve',
            'Still common in speech and some seq2seq services',
          ]},
        ]},

        { t: 'h2', x: 'Scaling laws and Chinchilla' },
        { t: 'p', x: '[[scaling-laws|Scaling laws]] say loss falls predictably as a power law in parameters N, data D and compute C. Training compute is well approximated by **C ≈ 6·N·D** FLOPs (2ND for the forward pass, 4ND for backward). The [[chinchilla|Chinchilla]] study (Hoffmann et al., 2022) found that for a fixed compute budget, N and D should grow together — roughly **20 training tokens per parameter**.' },
        { t: 'table', head: ['Model size N', 'Chinchilla tokens (20·N)', 'Compute 6·N·D', 'Notes'], rows: [
          ['125M', '2.5B', '1.9 × 10¹⁸ FLOPs', 'Hours on one modern GPU'],
          ['1B', '20B', '1.2 × 10²⁰ FLOPs', 'At 40% MFU on one H100 (~400 TFLOPS dense BF16 effective): ≈ 3–4 days'],
          ['8B', '160B', '7.7 × 10²¹ FLOPs', 'Llama 3 8B actually used ~15T tokens — ~1,900 tokens/param'],
          ['70B', '1.4T', '5.9 × 10²³ FLOPs', 'Chinchilla itself: 70B params, 1.4T tokens'],
        ]},
        { t: 'p', x: 'Chinchilla optimises **training** compute. If a model will serve billions of requests, it is cheaper overall to train a smaller model on far more tokens ("over-training"), because inference cost scales with N. That is why modern small models are trained on 10–100× their Chinchilla token count.' },
        { t: 'callout', kind: 'trap', title: 'Compute-optimal is not deployment-optimal', x: 'If a question asks for the best model for a fixed **training** budget, apply ~20 tokens/param. If it stresses **inference cost or latency**, the right answer is usually a smaller model trained on more data than Chinchilla suggests.' },

        { t: 'h2', x: 'Reading a real config.json' },
        { t: 'p', x: 'Every Hugging Face model ships a `config.json` that tells you its architecture in a dozen fields. This is the config of `Qwen/Qwen2.5-0.5B`, the small model you will fine-tune later in the course:' },
        { t: 'code', lang: 'json', title: 'Qwen2.5-0.5B config.json (key fields, annotated — real JSON has no comments)', x: `{
  "architectures": ["Qwen2ForCausalLM"],     // decoder-only causal LM
  "hidden_size": 896,                          // d
  "intermediate_size": 4864,                   // SwiGLU hidden size (~5.4 d)
  "num_hidden_layers": 24,                     // L
  "num_attention_heads": 14,                   // query heads -> head_dim = 896/14 = 64
  "num_key_value_heads": 2,                    // GQA: 7 query heads share each KV head
  "hidden_act": "silu",                        // SwiGLU gate activation
  "rms_norm_eps": 1e-06,                       // RMSNorm
  "rope_theta": 1000000.0,                     // RoPE base, raised for long context
  "max_position_embeddings": 32768,
  "vocab_size": 151936,
  "tie_word_embeddings": true,                 // LM head shares the embedding matrix
  "torch_dtype": "bfloat16"
}` },
        { t: 'p', x: 'From these numbers: [[kv-cache|KV cache]] = 2 × 24 × 2 × 64 × 2 bytes = **12 KiB per token**, so a 32k-token context costs only ~384 MiB. The embedding table is 151,936 × 896 ≈ 136M parameters — over a quarter of the ~0.5B total, which is why tying matters for small models. A Llama-3-8B config reads the same way: hidden 4096, intermediate 14,336, 32 layers, 32 heads, 8 KV heads, vocab 128,256, rope_theta 500,000, untied embeddings. NVIDIA’s Nemotron family publishes the same kind of config (some recent variants are hybrid Mamba-Transformer models and add fields for their state-space layers), and [[nemo]] / [[megatron]] use equivalent YAML fields such as `num_layers`, `hidden_size`, `num_attention_heads` and `num_query_groups`.' },
        { t: 'code', lang: 'python', title: 'Inspect any config without downloading the weights', x: `from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
head_dim = cfg.hidden_size // cfg.num_attention_heads
kv_bytes = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2   # BF16
print(f"layers={cfg.num_hidden_layers} d={cfg.hidden_size} heads={cfg.num_attention_heads} "
      f"kv_heads={cfg.num_key_value_heads} head_dim={head_dim}")
print(f"KV cache: {kv_bytes/1024:.0f} KiB/token, "
      f"{kv_bytes*cfg.max_position_embeddings/(1 << 30):.2f} GiB at full context")` },

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Modern recipe: decoder-only + pre-RMSNorm + RoPE + SwiGLU + GQA (+ MoE at large scale).',
          'KV cache bytes/token = 2 × layers × kv_heads × head_dim × bytes; GQA/MQA cut it by H/G.',
          'MoE: compute ∝ active params, memory ∝ total params; routers need load balancing.',
          'Encoders for understanding/embeddings, decoders for generation, encoder-decoders for seq2seq.',
          'C ≈ 6ND; Chinchilla-optimal ≈ 20 tokens/param, but inference-heavy models are deliberately over-trained.',
        ]},
      ],
    },
  ],

  /* -------------------------------------------------------------------- */
  quiz: [
    { id: 'a01', domain: 'architecture',
      scenario: 'A new team member trains a character-level model with a 65-symbol vocabulary. The first logged loss is 41.7 and it falls slowly.',
      q: 'What is the most likely explanation?',
      opts: [
        'This is normal; character models always start with very high loss',
        'Something is wrong with initialisation or the loss computation — a sane start is near ln(65) ≈ 4.2',
        'The learning rate is too low, which inflates the first loss value',
        'The validation split is too small to give a meaningful first loss',
      ],
      a: 1,
      why: 'A freshly initialised model should be close to a uniform guess, whose [[cross-entropy]] is ln(V) ≈ 4.17 for V = 65. A value ten times higher means badly scaled weights (e.g. huge logits), a loss summed instead of averaged, or wrong labels.',
      distract: 'High start loss is not normal — ln(V) is the reference. The learning rate does not affect the loss before any update. The first **training** loss does not depend on the validation split.' },

    { id: 'a02', domain: 'architecture',
      q: 'Why is the attention score matrix divided by √d_k before the softmax?',
      opts: [
        'To make the attention weights sum to one across each row',
        'To normalise the value vectors to unit length',
        'To keep dot-product variance near 1 so the softmax does not saturate and gradients do not vanish',
        'To make attention cost linear rather than quadratic in sequence length',
      ],
      a: 2,
      why: 'With unit-variance components, q·k has variance d_k. Dividing by √d_k restores unit variance; otherwise softmax becomes nearly one-hot and its gradients vanish.',
      distract: 'Rows summing to one is the **softmax**’s job. Values are not normalised by this factor. Scaling does nothing to the T×T cost.' },

    { id: 'a03', domain: 'architecture',
      scenario: 'A model has 32 layers, 32 query heads, 8 key/value heads, head_dim 128, and stores its KV cache in BF16.',
      q: 'How much KV cache does one 4,096-token sequence need?',
      opts: [
        '128 MiB',
        '256 MiB',
        '2 GiB',
        '512 MiB',
      ],
      a: 3,
      why: 'Per token: 2 × 32 layers × 8 KV heads × 128 × 2 bytes = 131,072 B = 128 KiB. × 4,096 tokens = 512 MiB. This is the [[gqa|GQA]] Llama-3-8B shape.',
      distract: '128 MiB and 256 MiB drop a factor (e.g. forgetting K **and** V, or the 2 bytes). 2 GiB uses all 32 heads for K/V — the MHA figure.' },

    { id: 'a04', domain: 'architecture',
      q: 'A team must serve a mixture-of-experts model with 47B total and 13B active parameters in BF16. What drives the GPU memory needed for the weights?',
      opts: [
        'The total parameters — about 94 GB — because every expert must be resident',
        'The active parameters — about 26 GB — because only routed experts run',
        'The router size, since experts are loaded on demand from disk per token',
        'The number of experts per token multiplied by one expert’s size',
      ],
      a: 0,
      why: 'In [[moe|MoE]] models compute per token tracks active parameters, but any token may route to any expert, so all experts must be in GPU memory: 47B × 2 bytes ≈ 94 GB.',
      distract: 'Sizing by active parameters (26 GB) is the classic mistake. Loading experts from disk per token would be far too slow. Top-k × expert size ignores shared attention/embedding weights and the other experts that must stay resident.' },

    { id: 'a05', domain: 'architecture',
      q: 'Under Chinchilla-style compute-optimal scaling, roughly how many training tokens and FLOPs does a 3B-parameter model need?',
      opts: [
        '3B tokens and 5.4 × 10¹⁹ FLOPs',
        '60B tokens and about 1.1 × 10²¹ FLOPs',
        '60B tokens and about 1.8 × 10²⁰ FLOPs',
        '1.5T tokens and about 2.7 × 10²² FLOPs',
      ],
      a: 1,
      why: '[[chinchilla|Chinchilla]]: D ≈ 20·N = 60B tokens. C ≈ 6·N·D = 6 × 3×10⁹ × 6×10¹⁰ = 1.08 × 10²¹ FLOPs.',
      distract: '3B tokens is one token per parameter. 1.8 × 10²⁰ uses 1·N·D instead of 6·N·D. 1.5T tokens is heavy over-training (~500 tokens/param), not compute-optimal.' },

    { id: 'a06', domain: 'architecture',
      scenario: 'A product team needs a model that classifies support tickets into 40 categories and produces sentence embeddings for search. Generation is not required and latency must be low.',
      q: 'Which architecture family is the most natural fit?',
      opts: [
        'A large decoder-only chat model prompted for each ticket',
        'An encoder-decoder model generating the category name',
        'An encoder-only model such as a BERT-style encoder with a classification head',
        'A mixture-of-experts decoder to cover many categories',
      ],
      a: 2,
      why: '[[encoder-only|Encoder-only]] models attend bidirectionally and produce strong representations for classification and embeddings at low cost — exactly this workload.',
      distract: 'A decoder chat model works but is far heavier than needed. An [[encoder-decoder]] adds a generator you do not need. MoE adds capacity and serving complexity, not suitability for classification.' },

    { id: 'a07', domain: 'architecture',
      q: 'What does rotary position embedding (RoPE) do?',
      opts: [
        'Adds a learned vector for each absolute position to the token embedding',
        'Masks attention so tokens cannot see positions further than a fixed window',
        'Adds a fixed sinusoidal vector to the residual stream at each layer',
        'Rotates pairs of query and key dimensions by position-dependent angles so scores depend on relative distance',
      ],
      a: 3,
      why: '[[rope|RoPE]] applies a position-dependent rotation to q and k only; the dot product of two rotated vectors depends on their position difference, giving relative positioning and better length extrapolation (tunable via `rope_theta` and rope scaling).',
      distract: 'Learned absolute embeddings are GPT-2’s approach. A fixed window is sliding-window attention. Sinusoidal vectors added to the input are the original transformer’s absolute encoding — not rotations, and not per layer.' },

    { id: 'a08', domain: 'architecture',
      q: 'Approximately how many non-embedding parameters does a dense GPT-style transformer with 24 layers and hidden size 2048 have (standard 4× MLP)?',
      opts: [
        '1.2B',
        '100M',
        '400M',
        '4.8B',
      ],
      a: 0,
      why: '12 · L · d² = 12 × 24 × 2048² = 12 × 24 × 4.19M ≈ 1.21B. Each layer has 4d² in attention and 8d² in the MLP.',
      distract: '100M and 400M under-count (e.g. using 12·L·d instead of d², or only the attention 4d²). 4.8B over-counts by 4×.' },
  ],
}
