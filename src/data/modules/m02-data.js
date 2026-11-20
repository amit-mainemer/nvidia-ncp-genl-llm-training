export default {
  id: 'data',
  n: 2,
  domain: 'data',
  weight: 9,
  title: 'Data Preparation',
  blurb: 'Train your own BPE tokenizer, build a pretraining data pipeline (language ID, quality filtering, MinHash dedup, PII redaction, decontamination) with NeMo Curator as the GPU-scale reference, and turn conversations into correctly templated, loss-masked, packed SFT data.',
  hours: '3–4 h',
  lessons: [
    /* ------------------------------------------------------------------ */
    {
      id: 'tokenization',
      title: 'Tokenizers: BPE, vocabulary and special tokens',
      minutes: 25,
      outcomes: [
        'Trace the BPE training algorithm merge by merge',
        'Train a byte-level BPE tokenizer with Hugging Face tokenizers and load it in transformers',
        'Weigh vocabulary-size trade-offs: sequence length, embedding size and multilingual cost',
        'Add tokens to an existing model and resize its embeddings correctly',
        'Explain special tokens, byte-level fallback and why EOS handling matters',
      ],
      blocks: [
        { t: 'lede', x: 'The model never sees text — it sees integer ids produced by a [[tokenizer]]. The tokenizer decides how long every sequence is, how big the embedding table is, how well the model handles code, numbers and non-English text, and where conversations start and stop. It is trained once, before the model, and then frozen: get it wrong and you live with it for the model’s whole life.' },

        { t: 'h2', x: 'Why subwords' },
        { t: 'vs', cards: [
          { h: 'Characters / bytes', items: [
            'Tiny vocabulary (65 chars in lesson 1; 256 bytes)',
            'Never out-of-vocabulary',
            'Very long sequences: ~4–5× more tokens than subwords for English',
            'Attention cost grows quadratically with that length',
          ]},
          { h: 'Whole words', items: [
            'Short sequences',
            'Huge vocabulary (millions of forms across languages)',
            'Unknown words become a useless [UNK]',
            'No sharing between "train", "training", "retrained"',
          ]},
          { h: 'Subwords (BPE, Unigram, WordPiece)', items: [
            'Frequent words are one token; rare words split into pieces',
            'Vocabulary of 32k–256k covers any text',
            'English averages roughly 4 characters (~0.75 words) per token',
            'The standard for every modern LLM',
          ]},
        ]},

        { t: 'h2', x: 'The BPE algorithm, step by step' },
        { t: 'p', x: '[[bpe|Byte-pair encoding (BPE)]] builds a vocabulary bottom-up. Start with single symbols, count every adjacent pair in the training corpus, merge the most frequent pair into a new symbol, and repeat until the vocabulary reaches the target size. The ordered list of merges **is** the tokenizer: encoding new text replays the merges in the same order.' },
        { t: 'flow', steps: [
          { b: 'Pre-tokenise', s: 'Split text into words/chunks (by spaces, punctuation, digits) so merges never cross word boundaries.' },
          { b: 'Initialise', s: 'Each chunk becomes a sequence of base symbols — characters, or the 256 bytes in byte-level BPE.' },
          { b: 'Count pairs', s: 'Count every adjacent symbol pair, weighted by how often the chunk occurs.' },
          { b: 'Merge the top pair', s: 'Replace every occurrence of the most frequent pair with one new symbol; record the merge rule.' },
          { b: 'Repeat', s: 'Until vocab size = base symbols + merges + special tokens reaches the target (e.g. 32k).' },
          { b: 'Encode new text', s: 'Pre-tokenise, split into base symbols, apply the learned merges in priority order.' },
        ]},
        { t: 'code', lang: 'python', title: 'bpe_toy.py — BPE training in 30 lines of pure Python', x: `from collections import Counter

corpus = ("low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3).split()
# each word -> tuple of characters + an end-of-word marker, weighted by frequency
words = Counter(tuple(w) + ("</w>",) for w in corpus)

def pair_counts(words):
    counts = Counter()
    for w, freq in words.items():
        for a, b in zip(w, w[1:]):
            counts[(a, b)] += freq
    return counts

def merge(words, pair):
    out = Counter()
    for w, freq in words.items():
        new, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                new.append(w[i] + w[i + 1]); i += 2      # fuse the pair into one symbol
            else:
                new.append(w[i]); i += 1
        out[tuple(new)] += freq
    return out

merges = []
for step in range(8):
    counts = pair_counts(words)
    best = max(counts, key=counts.get)     # most frequent adjacent pair
    merges.append(best)
    words = merge(words, best)
    print(f"merge {step + 1}: {best} (count {counts[best]})")
print(list(words))
# First merges: ('e','s') 9, ('es','t') 9, ('est','</w>') 9, ('l','o') 7, ('lo','w') 7 ...
# "newest" and "widest" now share the suffix token 'est</w>'.` },
        { t: 'h3', x: 'The other algorithms you should recognise' },
        { t: 'table', head: ['Algorithm', 'How the vocabulary is built', 'Used by'], rows: [
          ['BPE (byte-level)', 'Greedy frequency-based merges over the 256 bytes', 'GPT-2/4, Llama 3, Qwen, Mistral (tiktoken / HF tokenizers)'],
          ['BPE via [[sentencepiece|SentencePiece]]', 'Same merges, but on raw text with spaces encoded as ▁; optional byte fallback', 'Llama 2, Mistral 7B, many multilingual models'],
          ['Unigram LM (SentencePiece)', 'Start large, prune tokens that least reduce corpus likelihood; probabilistic segmentation', 'T5, ALBERT, XLNet, some Gemma-era models'],
          ['[[wordpiece|WordPiece]]', 'Merges chosen by likelihood gain instead of raw count; ## marks continuations', 'BERT, DistilBERT'],
        ]},

        { t: 'h2', x: 'Lab: train a tokenizer' },
        { t: 'lab', title: 'Train an 8k byte-level BPE tokenizer on TinyStories', x: 'Stream 100k short stories from the Hub, train a byte-level BPE tokenizer with the Rust-backed `tokenizers` library, wrap it as a transformers tokenizer and compare its token counts with GPT-2’s 50k vocabulary on in-domain and out-of-domain text.', gpu: 'CPU / Apple Silicon OK', time: '20 min', steps: [
          '`pip install tokenizers transformers datasets` (datasets ≥ 3).',
          'Run the training script below; training on 100k stories takes well under a minute on a laptop.',
          'Run the comparison script and note tokens per sentence for each tokenizer.',
          'Try a sentence in another language and a line of Python: watch your 8k tokenizer fall apart outside its training domain.',
          'Retrain with `vocab_size=32000` and compare the in-domain token counts again.',
        ]},
        { t: 'code', lang: 'python', title: 'train_tokenizer.py', x: `from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Stream: nothing is downloaded up front; we read the first 100k stories lazily.
stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True).take(100_000)

def batches(bs=1000):
    buf = []
    for ex in stream:
        buf.append(ex["text"])
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf

tok = Tokenizer(models.BPE())                                        # no unk token needed:
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # every byte is in the base alphabet
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8192,
    min_frequency=2,                                     # ignore pairs seen only once
    special_tokens=["<|endoftext|>", "<|pad|>"],         # reserved ids 0 and 1, never split
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),# all 256 bytes -> no OOV ever
)
tok.train_from_iterator(batches(), trainer=trainer)
tok.save("tinystories-bpe.json")

enc = tok.encode("Once upon a time, a little dog named Max found a shiny red ball.")
print(len(enc.ids), enc.tokens)       # 'Ġ' marks a leading space in byte-level BPE` },
        { t: 'code', lang: 'python', title: 'compare_tokenizers.py', x: `from transformers import AutoTokenizer, PreTrainedTokenizerFast

mine = PreTrainedTokenizerFast(tokenizer_file="tinystories-bpe.json",
                               eos_token="<|endoftext|>", pad_token="<|pad|>")
mine.save_pretrained("tinystories-tokenizer")          # now loadable with AutoTokenizer
gpt2 = AutoTokenizer.from_pretrained("gpt2")           # 50,257-token byte-level BPE

samples = {
    "in-domain": "Once upon a time, a little dog named Max found a shiny red ball.",
    "code":      "def forward(self, x): return self.proj(torch.relu(self.fc(x)))",
    "german":    "Die Katze schläft auf dem warmen Fensterbrett in der Sonne.",
}
for name, s in samples.items():
    a, b = mine(s)["input_ids"], gpt2(s)["input_ids"]
    print(f"{name:10s} mine-8k={len(a):3d}  gpt2-50k={len(b):3d}  chars={len(s)}")
# Expect: similar counts in-domain, many more tokens for code and German with the 8k tokenizer.` },

        { t: 'h2', x: 'Vocabulary size: the trade-offs' },
        { t: 'p', x: 'The [[vocabulary]] size V is a hyperparameter with costs on both sides. A larger V compresses text into fewer tokens — shorter sequences, cheaper attention, more text per [[context-window]] — but grows the embedding and LM-head matrices (V·d each), makes the output softmax more expensive and leaves rare tokens under-trained.' },
        { t: 'table', head: ['Model', 'Vocab size', 'Embedding params at its d', 'Note'], rows: [
          ['GPT-2', '50,257', '50,257 × 768 ≈ 39M', '31% of the 124M model'],
          ['Llama 2', '32,000', '32,000 × 4,096 ≈ 131M (×2 untied)', 'SentencePiece BPE with byte fallback'],
          ['Llama 3', '128,256', '128,256 × 4,096 ≈ 525M (×2 untied)', 'Up to ~15% fewer tokens than Llama 2 on English; better on code and non-English'],
          ['Qwen2.5', '151,936 (matrix size)', '151,936 × 896 ≈ 136M at 0.5B', 'Over a quarter of the 0.5B model — tied'],
        ]},
        { t: 'p', x: '**Multilingual cost.** A tokenizer trained mostly on English splits other scripts into many more tokens. The same sentence can take 2–5× more tokens in Hindi, Thai or Amharic than in English with an English-centric vocabulary. That means higher API cost, slower generation, less effective context and usually worse quality. Large vocabularies trained on balanced multilingual data (Llama 3, Qwen, Gemma) narrow this gap — measure it on your own languages before choosing a base model.' },
        { t: 'callout', kind: 'tip', title: 'Byte-level fallback', x: 'Byte-level BPE starts from all 256 bytes, so any UTF-8 string — emoji, rare CJK characters, binary junk — can be encoded and decoded losslessly; there is no unknown token. SentencePiece tokenizers get the same guarantee with `byte_fallback=True` (Llama 2), emitting tokens like `<0xE2>` for characters missing from the vocabulary. Without fallback, unseen characters collapse to `<unk>` and the information is lost.' },

        { t: 'h2', x: 'Special tokens' },
        { t: 'p', x: '[[special-tokens|Special tokens]] are reserved ids that are never split or produced by merges. They carry structure, not text:' },
        { t: 'table', head: ['Token', 'Purpose', 'Examples'], rows: [
          ['BOS', 'Marks the start of a sequence', '`<s>` (Llama 2), `<|begin_of_text|>` (Llama 3)'],
          ['EOS', 'Marks the end of a document or turn; generation stops here', '`</s>`, `<|endoftext|>`, `<|eot_id|>` (Llama 3 turn end), `<|im_end|>` (Qwen)'],
          ['PAD', 'Filler to make a batch rectangular; masked out of attention and loss', '`<|pad|>`, often set to EOS when a model has none'],
          ['Role / chat markers', 'Delimit system / user / assistant turns in a [[chat-template]]', '`<|im_start|>`, `<|start_header_id|>`'],
          ['Tool / reasoning markers', 'Wrap tool calls or reasoning traces so they can be parsed', '`<tool_call>`, `<think>` in some recent models'],
        ]},
        { t: 'callout', kind: 'trap', title: 'pad = eos, then mask all pads → a model that never stops', x: [
          'Many base models have no pad token, so scripts set `tokenizer.pad_token = tokenizer.eos_token`. If your collator then sets **every** pad-id position to −100 in the labels, it also masks the genuine EOS at the end of each example.',
          'The model never learns to emit EOS and rambles until max_new_tokens. Fix: use a distinct pad token, or make sure the real end-of-turn token keeps its label.',
        ]},

        { t: 'h2', x: 'Adding tokens to an existing model' },
        { t: 'p', x: 'Domain jargon, new chat markers or tool-call tags sometimes deserve their own tokens. Adding them to the tokenizer is only half the job: the model’s embedding (and untied LM head) must grow to match, and the new rows start untrained, so you must fine-tune afterwards.' },
        { t: 'code', lang: 'python', title: 'Add tokens and resize embeddings', x: `from transformers import AutoTokenizer, AutoModelForCausalLM

name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
print(len(tok), model.get_input_embeddings().weight.shape)        # 49152 x 576

# special tokens are never split and can be skipped when decoding
n_added = tok.add_special_tokens({"additional_special_tokens": ["<tool_call>", "</tool_call>"]})
# ordinary tokens: domain words you want kept whole
n_added += tok.add_tokens(["NVLink", "TensorRT-LLM"])

# grow embeddings (and the tied LM head); pad the row count for Tensor Core-friendly shapes
model.resize_token_embeddings(len(tok), pad_to_multiple_of=64)
print(n_added, len(tok), model.get_input_embeddings().weight.shape)

print(tok.tokenize("Use NVLink <tool_call>"))   # new tokens appear whole
tok.save_pretrained("smollm2-ext"); model.save_pretrained("smollm2-ext")` },
        { t: 'callout', kind: 'warn', title: 'Before you resize', x: [
          'Some models already pad their embedding matrix beyond `len(tokenizer)` — Qwen2.5 has 151,936 rows for roughly 151.7k real tokens. Check both numbers first; you may not need to resize at all, and blindly resizing to `len(tok)` can shrink the matrix.',
          'New rows need training. Recent transformers versions initialise them near the mean of existing embeddings, but they still mean nothing until fine-tuned. With [[lora|LoRA]], also train the embedding and LM-head modules or the new tokens stay untrained.',
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Expect questions on vocabulary trade-offs (bigger V → shorter sequences but bigger embeddings/softmax), why multilingual text costs more tokens, and how byte-level BPE avoids OOV.',
          'Know the workflow for new tokens: add to tokenizer → `resize_token_embeddings` → fine-tune so the new embeddings learn something.',
          'Know the role of BOS/EOS/PAD and that the tokenizer must match the model it was trained with — swapping tokenizers breaks a model.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'BPE = repeatedly merge the most frequent adjacent pair; the merge list is the tokenizer.',
          'Byte-level BPE (or SentencePiece byte fallback) means no out-of-vocabulary failures.',
          'Vocab size trades sequence length against embedding size; non-English text often costs 2–5× more tokens.',
          'Special tokens carry structure; mishandling EOS/PAD produces models that never stop.',
          'Adding tokens requires resizing embeddings and then training them.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'curation',
      title: 'Pretraining data: cleaning, dedup, filtering and NeMo Curator',
      minutes: 24,
      outcomes: [
        'Lay out the stages of a web-scale pretraining data pipeline in order',
        'Apply heuristic and model-based quality filters and explain what each catches',
        'Implement exact and MinHash-LSH fuzzy deduplication and tune its threshold',
        'Redact PII and decontaminate training data against evaluation benchmarks',
        'Describe how NeMo Curator scales these steps on GPUs with RAPIDS',
      ],
      blocks: [
        { t: 'lede', x: 'Architecture differences between open models are small; data differences are large. FineWeb, Nemotron-CC, DCLM and similar projects showed that the same model trained on better-filtered tokens scores markedly higher on benchmarks. This lesson walks through the pipeline that turns raw web crawl into training tokens — and the tools, from a laptop script to [[nemo-curator|NeMo Curator]] on a GPU cluster, that run it.' },

        { t: 'h2', x: 'The pipeline' },
        { t: 'flow', steps: [
          { b: 'Acquire & extract', s: 'Download Common Crawl WARC files (or your corpora), extract main text from HTML (e.g. trafilatura, jusText), drop boilerplate.' },
          { b: 'Normalise', s: 'Fix broken Unicode (ftfy-style), normalise whitespace, drop empty or binary documents.' },
          { b: 'Language ID', s: 'A fastText classifier (lid.176 or similar) labels each document; keep target languages above a confidence threshold.' },
          { b: 'Heuristic filters', s: 'Cheap rules: length, symbol ratios, repeated lines/n-grams, stop-word presence, URL blocklists.' },
          { b: 'Deduplicate', s: 'Exact (hash of normalised text), then fuzzy (MinHash + LSH), optionally semantic (embedding clustering).' },
          { b: 'Model-based quality', s: 'Classifiers score "educational value" or similarity to high-quality references; keep the top slice.' },
          { b: 'PII & safety', s: 'Redact emails, phone numbers, IPs, keys; drop toxic or adult content with classifiers.' },
          { b: 'Decontaminate', s: 'Remove documents that overlap evaluation benchmarks (n-gram matching).' },
          { b: 'Mix, tokenise, shard', s: 'Blend sources by target weights, tokenise, write shards (e.g. Megatron .bin/.idx or parquet).' },
        ]},
        { t: 'p', x: 'Order matters for cost: run the cheapest filters first so expensive steps (fuzzy dedup, neural classifiers) see less data. It is common for 80–90% of raw crawl text to be discarded before training.' },

        { t: 'h2', x: 'Language ID and heuristic filtering' },
        { t: 'p', x: '[[language-identification|Language identification]] is a fast fastText model over character n-grams; it is cheap enough to run on billions of documents on CPUs. [[quality-filtering|Heuristic quality filters]] then remove obvious junk. The Gopher and C4 rule sets are the classic references; most pipelines use variations of them.' },
        { t: 'code', lang: 'python', title: 'Gopher-style heuristic filter on a streamed dataset', x: `from datasets import load_dataset

# Streaming iterates over remote shards without downloading the whole dataset.
ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

STOP = {"the", "be", "to", "of", "and", "that", "have", "with"}

def keep(ex):
    text = ex["text"]
    words = text.split()
    n = len(words)
    if not 50 <= n <= 100_000:                                    # too short / absurdly long
        return False
    mean_len = sum(len(w) for w in words) / n
    if not 3 <= mean_len <= 10:                                   # gibberish or token soup
        return False
    if sum(any(c.isalpha() for c in w) for w in words) / n < 0.8: # mostly numbers/symbols
        return False
    if len(STOP & {w.lower() for w in words}) < 2:               # not natural prose
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and sum(l.rstrip().endswith("...") for l in lines) / len(lines) > 0.3:
        return False                                              # truncated snippet lists
    return True

filtered = ds.filter(keep)                     # lazy: applied as you iterate
for ex in filtered.take(3):
    print(ex["text"][:150].replace("\\n", " "), "...")` },
        { t: 'p', x: '**Model-based filtering** catches what rules cannot: well-formed but useless text. Approaches include a fastText classifier trained to separate reference-quality text (e.g. pages cited by Wikipedia) from random crawl; [[perplexity|perplexity]] filtering with a small LM trained on clean data; and LLM-annotated classifiers such as FineWeb-Edu, where a large model scores a sample of pages for educational value and a small encoder is trained to reproduce those scores at scale. NVIDIA’s Nemotron-CC used an ensemble of such classifiers and kept more, not fewer, tokens by bucketing quality instead of applying one hard cut.' },
        { t: 'callout', kind: 'warn', title: 'Filters encode bias', x: 'Aggressive classifiers trained on "high-quality" references can disproportionately remove dialects, non-Western sources and minority-community text. Audit what each filter removes, not just how much — sample rejected documents by language, region and topic.' },

        { t: 'h2', x: 'Deduplication: exact and fuzzy' },
        { t: 'p', x: 'Web crawls are full of duplicates: mirrored pages, templated product listings, boilerplate licences. Duplicates waste compute, increase memorisation (and privacy risk) and skew the data mixture. [[deduplication|Deduplication]] runs at three levels:' },
        { t: 'table', head: ['Level', 'Method', 'Catches', 'Cost'], rows: [
          ['Exact', 'Hash (MD5/xxHash) of normalised text; drop repeats', 'Byte-identical copies', 'Very cheap'],
          ['Fuzzy', '[[minhash|MinHash]] signatures + [[lsh|LSH]] bucketing, then Jaccard check', 'Near-duplicates: changed dates, ads, headers', 'Moderate; the GPU-accelerated step in Curator'],
          ['Semantic', 'Embed documents, cluster, drop near neighbours (SemDeDup)', 'Paraphrases and same content in different words', 'Expensive (embedding model over the corpus)'],
        ]},
        { t: 'h3', x: 'How MinHash LSH works' },
        { t: 'ol', items: [
          'Represent each document as a set of shingles (e.g. word 5-grams). Similarity = Jaccard |A ∩ B| / |A ∪ B|.',
          'Apply k independent hash functions; for each, keep the **minimum** hash over the document’s shingles. The probability that two documents share a min-hash equals their Jaccard similarity, so a k-value signature estimates it.',
          'Split the signature into b bands of r rows. Documents that match exactly in **any** band land in the same bucket and become candidate pairs: P(candidate) = 1 − (1 − sʳ)ᵇ.',
          'Verify candidates (optionally with true Jaccard), group them into connected components, keep one document per group.',
        ]},
        { t: 'code', lang: 'text', title: 'Choosing bands and rows (k = 128 = 16 bands × 8 rows)', x: `threshold ≈ (1/b)^(1/r) = (1/16)^(1/8) ≈ 0.71

Jaccard s   P(candidate) = 1 − (1 − s^8)^16
0.5         ≈ 0.06     mostly ignored
0.7         ≈ 0.61     just above the threshold, still often missed
0.8         ≈ 0.95     almost always caught
0.9         ≈ 1.00

More rows per band -> stricter (fewer false positives); more bands -> more recall.` },
        { t: 'code', lang: 'python', title: 'Fuzzy dedup with datasketch (pip install datasketch)', x: `from datasketch import MinHash, MinHashLSH

docs = {
    "a": "The quick brown fox jumps over the lazy dog near the old river bank this morning",
    "b": "The quick brown fox jumps over the lazy dog near the old river bank this morning!",
    "c": "Tensor Cores accelerate matrix multiplications in reduced precision on NVIDIA GPUs",
}

def shingles(text, n=3):
    w = text.lower().split()
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}

def signature(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for s in shingles(text):
        m.update(s.encode("utf8"))
    return m

sigs = {k: signature(v) for k, v in docs.items()}
lsh = MinHashLSH(threshold=0.7, num_perm=128)      # datasketch picks bands/rows for you
for k, m in sigs.items():
    lsh.insert(k, m)

print(lsh.query(sigs["a"]))                 # ['a', 'b'] in some order -> near-duplicates
print(round(sigs["a"].jaccard(sigs["b"]), 2))  # estimated Jaccard, about 0.8` },

        { t: 'h2', x: 'PII redaction and decontamination' },
        { t: 'p', x: '[[pii-redaction|PII redaction]] replaces personal data with placeholders before training, because LLMs can memorise and regurgitate rare strings. Regexes handle structured PII (emails, phone numbers, IPs, credit cards, API keys); named-entity models (Microsoft Presidio and similar) handle names and addresses. Replace rather than delete, so sentence structure survives.' },
        { t: 'code', lang: 'python', title: 'Regex PII redaction (structured identifiers only)', x: `import re

PATTERNS = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")),
    ("IP",    re.compile(r"\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b")),
    ("PHONE", re.compile(r"\\+?\\d[\\d\\s().-]{7,}\\d")),
]

def redact(text):
    for label, pat in PATTERNS:          # IP before PHONE so dotted quads are not read as phones
        text = pat.sub(f"<{label}>", text)
    return text

print(redact("Mail jane.doe@example.com or call +1 (555) 010-7788 from 10.0.0.12"))
# Mail <EMAIL> or call <PHONE> from <IP>` },
        { t: 'p', x: '[[data-contamination|Data contamination]] means benchmark test items leaked into training data, inflating scores without real capability. Decontaminate by building the set of n-grams (13-grams were used for GPT-3) from every evaluation set you care about and dropping — or at least flagging — training documents that share any. Do it **before** training: you cannot remove memorised answers afterwards.' },
        { t: 'code', lang: 'python', title: 'n-gram decontamination', x: `def ngrams(text, n=13):
    w = text.lower().split()
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}

benchmark_items = ["...every question and answer from your eval sets..."]
bench = set().union(*(ngrams(q) for q in benchmark_items))

def is_contaminated(doc):
    return not ngrams(doc).isdisjoint(bench)

# training_docs = [d for d in training_docs if not is_contaminated(d)]` },

        { t: 'h2', x: 'Data mixing' },
        { t: 'p', x: 'The final corpus is a [[data-mixture|mixture]] of sources with chosen sampling weights — not simply everything concatenated. Typical ingredients: filtered web text (the bulk), code, maths, books/papers, encyclopedic text and multilingual data. High-quality sources are often **upsampled** (seen for several epochs) while web text is seen once. Many recent recipes also **anneal** at the end: the last few percent of training uses a higher share of the best data as the learning rate decays.' },
        { t: 'table', head: ['Source', 'Illustrative share', 'Why'], rows: [
          ['Filtered web', '50–80%', 'Breadth of knowledge and language'],
          ['Code', '10–20%', 'Programming ability; also helps structured reasoning'],
          ['Maths & science', '5–10%', 'Quantitative reasoning'],
          ['Books, papers, encyclopedic', '5–10%', 'Long-range coherence, factual density'],
          ['Multilingual', 'Varies by goal', 'Target languages; too little → poor tokens and quality'],
        ]},
        { t: 'p', x: 'Treat these as ballpark figures — published mixtures vary widely by model and goal. The reliable approach is empirical: train small proxy models on candidate mixtures and compare downstream evaluations before committing to the big run.' },

        { t: 'h2', x: 'NeMo Curator: the GPU-scale version' },
        { t: 'p', x: '[[nemo-curator|NeMo Curator]] is NVIDIA’s open-source library for exactly this pipeline. It runs the stages across a cluster and moves the heavy ones onto GPUs with [[rapids|RAPIDS]] (cuDF dataframes, cuML/cuGraph) — fuzzy dedup’s [[minhash|MinHash]], LSH and connected-components steps in particular see large speed-ups over CPU implementations. It also ships GPU classifiers (quality, domain, educational value, safety), PII redaction, language ID, decontamination tasks and synthetic-data generation helpers, and writes output that feeds [[nemo]] / [[megatron]] training.' },
        { t: 'specs', rows: [
          ['Inputs', 'JSONL or Parquet documents, or Common Crawl / Wikipedia / arXiv download-and-extract helpers'],
          ['Filtering', 'Dozens of heuristic filters (word count, symbol ratio, repetition, URL), fastText language ID and quality, GPU classifier models'],
          ['Dedup', 'Exact (hash), fuzzy (GPU MinHash + LSH + connected components), semantic (embeddings + clustering)'],
          ['Privacy & safety', 'PII detection/redaction modifiers; safety and content classifiers'],
          ['Contamination', 'n-gram based task decontamination against common benchmarks'],
          ['Scale-out', 'Distributed across many GPUs and nodes; RAPIDS for GPU dataframes'],
        ]},
        { t: 'code', lang: 'python', title: 'NeMo Curator heuristic pass — classic DocumentDataset API (illustrative)', x: `# Classic NeMo Curator API (Dask + cuDF). Newer releases re-architected the library around
# pipeline/stage objects; module paths differ but the stages are the same. Check the docs for
# the version you install before copying this.
from nemo_curator import Sequential, ScoreFilter, Modify
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import WordCountFilter, NonAlphaNumericFilter, RepeatingTopNGramsFilter
from nemo_curator.modifiers import UnicodeReformatter
from nemo_curator.utils.distributed_utils import get_client

client = get_client(cluster_type="gpu")                         # one Dask worker per GPU
docs = DocumentDataset.read_json("raw/*.jsonl", backend="cudf") # documents live in GPU memory

pipeline = Sequential([
    Modify(UnicodeReformatter()),                                # fix mojibake
    ScoreFilter(WordCountFilter(min_words=50, max_words=100_000), text_field="text"),
    ScoreFilter(NonAlphaNumericFilter(), text_field="text"),     # symbol-heavy junk
    ScoreFilter(RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2), text_field="text"),
])
clean = pipeline(docs)
clean.to_json("curated/", write_to_filename=False)
# Next stages: fuzzy dedup (GPU MinHash/LSH), classifiers, PII redaction, decontamination.` },

        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Know the stages and their purpose: language ID, heuristic vs model-based filtering, exact vs fuzzy (MinHash LSH) vs semantic dedup, PII redaction, decontamination, mixing.',
          'Know that NeMo Curator accelerates curation on GPUs with RAPIDS (cuDF, etc.), and that fuzzy dedup is the step that benefits most.',
          'Contamination questions: suspiciously high benchmark scores after training on web data → check n-gram overlap with the test set.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Pipeline: extract → normalise → language ID → heuristics → dedup → quality models → PII/safety → decontaminate → mix and tokenise.',
          'Run cheap filters first; expect to discard most raw crawl.',
          'MinHash estimates Jaccard similarity; LSH bands find candidates with P = 1 − (1 − sʳ)ᵇ.',
          'Redact PII and decontaminate before training — neither can be undone afterwards.',
          'NeMo Curator runs the same stages on GPUs with RAPIDS, at cluster scale.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'sft-datasets',
      title: 'Instruction data: chat templates, packing and loss masking',
      minutes: 22,
      outcomes: [
        'Convert Alpaca, ShareGPT and messages-format data into one schema',
        'Render conversations with apply_chat_template and add a generation prompt correctly',
        'Mask prompt tokens out of the loss with labels = -100',
        'Choose between padding and packing and avoid cross-example contamination',
        'Plan synthetic data generation, quality filtering and a clean train/val split',
      ],
      blocks: [
        { t: 'lede', x: 'Pretraining teaches a model language; [[sft|supervised fine-tuning]] teaches it to follow instructions in a conversation format. The training loop is the same next-token objective you already know — what changes is the data: how conversations are formatted into tokens, which tokens count toward the loss, and how examples are batched. Each of those has a classic bug, and this lesson shows all three.' },

        { t: 'h2', x: 'Instruction data formats' },
        { t: 'table', head: ['Format', 'Shape of one example', 'Where you meet it'], rows: [
          ['Alpaca', '`{instruction, input, output}` — single turn', 'Early instruction datasets (Stanford Alpaca, many clones)'],
          ['ShareGPT', '`{conversations: [{from: "human"|"gpt", value}]}` — multi-turn', 'Community chat datasets'],
          ['Messages (OpenAI-style)', '`{messages: [{role: "system"|"user"|"assistant", content}]}`', 'The de-facto standard; native to TRL and chat templates'],
          ['Prompt / completion', '`{prompt, completion}`', 'Simple single-turn SFT; completion-only loss'],
          ['Preference', '`{prompt, chosen, rejected}`', 'DPO-style alignment (Module 4)'],
        ]},
        { t: 'code', lang: 'python', title: 'Normalise everything to the messages format', x: `def from_alpaca(ex):
    user = ex["instruction"] if not ex.get("input") else f"{ex['instruction']}\\n\\n{ex['input']}"
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": ex["output"]}]}

ROLE = {"system": "system", "human": "user", "gpt": "assistant"}
def from_sharegpt(ex):
    return {"messages": [{"role": ROLE[t["from"]], "content": t["value"]} for t in ex["conversations"]]}

# with datasets: ds = ds.map(from_alpaca, remove_columns=ds.column_names)` },

        { t: 'h2', x: 'Chat templates' },
        { t: 'p', x: 'A [[chat-template]] is a Jinja template, stored in the tokenizer config, that turns a list of messages into the exact string — role markers, special tokens, newlines — the model was trained on. Each model family has its own. `apply_chat_template` renders it for you; never hand-write the format.' },
        { t: 'code', lang: 'python', title: 'apply_chat_template with Qwen2.5', x: `from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What does GQA reduce?"},
    {"role": "assistant", "content": "The KV-cache size."},
]

# Training: render the whole conversation, including the assistant answer
print(tok.apply_chat_template(messages, tokenize=False))

# Inference: render up to the user turn and append the assistant header
prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
print(prompt[-40:])     # ends with "<|im_start|>assistant\\n" - the model writes from here` },
        { t: 'code', lang: 'text', title: 'Rendered training string (ChatML-style template)', x: `<|im_start|>system
You are a concise assistant.<|im_end|>
<|im_start|>user
What does GQA reduce?<|im_end|>
<|im_start|>assistant
The KV-cache size.<|im_end|>` },
        { t: 'callout', kind: 'trap', title: 'Template mismatches fail silently', x: [
          'Fine-tuning with one template and serving with another (or with a hand-rolled prompt) gives a model that rambles, ignores the system prompt or never emits its stop token — with no error anywhere.',
          'Forgetting `add_generation_prompt=True` at inference makes the model continue the **user** turn. Base models may have no template at all: when you fine-tune one, pick a template, save it with the tokenizer, and use the same one in serving (vLLM, NIM and Triton read it from the tokenizer config).',
        ]},

        { t: 'h2', x: 'Loss masking: train on answers, not prompts' },
        { t: 'p', x: 'In SFT you usually want the model to learn to **produce** assistant responses, not to predict the user’s questions or the system prompt. [[loss-masking|Loss masking]] sets the label of every non-assistant token to −100, the `ignore_index` of PyTorch’s cross-entropy, so those positions contribute nothing to the loss or gradients. The tokens are still in `input_ids` — the model reads them as context.' },
        { t: 'code', lang: 'python', title: 'Build input_ids and masked labels for one conversation', x: `from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def build_example(messages, max_len=1024):
    # everything before the final assistant answer, ending with the assistant header
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    assert full.startswith(prompt), "template is not prefix-stable; mask turn by turn instead"

    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]   # loss only on the answer
    full_ids, labels = full_ids[:max_len], labels[:max_len]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

ex = build_example([
    {"role": "user", "content": "Name one NVIDIA inference library."},
    {"role": "assistant", "content": "TensorRT-LLM."},
])
trained = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
print(repr(tok.decode(trained)))     # 'TensorRT-LLM.<|im_end|>\\n' - answer + end-of-turn token` },
        { t: 'p', x: 'Note that the end-of-turn token (`<|im_end|>`) keeps its label: that is how the model learns when to stop. For multi-turn data, mask every user/system turn and train on every assistant turn. Recent TRL releases do this for you with `assistant_only_loss=True` (it needs a chat template that marks assistant spans) or `completion_only_loss` for prompt/completion datasets; Module 4 uses them. Older tutorials use `DataCollatorForCompletionOnlyLM`, which is deprecated in recent TRL releases.' },
        { t: 'callout', kind: 'tip', title: 'Masking is a choice', x: 'Loss on prompts too ("train on inputs") is sometimes fine for small, short-prompt datasets and is the default in some frameworks. Masking usually helps when prompts are long (RAG context, documents to summarise) — otherwise most of the gradient goes into predicting text the model will never have to write.' },

        { t: 'h2', x: 'Padding vs packing' },
        { t: 'p', x: 'Batches must be rectangular. **Padding** fills short sequences up to the longest in the batch with pad tokens (masked out of attention and loss). It is simple but wasteful when lengths vary: a batch containing one 2,048-token example and fifteen 300-token examples is ~85% padding. **[[packing|Packing]]** concatenates several examples into each fixed-length row, so almost every position is a real token.' },
        { t: 'vs', cards: [
          { h: 'Padding', items: [
            'One example per row; pad to the batch’s longest',
            'No cross-example interference',
            'Wastes compute when lengths vary (sort/bucket by length to reduce it)',
            'Simplest; fine for evaluation and small runs',
          ]},
          { h: 'Packing', items: [
            'Several examples per row, separated by EOS',
            'Near-100% token utilisation → often 2–5× faster SFT',
            'Risk: examples attend to each other unless boundaries are respected',
            'Fix: reset position_ids per example + varlen attention (padding-free)',
          ]},
        ]},
        { t: 'code', lang: 'python', title: 'Greedy packing that keeps example boundaries', x: `def pack(examples, max_len=2048):
    """examples: dicts with input_ids/labels (each already <= max_len)."""
    rows, cur = [], {"input_ids": [], "labels": [], "position_ids": []}
    for ex in examples:
        n = len(ex["input_ids"])
        if cur["input_ids"] and len(cur["input_ids"]) + n > max_len:
            rows.append(cur)
            cur = {"input_ids": [], "labels": [], "position_ids": []}
        cur["input_ids"] += ex["input_ids"]
        cur["labels"] += ex["labels"]
        cur["position_ids"] += list(range(n))   # restart at 0: marks where each example begins
    if cur["input_ids"]:
        rows.append(cur)
    return rows

# With a varlen attention backend (e.g. attn_implementation="flash_attention_2"), transformers can
# use the restarting position_ids to stop packed examples attending to each other.
# transformers' DataCollatorWithFlattening and TRL's padding-free packing build on the same idea.` },
        { t: 'callout', kind: 'warn', title: 'Packing changes the loss weighting', x: 'With packing, each row’s loss averages over all its answer tokens, so long examples weigh more than in padded per-example training. That is usually acceptable, but if a run behaves differently after you switch packing on, this (and cross-contamination) is where to look.' },

        { t: 'h2', x: 'Synthetic data, quality and splits' },
        { t: 'p', x: '[[synthetic-data|Synthetic data]] — instructions and answers generated by a stronger model — now makes up a large share of most SFT and alignment corpora. Common patterns: Self-Instruct (bootstrap new tasks from seed tasks), Evol-Instruct (rewrite prompts to be harder), persona-driven generation for diversity, and generate-then-filter with a [[reward-model|reward model]] or LLM judge. NVIDIA released Nemotron-4 340B Instruct and Reward specifically to support permissively licensed synthetic-data pipelines, and [[nemo-curator|NeMo Curator]] includes synthetic-data generation helpers.' },
        { t: 'flow', steps: [
          { b: 'Seed', s: 'Collect real prompts or topics from your domain (support tickets, docs, FAQ).' },
          { b: 'Generate', s: 'A strong model writes diverse instructions and candidate answers (vary personas, difficulty, format).' },
          { b: 'Filter', s: 'Score with a reward model / LLM judge; verify code by running it and maths by checking answers; drop refusals and near-duplicates.' },
          { b: 'Decontaminate', s: 'Remove anything overlapping your evaluation sets.' },
          { b: 'Review', s: 'Humans spot-check a sample; fix systematic problems in the generator prompt, not by hand.' },
        ]},
        { t: 'callout', kind: 'warn', title: 'Licences and model collapse', x: 'Check the terms of the model that generated your data — some licences restrict using outputs to train other models. And do not train repeatedly on a model’s own unfiltered outputs: diversity shrinks and errors compound. Filter hard and keep real human data in the mix.' },
        { t: 'p', x: '**Quality beats quantity.** The LIMA study fine-tuned a 65B model on just 1,000 carefully curated examples and got strong chat behaviour; many teams see the same pattern at small scale. A few thousand clean, diverse, correctly formatted examples generally outperform a hundred thousand noisy ones. Deduplicate prompts, balance task types, filter wrong or truncated answers, and keep response style consistent.' },
        { t: 'p', x: '**Hold out a validation set** (typically 2–5%, at least a few hundred examples) before any training. Split by source document or conversation rather than by row, so paraphrases of the same item do not land on both sides, and deduplicate across the split. Watch validation loss during SFT: it usually bottoms out after 1–3 epochs, and rising validation loss is your signal to stop.' },

        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Labels of −100 are ignored by cross-entropy: that is how prompt tokens are excluded from the loss.',
          'Packing improves GPU utilisation for variable-length data; know the cross-contamination caveat and its fix.',
          'Chat templates must match between fine-tuning and inference; `add_generation_prompt=True` at inference.',
          'Small, high-quality, deduplicated and decontaminated data usually beats large noisy data for SFT.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Normalise all instruction data to the messages format and render it with the model’s own chat template.',
          'Mask prompt and system tokens with −100; keep the end-of-turn token trainable.',
          'Pad for simplicity, pack for throughput — and respect example boundaries when packing.',
          'Synthetic data works when it is filtered, verified, decontaminated and licensed for the purpose.',
          'Split train/val by source before training and stop when validation loss turns up.',
        ]},
      ],
    },
  ],

  /* -------------------------------------------------------------------- */
  quiz: [
    { id: 'd01', domain: 'data',
      scenario: 'A team fine-tunes a base model whose tokenizer has no pad token. Their script sets `tokenizer.pad_token = tokenizer.eos_token`, and the collator replaces every pad-id position in the labels with −100. After training, the model answers correctly but keeps generating until max_new_tokens.',
      q: 'What is the most likely cause?',
      opts: [
        'The learning rate was too high, so the model forgot how to stop',
        'The chat template was not applied at inference',
        'The genuine EOS tokens were masked along with padding, so the model never learned to emit EOS',
        'The vocabulary is too small to contain an EOS token',
      ],
      a: 2,
      why: 'Because pad and EOS share an id, masking all pad positions also masks the real end-of-sequence token, so it never receives a training signal. Use a distinct [[special-tokens|pad token]] or keep the true EOS label.',
      distract: 'A high LR causes broader damage, not this specific symptom. A missing template causes format problems but is not indicated here. Every tokenizer has an EOS token; vocabulary size is irrelevant.' },

    { id: 'd02', domain: 'data',
      q: 'Which statement about increasing a tokenizer’s vocabulary from 32k to 128k is correct?',
      opts: [
        'Sequences get shorter for the same text, but the embedding and output layers grow by about 4×',
        'Sequences get longer, but the model gets smaller',
        'Nothing changes except the tokenizer file size',
        'The model no longer needs special tokens',
      ],
      a: 0,
      why: 'A bigger [[vocabulary]] encodes text in fewer tokens (cheaper attention, more text per context), but the V×d embedding and LM-head matrices grow 4× and the softmax gets costlier.',
      distract: 'Larger vocabularies shorten sequences and enlarge the model. The change reaches the model’s parameters, not just the file. Special tokens are still required.' },

    { id: 'd03', domain: 'data',
      scenario: 'A pretraining corpus of 2 billion web documents contains many near-identical pages that differ only in dates, navigation menus and ad blocks.',
      q: 'Which technique is designed to find these efficiently?',
      opts: [
        'Exact deduplication with MD5 hashes of the raw HTML',
        'Language identification with fastText',
        'Perplexity filtering with a small reference model',
        'MinHash signatures with locality-sensitive hashing over shingles',
      ],
      a: 3,
      why: '[[minhash|MinHash]] estimates Jaccard similarity of shingle sets and [[lsh|LSH]] bucketing finds candidate pairs without comparing all pairs — the standard fuzzy dedup method (GPU-accelerated in NeMo Curator).',
      distract: 'Exact hashes miss documents that differ by a single byte. Language ID detects language, not duplicates. Perplexity filtering scores quality and would keep every copy of a fluent page.' },

    { id: 'd04', domain: 'data',
      q: 'In a Hugging Face causal-LM training batch, what does a label value of −100 do?',
      opts: [
        'Marks the token as padding in the attention mask',
        'Excludes that position from the cross-entropy loss and its gradients',
        'Tells the tokenizer to skip the token when decoding',
        'Signals the end of a packed example',
      ],
      a: 1,
      why: '−100 is the `ignore_index` of PyTorch’s cross-entropy. [[loss-masking|Loss masking]] uses it so prompt, system or pad positions do not contribute to the loss while still being visible as context.',
      distract: 'The attention mask is a separate tensor. Decoding skips special tokens via `skip_special_tokens`, not labels. Packed example boundaries are marked by EOS tokens and position ids.' },

    { id: 'd05', domain: 'data',
      scenario: 'After continued pretraining on a fresh web crawl, a model’s score on a popular reasoning benchmark jumps from 41% to 78%, while scores on a private held-out set of similar questions barely move.',
      q: 'What should the team check first?',
      opts: [
        'Whether benchmark questions and answers leaked into the crawl — n-gram overlap with the test set',
        'Whether the learning-rate schedule had enough warmup',
        'Whether the tokenizer vocabulary was large enough for the benchmark',
        'Whether packing was enabled during training',
      ],
      a: 0,
      why: 'A big public-benchmark gain without a matching private-set gain is the signature of [[data-contamination]]. Measure n-gram (e.g. 13-gram) overlap between training data and the benchmark and decontaminate.',
      distract: 'Warmup, vocabulary size and packing affect training generally; none explains a gain confined to one public benchmark.' },

    { id: 'd06', domain: 'data',
      q: 'Which description of NeMo Curator is accurate?',
      opts: [
        'A model-serving runtime that caches curated prompts',
        'A labelling UI for human annotators',
        'A GPU-accelerated, scalable data-curation library (using RAPIDS) for filtering, deduplication, PII redaction, classification and synthetic data',
        'A tokenizer-training tool that replaces SentencePiece',
      ],
      a: 2,
      why: '[[nemo-curator|NeMo Curator]] runs curation pipelines across GPUs and nodes; [[rapids|RAPIDS]] accelerates dataframe work and fuzzy deduplication.',
      distract: 'Serving is NIM/Triton territory. It is not an annotation tool. Tokenizer training is done by tokenizers/SentencePiece.' },

    { id: 'd07', domain: 'data',
      scenario: 'An SFT dataset contains 200k examples whose lengths range from 40 to 2,048 tokens (median 280). Training with max length 2,048 and plain padding shows low GPU utilisation.',
      q: 'Which change most improves throughput while preserving correctness?',
      opts: [
        'Raise the max length to 8,192 so fewer examples are truncated',
        'Pack multiple examples per sequence, resetting position ids so examples cannot attend to each other',
        'Remove the attention mask so pad tokens are processed like real tokens',
        'Convert the dataset from messages to Alpaca format',
      ],
      a: 1,
      why: 'With a median of 280 tokens, padding to long rows wastes most compute. [[packing|Packing]] with boundary-aware (padding-free / varlen) attention fills rows with real tokens without cross-example leakage.',
      distract: 'A longer max length adds even more padding. Removing the mask makes the model attend to and learn from padding. Changing the data format does not change sequence lengths.' },

    { id: 'd08', domain: 'data',
      q: 'A fine-tuned chat model works in the team’s test script but produces rambling, oddly formatted answers when served with a hand-written prompt string. What is the most likely fix?',
      opts: [
        'Retrain with a larger vocabulary',
        'Increase the temperature at inference',
        'Add more synthetic data and retrain',
        'Format serving requests with the same chat template used in training, including the generation prompt',
      ],
      a: 3,
      why: 'A [[chat-template]] mismatch between training and serving is a classic silent failure. Use `apply_chat_template(..., add_generation_prompt=True)` or a server that reads the template from the tokenizer config.',
      distract: 'Vocabulary size and extra data do not address a formatting mismatch. A higher temperature makes output more random, not better formatted.' },
  ],
}
