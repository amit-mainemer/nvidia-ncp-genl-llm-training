// End-of-lesson comprehension checks for Module 2: Data Preparation. Keyed by lesson id.
export default {
  tokenization: [
    {
      q: 'During BPE training, what happens at each iteration?',
      opts: [
        'The least frequent token is removed from the vocabulary',
        'Words are split at random positions and the best split is kept',
        'A neural network predicts the next merge',
        'The most frequent adjacent symbol pair is merged into a new symbol and the merge is recorded',
      ],
      a: 3,
      why: '[[bpe|BPE]] greedily merges the most frequent adjacent pair until the target vocabulary size is reached; encoding replays those merges in order.',
      distract: 'Removing low-value tokens is how Unigram LM prunes. Random splitting is not a training algorithm. No neural network is involved in BPE.',
    },
    {
      q: 'Why can a byte-level BPE tokenizer encode any input string without an unknown token?',
      opts: [
        'Its vocabulary contains every word in every language',
        'Its base alphabet is all 256 byte values, so any UTF-8 text can fall back to byte tokens',
        'It replaces unknown characters with the closest known character',
        'It drops characters it does not recognise',
      ],
      a: 1,
      why: 'Starting from the 256 bytes guarantees coverage: anything not covered by a merge is still representable as bytes, and decoding is lossless.',
      distract: 'No vocabulary contains every word. Substituting or dropping characters would lose information — byte-level BPE does neither.',
    },
    {
      q: 'You add 20 new tokens to a model’s tokenizer. What must you do before the new tokens are useful?',
      opts: [
        'Call `resize_token_embeddings` on the model and then fine-tune so the new embedding rows are trained',
        'Nothing; the model learns new tokens automatically at inference',
        'Retrain the tokenizer from scratch on the original corpus',
        'Lower the temperature so the new tokens are sampled more often',
      ],
      a: 0,
      why: 'The embedding (and untied LM head) must grow to match `len(tokenizer)`, and the new rows start with no meaning, so they need training data that uses them.',
      distract: 'Models do not learn at inference. Retraining the tokenizer would change all ids and break the model. Temperature cannot give untrained embeddings meaning.',
    },
    {
      q: 'An English-centric tokenizer encodes the same sentence in English as 20 tokens and in Thai as 85 tokens. What is a direct consequence for Thai users?',
      opts: [
        'Better quality, because more tokens means more detail',
        'No effect, because cost is charged per character',
        'Higher cost and latency, and less effective context per request',
        'The model automatically switches to a Thai tokenizer',
      ],
      a: 2,
      why: 'More tokens per sentence means more compute per request, slower generation, higher per-token prices and fewer words fitting in the [[context-window]].',
      distract: 'Over-segmentation usually hurts quality. LLM cost is per token, not per character. Tokenizers are fixed per model; nothing switches automatically.',
    },
    {
      q: 'Which statement about special tokens is correct?',
      opts: [
        'They are reserved ids for structure such as BOS, EOS, PAD and chat-role markers, and are never split by merges',
        'They are the most frequent words in the training corpus',
        'They are added automatically for every new domain word',
        'They are ignored by the model during training',
      ],
      a: 0,
      why: '[[special-tokens|Special tokens]] mark sequence boundaries, padding and conversation structure; they are matched whole before normal tokenisation.',
      distract: 'Frequency has nothing to do with being special. Domain words are added as normal tokens if at all. Special tokens such as EOS must be learned — that is how a model knows when to stop.',
    },
  ],

  curation: [
    {
      q: 'Why are cheap heuristic filters usually run before fuzzy deduplication and neural quality classifiers?',
      opts: [
        'Heuristic filters only work on raw HTML',
        'Fuzzy dedup requires documents to be sorted by length first',
        'To shrink the data before the expensive stages, reducing total compute',
        'Neural classifiers cannot process documents that contain symbols',
      ],
      a: 2,
      why: 'Pipeline order is a cost decision: discard obvious junk with fast rules so the expensive steps see less data.',
      distract: 'Heuristics run on extracted text. Dedup has no sorting requirement. Classifiers handle symbols fine — they are just slower per document.',
    },
    {
      q: 'MinHash LSH uses 20 bands of 5 rows. Roughly what is the probability that two documents with Jaccard similarity 0.5 become a candidate pair?',
      opts: [
        'About 1%',
        'Exactly 50%',
        'About 99%',
        'About 47%',
      ],
      a: 3,
      why: 'P = 1 − (1 − s^r)^b = 1 − (1 − 0.5⁵)²⁰ = 1 − (1 − 0.03125)²⁰ ≈ 1 − 0.53 ≈ 0.47. The threshold (1/b)^(1/r) ≈ 0.55 sits near 0.5, so this is close to a coin flip.',
      distract: '1% and 99% ignore the band structure. 50% assumes the probability equals the similarity, which is true for a single min-hash, not for banded LSH.',
    },
    {
      q: 'What does decontamination of a pretraining corpus remove?',
      opts: [
        'Toxic and hateful documents',
        'Documents that overlap with evaluation benchmarks, so scores reflect real capability',
        'Personal information such as emails and phone numbers',
        'Documents in languages other than the target language',
      ],
      a: 1,
      why: '[[data-contamination|Decontamination]] removes text that matches benchmark items (often via 13-gram overlap) so evaluation is not inflated by memorisation.',
      distract: 'Toxic content is removed by safety filters, personal data by [[pii-redaction|PII redaction]], other languages by language ID.',
    },
    {
      q: 'Which stage benefits most from NeMo Curator’s RAPIDS-based GPU acceleration?',
      opts: [
        'Fuzzy deduplication — MinHash computation, LSH bucketing and connected components',
        'Downloading WARC files from Common Crawl',
        'Writing the final shards to disk',
        'Reading the licence of each source',
      ],
      a: 0,
      why: 'Fuzzy dedup is compute- and shuffle-heavy over billions of documents; [[nemo-curator|NeMo Curator]] runs it on GPUs with cuDF/cuGraph for large speed-ups.',
      distract: 'Downloading and writing are I/O-bound. Licence review is a governance task, not a compute stage.',
    },
    {
      q: 'Why is model-based quality filtering used in addition to heuristic rules?',
      opts: [
        'It is cheaper than heuristic rules',
        'It replaces the need for deduplication',
        'It guarantees that no bias is introduced',
        'It catches well-formed but low-value text that passes simple rules, such as fluent spam or boilerplate prose',
      ],
      a: 3,
      why: '[[quality-filtering|Model-based filters]] (fastText classifiers, LLM-annotated scorers like FineWeb-Edu) judge content value, not just surface statistics.',
      distract: 'Classifiers are more expensive than rules. They do not remove duplicates. They can introduce bias, which is why rejected samples should be audited.',
    },
  ],

  'sft-datasets': [
    {
      q: 'At inference, why do you call `apply_chat_template(messages, add_generation_prompt=True)`?',
      opts: [
        'To add a system prompt automatically',
        'To append the assistant-turn header so the model starts writing the assistant’s reply',
        'To tokenise the messages with a larger vocabulary',
        'To mask the user turn from the loss',
      ],
      a: 1,
      why: 'The generation prompt adds the opening marker of an assistant turn (e.g. `<|im_start|>assistant`). Without it, the model may continue the user’s message instead of answering.',
      distract: 'System prompts come from the messages you pass. The vocabulary is fixed. Loss masking is a training concern, not an inference one.',
    },
    {
      q: 'In an SFT example, which tokens should normally keep their labels (not −100)?',
      opts: [
        'The assistant’s response tokens, including its end-of-turn token',
        'The system prompt only',
        'The user’s question only',
        'Every token, including padding',
      ],
      a: 0,
      why: 'You train the model to produce assistant output — and to stop, which is why the end-of-turn token stays trainable. Everything else is masked by [[loss-masking]].',
      distract: 'System and user tokens are context, not targets. Padding must always be ignored.',
    },
    {
      q: 'What is the main risk of naive packing, and how is it addressed?',
      opts: [
        'Packed examples are truncated; fix by raising the learning rate',
        'Packing removes EOS tokens; fix by switching to Alpaca format',
        'Tokens from one example can attend to another; fix by resetting position ids per example with varlen (padding-free) attention',
        'Packing doubles memory use; fix by disabling the attention mask',
      ],
      a: 2,
      why: 'Concatenated examples share one causal attention window, so later examples can see earlier unrelated ones. Boundary-aware [[packing]] stops attention at example boundaries.',
      distract: 'Packing does not inherently truncate or remove EOS (EOS separates examples), and it does not double memory. The LR and data format are unrelated.',
    },
    {
      q: 'A team has 300k noisy auto-scraped instruction pairs and 4k carefully reviewed ones. With a fixed compute budget, what does experience (e.g. LIMA) suggest?',
      opts: [
        'Always use all 304k, because more data is always better',
        'Use only the noisy set, because reviewed data is biased',
        'Mix them 50/50 by upsampling the noisy set',
        'Start from the 4k high-quality set and add noisy data only if it is filtered and shown to help on validation',
      ],
      a: 3,
      why: 'For SFT, quality and diversity usually beat volume; a small curated set is a strong baseline and noisy data must earn its place via filtering and validation results.',
      distract: 'More noisy data often hurts. Reviewed data is not more biased by default. Upsampling noise amplifies its errors.',
    },
    {
      q: 'Which validation-split practice is most important for an SFT dataset built from support conversations?',
      opts: [
        'Use the last 2% of rows in file order',
        'Split by conversation or source so paraphrases of the same case do not appear on both sides, and deduplicate across the split',
        'Use the training set as validation to save data',
        'Make the validation set 50% of the data',
      ],
      a: 1,
      why: 'Leakage between train and validation hides [[overfitting]]. Splitting by source/conversation and deduplicating across the split keeps the validation loss honest.',
      distract: 'File order can cluster topics or dates. Reusing training data gives no signal. 50% wastes most of your training data.',
    },
  ],
}
