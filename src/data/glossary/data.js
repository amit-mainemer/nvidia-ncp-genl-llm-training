/**
 * Module 2 glossary — Data Preparation (tokenization, curation, SFT data).
 * Shared terms (token, tokenizer, pretraining, sft, nemo …) live in core.js.
 */
export default [
  /* ---------------- Tokenization ---------------- */
  { id: 'bpe', term: 'BPE (byte-pair encoding)', cat: 'Data & tokenization',
    def: 'A subword tokenization algorithm that starts from single characters or bytes and repeatedly merges the most frequent adjacent pair into a new token until a target vocabulary size is reached. The ordered merge list defines the tokenizer. Used by GPT-2/4, Llama, Qwen and Mistral.',
    exam: 'Be able to trace a few merges by hand and explain that encoding replays merges in learned order.' },

  { id: 'byte-level-bpe', term: 'Byte-level BPE', cat: 'Data & tokenization',
    def: 'BPE whose base alphabet is the 256 byte values of UTF-8, so every possible string can be encoded and decoded losslessly with no unknown token. GPT-2 introduced it; byte-level pre-tokenisers mark a leading space with Ġ.',
    exam: 'The answer to "how does a tokenizer avoid out-of-vocabulary failures?" — byte-level BPE or SentencePiece byte fallback.' },

  { id: 'sentencepiece', term: 'SentencePiece', cat: 'Data & tokenization',
    def: 'A Google library that trains BPE or Unigram tokenizers directly on raw text, encoding spaces as ▁ so no language-specific pre-tokenisation is needed. Supports `byte_fallback` to emit byte tokens like <0xE2> for unseen characters. Used by Llama 2, T5 and many multilingual models.',
    exam: 'Language-agnostic (good for scripts without spaces such as Chinese and Japanese); byte fallback prevents <unk>.' },

  { id: 'wordpiece', term: 'WordPiece', cat: 'Data & tokenization',
    def: 'The subword algorithm used by BERT: like BPE, but merges are chosen by how much they increase training-data likelihood rather than by raw frequency. Continuation pieces are prefixed with ##.',
    exam: 'Associate WordPiece with BERT-style encoders and BPE with GPT-style decoders.' },

  { id: 'vocabulary', term: 'Vocabulary (vocab size)', cat: 'Data & tokenization',
    def: 'The fixed set of tokens a tokenizer can emit, typically 32k–256k for modern LLMs. Vocab size V sets the embedding and LM-head shapes (V × d) and the cost of the output softmax.',
    exam: 'Bigger V → shorter sequences but more embedding parameters and rare-token under-training; smaller V → longer sequences. Multilingual coverage needs a larger, balanced vocabulary.' },

  { id: 'special-tokens', term: 'Special tokens', cat: 'Data & tokenization',
    def: 'Reserved tokens that carry structure rather than text — BOS, EOS, PAD, UNK, chat role markers (<|im_start|>, <|eot_id|>), tool-call tags. They are matched whole and never produced by merges.',
    exam: 'EOS handling is a classic bug: if pad = eos and all pads are masked from the loss, the model never learns to stop.' },

  { id: 'chat-template', term: 'Chat template', cat: 'Data & tokenization',
    def: 'A Jinja template stored with a tokenizer that renders a list of {role, content} messages into the exact token string a chat model was trained on. Applied with `tokenizer.apply_chat_template(...)`; `add_generation_prompt=True` appends the assistant header for inference.',
    exam: 'Use the same template in fine-tuning and serving; a mismatch silently degrades output and stop behaviour.' },

  /* ---------------- Curation ---------------- */
  { id: 'language-identification', term: 'Language identification', cat: 'Data & tokenization',
    def: 'Classifying each document’s language, usually with a fastText model over character n-grams (e.g. lid.176), and keeping only target languages above a confidence threshold. An early, cheap stage of pretraining curation.',
    exam: 'Runs early in the pipeline because it is cheap and removes large volumes of off-target text.' },

  { id: 'quality-filtering', term: 'Quality filtering', cat: 'Data & tokenization',
    def: 'Removing low-value documents from a corpus. Heuristic filters use rules (word counts, symbol and repetition ratios, stop-word presence — e.g. Gopher/C4 rules); model-based filters use classifiers or perplexity scores (e.g. fastText reference classifiers, FineWeb-Edu-style educational-value scorers).',
    exam: 'Heuristics first (cheap), classifiers later (expensive, catch fluent junk). Audit removed data for bias.' },

  { id: 'deduplication', term: 'Deduplication', cat: 'Data & tokenization',
    def: 'Removing repeated content from training data at three levels: exact (hash of normalised text), fuzzy (MinHash + LSH near-duplicate detection) and semantic (embedding similarity). Reduces wasted compute, memorisation and privacy leakage.',
    exam: 'Exact hashing misses near-duplicates; fuzzy dedup with MinHash LSH is the scalable standard; semantic dedup catches paraphrases at higher cost.' },

  { id: 'minhash', term: 'MinHash', cat: 'Data & tokenization',
    def: 'A technique that summarises a set (a document’s shingles) by the minimum value of each of k hash functions. The fraction of matching signature values between two documents estimates their Jaccard similarity.',
    exam: 'Pairs with LSH banding for near-duplicate detection; candidate probability is 1 − (1 − sʳ)ᵇ for b bands of r rows.' },

  { id: 'lsh', term: 'LSH (locality-sensitive hashing)', cat: 'Data & tokenization',
    def: 'Hashing designed so similar items collide. For MinHash, the signature is cut into b bands of r rows; documents matching in any band become candidate duplicates, avoiding an all-pairs comparison. Threshold ≈ (1/b)^(1/r).',
    exam: 'More rows per band → stricter (fewer false positives); more bands → higher recall.' },

  { id: 'pii-redaction', term: 'PII redaction', cat: 'Data & tokenization',
    def: 'Detecting and replacing personally identifiable information — emails, phone numbers, IP addresses, IDs, names, addresses — with placeholders before training. Uses regexes for structured identifiers and NER models (e.g. Presidio-style) for names and addresses.',
    exam: 'Must happen before training: a model can memorise and regurgitate PII, and it cannot be reliably removed afterwards.' },

  { id: 'data-contamination', term: 'Data contamination', cat: 'Data & tokenization',
    def: 'Evaluation benchmark items (questions, answers) appearing in training data, which inflates scores through memorisation. Detected and removed by n-gram overlap (e.g. 13-grams) between training documents and test sets.',
    exam: 'Signature: large gains on a public benchmark without matching gains on a private held-out set. Decontaminate before training.' },

  { id: 'data-mixture', term: 'Data mixture (blending)', cat: 'Data & tokenization',
    def: 'The sampling weights given to each source in a training corpus (web, code, maths, books, multilingual…). High-quality sources are often upsampled; many recipes anneal on the best data at the end of training.',
    exam: 'Mixture weights are tuned empirically with small proxy models before a large run.' },

  { id: 'nemo-curator', term: 'NeMo Curator', cat: 'NVIDIA stack',
    def: 'NVIDIA’s open-source, GPU-accelerated data-curation library for LLM training data: download and extraction, language ID, heuristic and classifier-based filtering, exact/fuzzy/semantic deduplication, PII redaction, decontamination and synthetic-data generation, scaled across multi-GPU, multi-node clusters with RAPIDS.',
    exam: 'The NVIDIA answer for "curate trillions of tokens quickly"; fuzzy dedup and classifiers are the big GPU wins.' },

  { id: 'rapids', term: 'RAPIDS', cat: 'NVIDIA stack',
    def: 'NVIDIA’s suite of open-source GPU data-science libraries — cuDF (dataframes), cuML (machine learning), cuGraph (graphs) and others — with APIs close to pandas and scikit-learn. NeMo Curator uses it to run curation on GPUs.',
    exam: 'Recognise cuDF/RAPIDS as the GPU-acceleration layer behind NeMo Curator.' },

  { id: 'datasets-streaming', term: 'Dataset streaming', cat: 'Tools & frameworks',
    def: 'Loading a dataset with `load_dataset(..., streaming=True)` to get an IterableDataset that reads remote shards lazily instead of downloading everything first. Supports lazy `map`, `filter`, `shuffle` (buffered) and `take`.',
    exam: 'The way to process or sample terabyte-scale corpora without local disk for the full copy.' },

  /* ---------------- SFT data ---------------- */
  { id: 'loss-masking', term: 'Loss masking', cat: 'Data & tokenization',
    def: 'Excluding positions from the training loss by setting their label to −100 (PyTorch cross-entropy’s ignore_index). In SFT, prompt, system and padding tokens are masked so only assistant responses (including end-of-turn) are learned.',
    exam: 'labels = −100 → ignored. Masked tokens still appear in input_ids as context.' },

  { id: 'packing', term: 'Packing', cat: 'Data & tokenization',
    def: 'Concatenating multiple short training examples into each fixed-length sequence (separated by EOS) to avoid wasted padding. Boundary-aware ("padding-free") packing resets position ids per example and uses variable-length attention so examples do not attend to each other.',
    exam: 'Improves GPU utilisation for variable-length data; know the cross-contamination caveat and its fix.' },

  { id: 'synthetic-data', term: 'Synthetic data', cat: 'Data & tokenization',
    def: 'Training data generated by models — instructions, answers, rephrasings, preference pairs — typically from a stronger teacher model, then filtered by reward models, LLM judges or execution checks. Techniques include Self-Instruct and Evol-Instruct; NVIDIA released Nemotron-4 340B Instruct and Reward to support such pipelines.',
    exam: 'Generate → filter/verify → decontaminate → spot-check. Check the generator’s licence; avoid training on unfiltered self-outputs.' },

  { id: 'instruction-format', term: 'Instruction data formats', cat: 'Data & tokenization',
    def: 'Common schemas for SFT data: Alpaca {instruction, input, output}; ShareGPT {conversations: [{from, value}]}; messages {messages: [{role, content}]}; prompt/completion; and preference {prompt, chosen, rejected}.',
    exam: 'Normalise to the messages format and let the chat template produce the model-specific string.' },
]
