/**
 * Module 6 glossary — Prompt Engineering. Prompting techniques, reasoning patterns,
 * decoding controls, structured output and retrieval-augmented generation.
 * Shared terms (rag, hallucination, embedding, logits, softmax, context-window …) live in core.js.
 */
export default [
  /* ---------------- Prompting techniques ---------------- */
  { id: 'in-context-learning', term: 'In-context learning (ICL)', cat: 'Prompting & RAG',
    def: 'A model\'s ability to perform a task from instructions and examples placed in the prompt, with no weight update. The demonstrations condition the next-token predictions only for that request; they mostly convey format, label space and input distribution.',
    exam: 'If an option says few-shot prompting "updates", "trains" or "fine-tunes" the model, it is wrong. ICL is ephemeral and costs prompt tokens on every call.' },

  { id: 'zero-shot', term: 'Zero-shot prompting', cat: 'Prompting & RAG',
    def: 'Asking a model to perform a task from an instruction alone, without any worked examples. Works well on instruct models for common tasks.',
    exam: 'Zero examples. If the model gets the output format wrong, the usual next step is one/few-shot examples or constrained decoding.' },

  { id: 'few-shot', term: 'Few-shot prompting', cat: 'Prompting & RAG',
    def: 'Including a small number (typically 2–8) of input→output demonstrations in the prompt before the real query. One-shot is the special case of a single example. Quality depends on example relevance, label balance, consistent formatting and order.',
    exam: 'Expect questions on majority-label and recency bias, and on choosing similar examples dynamically (retrieve nearest labelled examples per query).' },

  { id: 'system-prompt', term: 'System prompt', cat: 'Prompting & RAG',
    def: 'The first message of a chat, in the `system` role, setting persistent behaviour: persona, rules, constraints, output format and safety instructions. The chat template places it before user and assistant turns.',
    exam: 'Use it for rules that apply to the whole conversation. Assume it can be extracted — never store secrets in it.' },

  { id: 'prompt-template', term: 'Prompt template', cat: 'Prompting & RAG',
    def: 'A reusable prompt with fixed instructions and placeholders (e.g. {context}, {question}) filled at runtime, typically written as a format string or Jinja template and kept under version control.',
    exam: 'Production practice: version templates, log the version with each request and run an eval set on every change — a wording tweak is a behaviour change.' },

  /* ---------------- Reasoning patterns ---------------- */
  { id: 'chain-of-thought', term: 'Chain-of-thought (CoT)', cat: 'Prompting & RAG',
    def: 'Prompting the model to produce intermediate reasoning steps before its final answer. Zero-shot CoT adds a trigger such as "Let\'s think step by step"; few-shot CoT shows demonstrations that contain worked reasoning.',
    exam: 'CoT improves multi-step arithmetic and logic on capable models at the cost of more output tokens (latency). Gains are small or negative on very small models.' },

  { id: 'self-consistency', term: 'Self-consistency', cat: 'Prompting & RAG',
    def: 'Sampling several chain-of-thought completions at temperature > 0 and returning the most common final answer (majority vote). The agreement rate doubles as a confidence signal.',
    exam: 'Requires sampling — greedy decoding produces N identical chains. Costs roughly N× the generation tokens.' },

  { id: 'tree-of-thought', term: 'Tree-of-thought (ToT)', cat: 'Prompting & RAG',
    def: 'A search over partial reasoning steps: the model proposes several next thoughts, evaluates them, and a search algorithm (BFS/DFS) expands promising branches and backtracks from dead ends.',
    exam: 'Pick it for planning or puzzle tasks where early choices matter; it costs many model calls per problem.' },

  { id: 'react', term: 'ReAct (Reason + Act)', cat: 'Prompting & RAG',
    def: 'An agent prompting pattern that interleaves Thought (reasoning), Action (a tool or API call) and Observation (the tool result) in a loop until the model produces a final answer.',
    exam: 'The answer whenever a scenario needs the model to look up live information, call tools or take actions between reasoning steps.' },

  { id: 'reasoning-model', term: 'Reasoning model', cat: 'Prompting & RAG',
    def: 'An LLM trained (typically with reinforcement learning on verifiable tasks) to generate a long internal reasoning trace before answering — e.g. DeepSeek-R1, Qwen3 in thinking mode, NVIDIA Nemotron reasoning models. Some support toggling reasoning on or off per request.',
    exam: 'Prompt them plainly, skip few-shot CoT examples, allow a large max_tokens budget and strip the trace before parsing. Trade-off: higher accuracy for more latency and tokens.' },

  /* ---------------- Decoding ---------------- */
  { id: 'greedy-decoding', term: 'Greedy decoding', cat: 'Prompting & RAG',
    def: 'Choosing the highest-probability token at every step (argmax). Deterministic and fast; prone to repetitive loops in long open-ended text. Equivalent to temperature → 0.',
    exam: 'Use for classification, extraction and other single-correct-answer tasks. In transformers, `do_sample=False` means greedy and ignores temperature/top-p.' },

  { id: 'beam-search', term: 'Beam search', cat: 'Prompting & RAG',
    def: 'Keeps the `num_beams` most probable partial sequences at each step and returns the highest-scoring complete one, optionally with a length penalty. Deterministic and approximately maximises sequence likelihood.',
    exam: 'Good for translation and short outputs; produces bland, repetitive text for open-ended generation and costs roughly num_beams× compute.' },

  { id: 'temperature', term: 'Temperature', cat: 'Prompting & RAG',
    def: 'A divisor applied to logits before softmax: p_i ∝ exp(z_i / T). T < 1 sharpens the distribution toward the top token; T > 1 flattens it. Logits [2,1,0] give a top probability of 0.665 at T=1 and 0.867 at T=0.5.',
    exam: 'Low (0–0.3) for factual/extraction/code, 0.7–1.0 for creative or diverse outputs. Very high values cause incoherence and more hallucination.' },

  { id: 'top-k', term: 'Top-k sampling', cat: 'Prompting & RAG',
    def: 'Restricts sampling to the k most probable tokens, renormalising their probabilities. A fixed cut-off regardless of how peaked or flat the distribution is.',
    exam: 'top_k = 1 is greedy. Its weakness — fixed k — is what top-p and min-p fix.' },

  { id: 'top-p', term: 'Top-p (nucleus) sampling', cat: 'Prompting & RAG',
    def: 'Restricts sampling to the smallest set of most-probable tokens whose cumulative probability is at least p (e.g. 0.9), so the candidate set grows when the model is uncertain and shrinks when it is confident.',
    exam: 'Typical 0.9–0.95. Very low values (e.g. 0.1) are effectively greedy. Usually tune temperature or top-p, not both aggressively.' },

  { id: 'min-p', term: 'Min-p sampling', cat: 'Prompting & RAG',
    def: 'Keeps tokens whose probability is at least min_p × the probability of the most likely token (e.g. 0.1 × 0.6 = 0.06). The threshold scales with model confidence, which keeps text coherent even at higher temperatures.',
    exam: 'Supported by transformers (`min_p`) and vLLM. Typical values 0.05–0.1.' },

  { id: 'repetition-penalty', term: 'Repetition, frequency and presence penalties', cat: 'Prompting & RAG',
    def: 'Logit adjustments that discourage reuse of tokens already generated. Repetition penalty (HF/vLLM) scales the logits of any previously seen token by a factor > 1; frequency penalty subtracts α × occurrence count; presence penalty subtracts β once if the token appeared at all (OpenAI-style, range −2 to 2).',
    exam: 'The fix for loops and repeated phrases. Too strong (repetition ≥ 1.5) breaks JSON, code and natural text that legitimately repeats tokens.' },

  { id: 'stop-sequence', term: 'Stop sequence', cat: 'Prompting & RAG',
    def: 'A string that ends generation as soon as it is produced (APIs usually exclude it from the returned text). Together with max tokens, it bounds output length and stops a model from continuing a pattern, e.g. inventing the next "Q:".',
    exam: 'Runaway or over-long output → max_tokens + stop sequences. Essential with base models and few-shot completion prompts.' },

  { id: 'structured-output', term: 'Structured output', cat: 'Prompting & RAG',
    def: 'Getting output in a machine-readable format (JSON, a fixed label set, a regex or grammar). Three strengths: prompt instructions (best effort), JSON mode (valid JSON, any shape) and schema-constrained decoding (guaranteed to match a schema).',
    exam: 'When downstream code must parse every response, the answer is constrained decoding with a schema, not stronger wording.' },

  { id: 'constrained-decoding', term: 'Constrained (guided) decoding', cat: 'Prompting & RAG',
    def: 'Enforcing a grammar during generation: a JSON Schema, regex, choice list or context-free grammar is compiled into a state machine, and at each step logits of tokens that would violate it are masked. Implemented by XGrammar, Outlines, llguidance and lm-format-enforcer; exposed by vLLM, TensorRT-LLM and NIM.',
    exam: 'Guarantees shape, not correctness. Output can still be truncated by max_tokens. Put reasoning fields before answer fields in the schema.' },

  /* ---------------- Retrieval ---------------- */
  { id: 'chunking', term: 'Chunking', cat: 'Prompting & RAG',
    def: 'Splitting documents into passages for embedding and retrieval — typically 200–800 tokens with 10–20% overlap, ideally along structure (headings, paragraphs, tables).',
    exam: 'Too small loses context; too large blurs the embedding and wastes context window. Overlap keeps boundary-straddling facts intact.' },

  { id: 'vector-database', term: 'Vector database', cat: 'Prompting & RAG',
    def: 'A store for embedding vectors plus metadata that answers approximate-nearest-neighbour queries using indexes such as HNSW, IVF or IVF-PQ, with filtering, updates and persistence. Examples: Milvus (GPU-accelerated via cuVS), Qdrant, Weaviate, pgvector; FAISS is the underlying library many use.',
    exam: 'Know the index trade-offs (flat = exact but O(N); HNSW = fast, memory-heavy; PQ = compressed, some recall loss) and that metadata filtering enforces access control and versioning.' },

  { id: 'bm25', term: 'BM25', cat: 'Prompting & RAG',
    def: 'A classic sparse keyword-ranking function based on term frequency, inverse document frequency and document-length normalisation. Excellent at exact matches (error codes, identifiers, names).',
    exam: 'The "sparse" half of hybrid search — the fix when dense retrieval misses exact strings.' },

  { id: 'hybrid-search', term: 'Hybrid search', cat: 'Prompting & RAG',
    def: 'Combining sparse (BM25) and dense (embedding) retrieval and fusing the ranked lists, commonly with reciprocal rank fusion: score(d) = Σ 1/(k + rank_i(d)), k ≈ 60.',
    exam: 'Choose it when queries mix natural language with exact identifiers. RRF needs no score normalisation between the two retrievers.' },

  { id: 'reranking', term: 'Reranking (cross-encoder)', cat: 'Prompting & RAG',
    def: 'A second retrieval stage in which a cross-encoder reads each (query, passage) pair jointly and outputs a relevance score, reordering 20–100 first-stage candidates so only the best few reach the LLM.',
    exam: 'More accurate than bi-encoder similarity but one forward pass per pair, so it is applied to candidates, never the whole corpus. NVIDIA offers reranking NIMs in NeMo Retriever.' },

  { id: 'nemo-retriever', term: 'NVIDIA NeMo Retriever', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s collection of retrieval microservices for RAG — document extraction, embedding and reranking models — packaged as NIMs with OpenAI-style APIs (`/v1/embeddings`, and a `/v1/ranking` endpoint for reranking), used in the NVIDIA RAG Blueprint.',
    exam: 'The NVIDIA answer for "production embedding/reranking for an enterprise RAG pipeline". Embedding NIMs distinguish `input_type` query vs passage.' },
]
