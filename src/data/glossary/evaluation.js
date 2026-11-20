/**
 * Evaluation glossary — terms owned by Module 5 (Evaluation).
 * Shared terms (perplexity, loss, cross-entropy, rag, hallucination…) live in core.js.
 */
export default [
  { id: 'bleu', term: 'BLEU', cat: 'Evaluation',
    def: 'Bilingual Evaluation Understudy: a precision-oriented n-gram overlap metric (1- to 4-grams, geometric mean) with a brevity penalty for short outputs, scored 0–100 at corpus level. Designed for machine translation.',
    exam: 'BLEU rewards surface overlap with a reference — fine for translation, poor for open-ended generation where many correct answers share few n-grams. Use sacreBLEU so scores are comparable (it fixes tokenisation).' },

  { id: 'rouge', term: 'ROUGE (ROUGE-1 / 2 / L)', cat: 'Evaluation',
    def: 'Recall-Oriented Understudy for Gisting Evaluation: n-gram overlap between a candidate and reference summary. ROUGE-1 and ROUGE-2 count unigram and bigram overlap; ROUGE-L uses the longest common subsequence, rewarding in-order matches without requiring contiguity. Usually reported as F1.',
    exam: 'The classic summarisation metric. Recall orientation means it asks "how much of the reference did you cover?" — and, like BLEU, it is blind to paraphrase and factual errors.' },

  { id: 'bertscore', term: 'BERTScore', cat: 'Evaluation',
    def: 'A semantic similarity metric: embed candidate and reference tokens with a pretrained encoder, greedily match each token to its most similar counterpart by cosine similarity, and report precision, recall and F1 over those matches.',
    exam: 'Pick BERTScore over BLEU/ROUGE when paraphrases should count as correct. It still needs a reference and still cannot tell whether a fluent sentence is factually true.' },

  { id: 'exact-match', term: 'Exact match (EM) and token F1', cat: 'Evaluation',
    def: 'EM = 1 if the normalised prediction string equals the normalised reference (lower-case, strip punctuation and articles, collapse whitespace), else 0. Token F1 is the harmonic mean of token-level precision and recall between prediction and reference, giving partial credit.',
    exam: 'The standard pair for extractive QA (SQuAD-style). EM is strict and also used for GSM8K final answers; F1 forgives partially correct spans. Normalisation choices change scores by several points.' },

  { id: 'pass-at-k', term: 'pass@k', cat: 'Evaluation',
    def: 'The probability that at least one of k sampled programs passes the unit tests. Unbiased estimator: generate n ≥ k samples, count c correct, pass@k = 1 − C(n−c, k) / C(n, k), averaged over problems.',
    exam: 'Know why you sample n > k instead of exactly k (lower variance) and that pass@1 with greedy decoding is the usual headline number for code models (HumanEval, MBPP).' },

  { id: 'mmlu', term: 'MMLU', cat: 'Evaluation',
    def: 'Massive Multitask Language Understanding: ~14k four-option multiple-choice questions across 57 subjects (STEM, humanities, social sciences, professional exams). Usually reported 5-shot; random guessing scores 25%. MMLU-Pro is a harder 10-option successor.',
    exam: 'The go-to knowledge benchmark — and therefore one of the most contaminated. Scores depend on prompt format and on whether you score by log-likelihood or by generated letter.' },

  { id: 'lm-eval-harness', term: 'lm-evaluation-harness (lm_eval)', cat: 'Tools & frameworks',
    def: 'EleutherAI\'s open-source framework that runs hundreds of benchmark tasks (MMLU, HellaSwag, ARC, GSM8K, IFEval…) against Hugging Face, vLLM or API-served models with fixed prompts, few-shot settings and metrics. Exposed as the `lm_eval` CLI and the `lm_eval.simple_evaluate` Python API.',
    exam: 'The de-facto standard for reproducible benchmark numbers, and the engine behind many leaderboards and NVIDIA evaluation tooling. Report the task version, number of shots and normalisation with every score.' },

  { id: 'llm-as-judge', term: 'LLM-as-a-judge', cat: 'Evaluation',
    def: 'Using a strong LLM, prompted with a rubric, to grade outputs of another model — either pointwise (score one answer, e.g. 1–10) or pairwise (pick the better of two answers). MT-Bench and AlpacaEval are built on it.',
    exam: 'Know its biases — position, verbosity, self-preference, weak maths grading — and the mitigations: swap order, rubrics and references, a judge from a different model family, and validating against human labels.' },

  { id: 'human-eval', term: 'Human evaluation', cat: 'Evaluation',
    def: 'Having trained annotators rate or compare model outputs against written guidelines (Likert scales, pairwise preference, error tagging). The gold standard for helpfulness and tone, but slow and expensive; quality is measured with inter-annotator agreement.',
    exam: 'Not to be confused with the HumanEval code benchmark. Human eval is what you use to validate automatic metrics and LLM judges, not to run on every commit.' },

  { id: 'humaneval', term: 'HumanEval (benchmark)', cat: 'Evaluation',
    def: 'OpenAI\'s 164 hand-written Python programming problems, each a function signature plus docstring, checked by hidden unit tests. Scored with pass@k. MBPP (~1,000 basic problems) is its common companion.',
    exam: 'Code is graded by executing it, not by comparing text — so BLEU against a reference solution is the wrong metric. Running generated code requires a sandbox.' },

  { id: 'error-analysis', term: 'Error analysis', cat: 'Evaluation',
    def: 'Systematically reading failures to find out why a model fails: slice results by metadata (length, topic, language), cluster failing examples, build an error taxonomy with counts, then fix the largest category first and add regression tests.',
    exam: 'The step that turns an aggregate score into an action. Exam scenarios that say "accuracy dropped, what next?" usually want slicing and reading failures before retraining.' },

  { id: 'faithfulness', term: 'Faithfulness (groundedness)', cat: 'Evaluation',
    def: 'In RAG, the fraction of claims in a generated answer that are supported by the retrieved context. Computed by splitting the answer into atomic claims and checking each against the context, usually with an LLM judge or NLI model.',
    exam: 'Faithfulness measures hallucination relative to the context, not truth: an answer can be faithful to a wrong document. Pair it with context precision/recall to locate retrieval vs generation errors.' },

  { id: 'ragas', term: 'RAGAS', cat: 'Tools & frameworks',
    def: 'An open-source framework for reference-light RAG evaluation. Core metrics: faithfulness, answer relevancy, context precision and context recall, mostly computed with an LLM judge and embeddings.',
    exam: 'Map each metric to the stage it diagnoses: context precision/recall → retriever, faithfulness/answer relevancy → generator.' },

  { id: 'contamination-detection', term: 'Contamination detection', cat: 'Evaluation',
    def: 'Checking whether benchmark test items leaked into a model\'s training data: n-gram overlap between test and training corpora (e.g. 13-gram matches), canary strings, membership-inference signals such as unusually low perplexity on test items, and comparing scores on original vs freshly rewritten questions.',
    exam: 'A contaminated benchmark overstates capability. If a fine-tuned model jumps on one benchmark but not on rephrased or held-out variants, suspect contamination first.' },

  { id: 'inter-annotator-agreement', term: 'Inter-annotator agreement (IAA)', cat: 'Evaluation',
    def: 'How consistently independent raters label the same items, corrected for chance. Cohen\'s κ = (p_o − p_e)/(1 − p_e) for two raters; Fleiss\' κ or Krippendorff\'s α for more raters or missing labels. κ above ~0.6 is usually considered substantial agreement.',
    exam: 'Low IAA means the guidelines (or the task) are ambiguous — fix the rubric before trusting the labels or using them to validate an LLM judge.' },

  { id: 'position-bias', term: 'Position bias (judge bias)', cat: 'Evaluation',
    def: 'An LLM judge\'s tendency to prefer the answer shown first (or second) regardless of quality. Related judge biases: verbosity bias (longer looks better) and self-preference (a model rates its own outputs higher).',
    exam: 'Mitigation is to evaluate both orders and only count consistent wins (or average), randomise order, and control for length.' },

  { id: 'log-likelihood-scoring', term: 'Log-likelihood (multiple-choice) scoring', cat: 'Evaluation',
    def: 'Evaluating multiple-choice tasks without generation: compute the log-probability the model assigns to each answer continuation given the question, and pick the highest. `acc` uses the raw sum; `acc_norm` divides by the answer length (bytes in lm-eval) so long answers are not penalised.',
    exam: 'Explains why the same model gets different MMLU or HellaSwag numbers under different harnesses: likelihood vs generated-letter scoring and length normalisation.' },

  { id: 'few-shot-eval', term: 'n-shot evaluation setting', cat: 'Evaluation',
    def: 'The number of solved examples prepended to each benchmark prompt (0-shot, 5-shot, 25-shot…). Classic settings: MMLU 5-shot, ARC-Challenge 25-shot, HellaSwag 10-shot, GSM8K 5-shot (or 8-shot CoT), TruthfulQA 0-shot.',
    exam: 'Scores are only comparable with identical shots, prompt template and chat-template use. Base models need few-shot examples to understand the format; instruct models are often evaluated 0-shot with a chat template.' },

  { id: 'nemo-evaluator', term: 'NeMo Evaluator', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s evaluation tooling, available as a NeMo microservice and as an open-source SDK/launcher. It runs academic benchmarks (wrapping harnesses such as lm-evaluation-harness), custom datasets, LLM-as-a-judge and retriever/RAG evaluations against models served behind an OpenAI-compatible endpoint (NIM, vLLM, TensorRT-LLM).',
    exam: 'The NVIDIA answer to "how do I evaluate a customised model at scale?" — it fits into the NeMo customise → evaluate → deploy loop.' },

  { id: 'eval-set', term: 'Domain evaluation set (golden set)', cat: 'Evaluation',
    def: 'A curated, versioned set of inputs with references or rubrics that represents your real traffic: covering common cases, known hard cases and safety cases, kept strictly out of training data, and grown from production failures.',
    exam: 'Public benchmarks rarely match your task. The best evaluation investment is a few hundred well-labelled in-domain examples run on every model change.' },
]
