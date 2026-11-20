export default {
  id: 'evaluation', n: 5, domain: 'evaluation', weight: 7,
  title: 'Evaluation',
  blurb: 'Measure what your model can actually do: pick the right metric for the task, run reproducible benchmarks with lm-evaluation-harness, detect contamination, and build LLM-judge, human and RAG evaluations you can run on every change.',
  hours: '3–4 h',
  lessons: [
    /* ------------------------------------------------------------------ */
    {
      id: 'metrics',
      title: 'Metrics: loss, perplexity, BLEU/ROUGE, exact match, pass@k',
      minutes: 22,
      outcomes: [
        'Interpret training and validation loss curves and diagnose overfitting, divergence and data issues',
        'Compute perplexity correctly with a sliding window and explain when it is comparable',
        'Calculate BLEU, ROUGE-1/2/L, BERTScore, exact match and token F1, and choose between them',
        'Apply the unbiased pass@k estimator and log-likelihood scoring for multiple-choice tasks',
      ],
      blocks: [
        { t: 'lede', x: 'A metric is a compression of model behaviour into one number — and every compression throws something away. This lesson walks through the metrics you will meet in training logs, papers and the exam, shows how to compute each one in a few lines of Python, and, most importantly, when each one lies to you.' },

        { t: 'h2', x: 'Loss curves: your first evaluation' },
        { t: 'p', x: 'Before any benchmark, the [[loss]] curve tells you whether training worked. Always log **training loss** (noisy, per step) and **validation loss** (on held-out data, every N steps). Validation loss is the same [[cross-entropy]], just on text the model never trained on.' },
        { t: 'table', head: ['What you see', 'Likely meaning', 'What to do'], rows: [
          ['Train ↓, val ↓, gap small', 'Healthy learning', 'Keep going; watch for the plateau'],
          ['Train ↓, val ↑ after a minimum', '[[overfitting|Overfitting]] (common with small SFT sets, many epochs)', 'Early-stop at the val minimum, fewer epochs, lower LR or LoRA rank, more data'],
          ['Initial loss ≫ ln(V)', 'Bug: wrong labels, init or tokenizer mismatch', 'Check a batch by hand; ln(50k) ≈ 10.8 is the uniform-guess baseline'],
          ['Sudden spikes, then recovery or NaN', 'LR too high, bad data batch, fp16 overflow', 'Lower LR, gradient clipping, bf16, inspect the batch at the spike'],
          ['Val loss far below train loss', 'Leakage, or dropout only active in train', 'Check train/val overlap; compare with dropout off'],
          ['Flat from the start', 'LR too low, frozen params, labels all masked', 'Verify `requires_grad` and that loss-masked labels are not all −100'],
        ]},
        { t: 'callout', kind: 'tip', title: 'Loss is necessary, not sufficient', x: 'Lower validation loss usually means a better language model, but after SFT or alignment, loss correlates weakly with what users care about (instruction following, helpfulness, safety). Use loss to debug training; use task metrics and judges to decide what ships.' },

        { t: 'h2', x: 'Perplexity, computed properly' },
        { t: 'p', x: '[[perplexity|Perplexity]] = exp(mean token-level negative log-likelihood). A perplexity of 12 means the model is, on average, as uncertain as if choosing uniformly among 12 tokens. Two subtleties decide whether your number is meaningful: the **context** each token gets, and the **tokenizer**.' },
        { t: 'p', x: 'A model has a fixed [[context-window]], so long text must be split. Cutting it into disjoint chunks means the first tokens of every chunk are predicted with almost no context, inflating perplexity. The fix is a **sliding window**: advance by a `stride` smaller than the window and score only the tokens that are new in each window, so every scored token sees up to `max_len − stride` tokens of history.' },
        { t: 'code', lang: 'python', title: 'perplexity.py — sliding-window perplexity on WikiText-2', x: `import math, torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceTB/SmolLM2-135M"          # swap in your own checkpoint
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()

test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\\n\\n".join(test["text"]), return_tensors="pt").input_ids   # (1, N) — one long stream

max_len, stride = 1024, 512       # window size and step; stride < max_len => overlapping context
nll_sum, n_scored, prev_end = 0.0, 0, 0
for begin in range(0, ids.size(1), stride):
    end = min(begin + max_len, ids.size(1))
    new = end - prev_end                          # tokens not scored by an earlier window
    window = ids[:, begin:end].to(device)
    targets = window.clone()
    targets[:, :-new] = -100                      # context-only tokens: visible but not scored
    with torch.no_grad():
        logits = model(input_ids=window).logits.float()
    # position t predicts token t+1, so shift by one
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                           targets[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    nll_sum += loss.item()
    n_scored += (targets[:, 1:] != -100).sum().item()
    prev_end = end
    if end == ids.size(1):
        break

print(f"scored {n_scored:,} tokens  ppl = {math.exp(nll_sum / n_scored):.2f}")
# stride=max_len (disjoint chunks) gives a noticeably HIGHER ppl; smaller stride -> lower, slower` },
        { t: 'callout', kind: 'trap', title: 'Perplexity across tokenizers is meaningless', x: [
          'Perplexity is per **token**. A model with a 128k-token vocabulary packs more text into each token than one with 32k, so its per-token uncertainty is not comparable. Compare perplexity only between checkpoints that share a [[tokenizer]] (e.g. before/after fine-tuning, or FP16 vs INT4 of the same model).',
          'To compare across tokenizers, normalise to **bits per byte** (or per character): total NLL in bits ÷ number of UTF-8 bytes.',
        ]},

        { t: 'h2', x: 'Reference-based text metrics: BLEU, ROUGE, BERTScore' },
        { t: 'p', x: 'When there is a reference output, the classical approach is to measure overlap with it. [[bleu|BLEU]] is precision-oriented (how much of what you wrote appears in the reference), with a brevity penalty so you cannot game it with one perfect word. [[rouge|ROUGE]] is recall-oriented (how much of the reference you covered): ROUGE-1 for unigrams, ROUGE-2 for bigrams, ROUGE-L for the longest common subsequence. [[bertscore|BERTScore]] replaces exact n-gram matches with contextual-embedding similarity, so paraphrases get credit.' },
        { t: 'code', lang: 'python', title: 'text_metrics.py — pip install sacrebleu rouge-score bert-score', x: `import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore

refs  = ["The cat sat on the mat.", "Revenue grew 12% year over year."]
preds = ["A cat was sitting on the mat.", "Revenue rose by 12 percent compared with last year."]

# BLEU: corpus-level, 0-100. sacreBLEU fixes tokenisation so numbers are comparable across papers.
bleu = sacrebleu.corpus_bleu(preds, [refs])          # list of reference STREAMS
print(f"BLEU {bleu.score:.1f}")

# ROUGE-1/2/L F1 per example (stemming so "sitting" ~ "sit")
sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
for p, r in zip(preds, refs):
    s = sc.score(r, p)                               # score(target, prediction)
    print({k: round(v.fmeasure, 3) for k, v in s.items()})

# BERTScore: embedding similarity; paraphrase #2 scores far better here than under BLEU
P, R, F1 = bertscore(preds, refs, lang="en")
print("BERTScore F1", [round(x, 3) for x in F1.tolist()])` },
        { t: 'callout', kind: 'warn', title: 'Overlap ≠ correctness', x: '"Revenue grew 21% year over year" scores almost perfectly against the reference above with BLEU, ROUGE and BERTScore — and is wrong. Overlap metrics cannot see factual errors, negations or swapped numbers, and they punish valid answers that are phrased differently. Use them for translation and summarisation trend-tracking, not as the only gate for open-ended generation.' },

        { t: 'h2', x: 'Exact match, F1 and pass@k: when the answer is checkable' },
        { t: 'p', x: 'Many tasks have a verifiable answer, and then you should verify it rather than measure overlap. For short-answer QA use [[exact-match|exact match and token F1]] after normalisation; for maths, extract the final number and exact-match it (GSM8K); for code, run unit tests and report [[pass-at-k|pass@k]].' },
        { t: 'code', lang: 'python', title: 'em_f1_passk.py', x: `import re, string
from collections import Counter
import numpy as np

def normalize(s: str) -> str:
    """SQuAD-style normalisation: lower, drop punctuation and articles, squeeze spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\\b(a|an|the)\\b", " ", s)
    return " ".join(s.split())

def exact_match(pred, ref):
    return float(normalize(pred) == normalize(ref))

def token_f1(pred, ref):
    p, r = normalize(pred).split(), normalize(ref).split()
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)

print(exact_match("The Eiffel Tower.", "eiffel tower"))        # 1.0
print(round(token_f1("the tall Eiffel Tower", "Eiffel Tower"), 3))  # 0.8 — partial credit

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021): n samples, c of them pass the tests.
    = 1 - C(n-c, k) / C(n, k), computed as a stable product."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

print(pass_at_k(n=10, c=3, k=1))   # 0.3
print(pass_at_k(n=10, c=3, k=5))   # 0.9167 = 1 - C(7,5)/C(10,5) = 1 - 21/252` },
        { t: 'p', x: 'Why not just sample k programs and check whether any passes? Because that estimate has high variance. Sampling n > k (e.g. n = 200 for pass@1/10/100 in the Codex paper) and applying the combinatorial formula gives an unbiased, lower-variance estimate of the probability that k random draws contain a correct one. Note that 1 − (1 − c/n)^k looks similar but is **biased**.' },

        { t: 'h2', x: 'Multiple choice: scoring by log-likelihood' },
        { t: 'p', x: 'For multiple-choice benchmarks ([[mmlu|MMLU]], HellaSwag, ARC) you usually do not let the model generate. Instead you compute the log-probability of each candidate answer as a continuation of the prompt and pick the largest — [[log-likelihood-scoring|log-likelihood scoring]]. This works for base models that would never output a clean "B", and it is deterministic.' },
        { t: 'code', lang: 'python', title: 'mc_loglik.py — how lm-eval scores a multiple-choice item', x: `import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).eval()

def continuation_logprob(context: str, continuation: str) -> float:
    ctx = tok(context, return_tensors="pt").input_ids
    cont = tok(continuation, add_special_tokens=False, return_tensors="pt").input_ids
    ids = torch.cat([ctx, cont], dim=1)
    with torch.no_grad():
        logp = F.log_softmax(model(input_ids=ids).logits.float(), dim=-1)
    # logits at position t predict token t+1 -> score only the continuation tokens
    positions = torch.arange(ctx.size(1) - 1, ids.size(1) - 1)
    return logp[0, positions, ids[0, ctx.size(1):]].sum().item()

question = "Question: What gas do plants absorb for photosynthesis?\\nAnswer:"
choices = [" oxygen", " carbon dioxide", " nitrogen", " helium"]
scores = [continuation_logprob(question, c) for c in choices]
acc_pick = max(range(4), key=lambda i: scores[i])                                   # 'acc'
norm_pick = max(range(4), key=lambda i: scores[i] / len(choices[i].encode()))       # 'acc_norm'
print(choices[acc_pick], "|", choices[norm_pick])` },
        { t: 'p', x: 'Summed log-probabilities penalise long answers (more tokens, more negative). lm-eval reports both `acc` (raw sum) and `acc_norm` (divided by answer length in bytes); HellaSwag and ARC-Challenge are conventionally quoted with `acc_norm`, MMLU with `acc` over the single answer letter.' },

        { t: 'h2', x: 'Choosing the right metric' },
        { t: 'table', head: ['Task', 'Good primary metric', 'Avoid as the only metric'], rows: [
          ['Pretraining / checkpoint comparison (same tokenizer)', 'Validation loss, perplexity', 'Perplexity across tokenizers'],
          ['Machine translation', 'sacreBLEU, chrF, COMET (learned)', 'Raw BLEU with custom tokenisation'],
          ['Summarisation', 'ROUGE-1/2/L + BERTScore + faithfulness check', 'ROUGE alone (misses hallucinations)'],
          ['Extractive QA', 'Exact match + token F1', 'BLEU'],
          ['Maths word problems', 'Exact match on extracted final answer', 'Overlap with reference rationale'],
          ['Code generation', 'pass@k with unit tests (sandboxed)', 'BLEU / exact match vs reference code'],
          ['Multiple-choice knowledge', 'Accuracy via log-likelihood or constrained letter', 'Free-form generation + regex without validation'],
          ['Chat / open-ended helpfulness', 'LLM-as-a-judge validated by human eval', 'BLEU/ROUGE, perplexity'],
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Scenario questions describe a task and ask for the most appropriate metric: code → **pass@k**, summarisation → **ROUGE**, translation → **BLEU**, extractive QA → **EM/F1**, paraphrase-tolerant similarity → **BERTScore**, open-ended chat → **LLM judge / human eval**.',
          'Expect "perplexity = exp(cross-entropy)" arithmetic (loss 2.0 → ppl ≈ 7.39) and the rule that perplexity is only comparable with the same tokenizer.',
          'Know the pass@k formula and why n > k samples are drawn.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Loss curves debug training; a rising validation loss with falling training loss is overfitting — stop at the minimum.',
          'Perplexity = exp(mean NLL); compute it with a sliding window and compare only within one tokenizer (or use bits per byte).',
          'BLEU (precision, translation) and ROUGE (recall, summarisation) measure overlap, not truth; BERTScore adds paraphrase tolerance.',
          'When answers are verifiable, verify: EM/F1 for QA, final-answer EM for maths, unit tests and unbiased pass@k for code.',
          'Multiple-choice benchmarks are scored by comparing answer log-likelihoods; `acc_norm` corrects for answer length.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'benchmarks',
      title: 'Lab: benchmarks with lm-evaluation-harness, and contamination',
      minutes: 35,
      outcomes: [
        'Run lm-evaluation-harness from the CLI and Python on a base and a fine-tuned model and compare results',
        'Describe what MMLU, HellaSwag, ARC, GSM8K, HumanEval, IFEval, MT-Bench and TruthfulQA measure and how they are scored',
        'Control few-shot settings, chat templates and normalisation so numbers are reproducible',
        'Detect benchmark contamination with n-gram overlap and related signals',
        'Position NeMo Evaluator for evaluation at scale in the NVIDIA stack',
      ],
      blocks: [
        { t: 'lede', x: 'Benchmarks let you compare against the rest of the world — but only if you run them the same way the world does. In this lab you run [[lm-eval-harness|lm-evaluation-harness]] on a model before and after fine-tuning, learn which knobs silently change the numbers, and check whether a score can be trusted at all.' },

        { t: 'h2', x: 'The benchmark zoo' },
        { t: 'table', head: ['Benchmark', 'What it measures', 'Format and size', 'Typical setting / metric'], rows: [
          ['[[mmlu|MMLU]]', 'World knowledge across 57 subjects', '~14k 4-way multiple choice', '5-shot, accuracy (25% = chance)'],
          ['HellaSwag', 'Commonsense sentence completion (adversarially filtered endings)', '~10k validation items, 4 endings', '10-shot, `acc_norm`'],
          ['ARC (Easy / Challenge)', 'Grade-school science reasoning', '~7.8k questions, mostly 4 options', 'ARC-Challenge 25-shot, `acc_norm`'],
          ['GSM8K', 'Multi-step grade-school maths', '8.5k problems (1,319 test)', '5-shot or 8-shot CoT, exact match on the final number'],
          ['[[humaneval|HumanEval]]', 'Python code synthesis', '164 problems with unit tests', 'pass@1 (greedy) / pass@k'],
          ['IFEval', 'Following verifiable format instructions ("use exactly 3 bullet points", "no commas")', '~500 prompts, ~25 instruction types', '0-shot, strict/loose accuracy at prompt and instruction level'],
          ['MT-Bench', 'Multi-turn chat quality (writing, reasoning, coding…)', '80 two-turn questions, 8 categories', '[[llm-as-judge|LLM judge]] score 1–10'],
          ['TruthfulQA', 'Avoiding common misconceptions and imitative falsehoods', '817 questions, 38 categories', '0-shot, MC1/MC2 accuracy or judged generation'],
        ]},
        { t: 'p', x: 'Notice three different scoring styles: **likelihood** ([[mmlu|MMLU]], HellaSwag, ARC, TruthfulQA-MC), **generate-then-check** (GSM8K, HumanEval, IFEval) and **judge** (MT-Bench). Generative tasks are slower and depend on decoding settings; likelihood tasks are fast but do not test whether the model can produce the answer in a usable format.' },

        { t: 'lab', title: 'Benchmark a model before and after fine-tuning', x: 'Run the same lm-eval tasks on a base model and on a fine-tuned version (your SFT/LoRA checkpoint from Module 4, or the official Instruct variant as a stand-in), compare per-task scores, and save results as JSON for later regression checks.', gpu: 'Free Colab T4 (or CPU with --limit for a smoke test)', time: '40 min', steps: [
          'Install the harness: `pip install "lm_eval[hf]"` (add `vllm` for faster generative tasks).',
          'Smoke-test with `--limit 50` to check the pipeline end to end — never report limited numbers.',
          'Run the base model on arc_easy, hellaswag and gsm8k with fixed shots and save to `results/base`.',
          'Run the fine-tuned model (merged checkpoint or `peft=` adapter) with identical settings into `results/ft`.',
          'For the instruct/chat model, run a second time with `--apply_chat_template` and compare: the template alone can move scores by points.',
          'Load both JSON files in Python, print a per-task delta table, and flag drops larger than the standard error.',
        ]},
        { t: 'code', lang: 'bash', title: 'lm_eval CLI — base vs fine-tuned', x: `pip install "lm_eval[hf]"

# 1) base model: likelihood tasks + one generative task, fixed shots, saved samples for error analysis
lm_eval --model hf \\
  --model_args pretrained=Qwen/Qwen2.5-0.5B,dtype=bfloat16 \\
  --tasks arc_easy,hellaswag,gsm8k \\
  --num_fewshot 5 \\
  --batch_size auto \\
  --device cuda:0 \\
  --output_path results/base \\
  --log_samples

# 2) fine-tuned: a LoRA adapter on top of the same base (or pretrained=./my-merged-model)
lm_eval --model hf \\
  --model_args pretrained=Qwen/Qwen2.5-0.5B,peft=./qwen-lora,dtype=bfloat16 \\
  --tasks arc_easy,hellaswag,gsm8k --num_fewshot 5 --batch_size auto \\
  --output_path results/ft --log_samples

# 3) chat model: apply its chat template and present few-shot examples as turns
lm_eval --model hf \\
  --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16 \\
  --tasks gsm8k,ifeval --apply_chat_template --fewshot_as_multiturn \\
  --batch_size auto --output_path results/instruct

lm_eval ls tasks | grep -i mmlu      # discover task names (mmlu, mmlu_pro, arc_challenge, truthfulqa_mc2, ...)` },
        { t: 'code', lang: 'python', title: 'compare.py — Python API and a delta table', x: `import lm_eval

def run(model_args: str, tasks, shots=5, limit=None):
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=shots,
        batch_size=8,
        limit=limit,          # None for real numbers; an int only for smoke tests
    )
    return out["results"]     # {task: {"acc,none": ..., "acc_stderr,none": ..., ...}}

tasks = ["arc_easy", "hellaswag"]
base = run("pretrained=Qwen/Qwen2.5-0.5B,dtype=bfloat16", tasks)
ft   = run("pretrained=Qwen/Qwen2.5-0.5B,peft=./qwen-lora,dtype=bfloat16", tasks)

for t in tasks:
    metric = "acc_norm,none" if "acc_norm,none" in base[t] else "acc,none"
    b, f = base[t][metric], ft[t][metric]
    se = base[t][metric.replace(",", "_stderr,")]
    flag = "  <-- significant drop" if f < b - 2 * se else ""
    print(f"{t:12s} {metric:14s} base={b:.3f} ft={f:.3f} delta={f - b:+.3f} (+/-{se:.3f}){flag}")` },
        { t: 'callout', kind: 'tip', title: 'Read the samples, not just the score', x: '`--log_samples` writes every prompt, model output and per-item score to JSONL. When GSM8K drops after fine-tuning, open the samples: often the model still solves the problem but no longer ends with the "#### 42" format the answer extractor expects. That is a formatting regression, not a reasoning regression — and it changes what you fix.' },

        { t: 'h2', x: 'What silently changes your numbers' },
        { t: 'ul', items: [
          '**Number of shots** ([[few-shot-eval|n-shot setting]]): 0-shot vs 5-shot MMLU on a small base model can differ by 5–10 points.',
          '**Prompt template and chat template**: instruct models evaluated without their template under-perform; base models evaluated with one may break.',
          '**Scoring**: `acc` vs `acc_norm`, likelihood vs generated letter, strict vs flexible answer extraction (lm-eval GSM8K reports both `strict-match` and `flexible-extract`).',
          '**Decoding** for generative tasks: greedy vs sampled, max new tokens, stop sequences.',
          '**Task version**: harness tasks are versioned; report the lm-eval version and task config hash with results.',
          '**Precision and batch size**: bf16 vs fp32 and padding can shift likelihoods slightly — small, but enough to flip near-ties.',
        ]},
        { t: 'callout', kind: 'trap', title: 'Comparing numbers from different harnesses', x: 'A model card says "MMLU 65" and your run says 61. Before concluding anything, check shots, template, scoring method and harness version. Leaderboards are comparable only because they fix all of these. When you compare your own checkpoints, change exactly one thing — the model.' },

        { t: 'h2', x: 'Data contamination and how to detect it' },
        { t: 'p', x: 'Benchmarks are public, and web-scale pretraining corpora contain copies of them. If test items (or close paraphrases) were in training, the benchmark measures memorisation. Fine-tuning data is a second leak path: synthetic data generated by a model that saw the benchmark, or scraped Q&A that includes it. [[contamination-detection|Contamination detection]] tries to catch this.' },
        { t: 'table', head: ['Method', 'Idea', 'Limitation'], rows: [
          ['N-gram overlap', 'Flag test items sharing long n-grams (GPT-3 used 13-grams) with training documents', 'Needs access to training data; misses paraphrases'],
          ['Canary strings', 'Benchmarks embed a unique GUID; if the model can complete it, the benchmark was in training', 'Only works for benchmarks that ship canaries'],
          ['Membership inference', 'Test items have suspiciously low perplexity (e.g. Min-K% prob) vs fresh items of similar difficulty', 'Statistical, noisy per item'],
          ['Perturbed re-test', 'Re-score on rephrased questions, shuffled options or new numbers (e.g. GSM-style rewrites)', 'Costs effort to build; difficulty may change'],
          ['Guided completion', 'Prompt with the first half of a test item and see if the model reproduces the rest verbatim', 'Only detects strong memorisation'],
        ]},
        { t: 'code', lang: 'python', title: 'ngram_overlap.py — check your fine-tuning set against a benchmark', x: `import re
from datasets import load_dataset

def ngrams(text: str, n: int = 13):
    words = re.findall(r"\\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

# 1) index every 13-gram in the training data (use a Bloom filter / suffix array at scale)
train = load_dataset("json", data_files="my_sft_data.jsonl", split="train")
train_grams = set()
for ex in train:
    train_grams |= ngrams(ex["prompt"] + " " + ex["response"])

# 2) score each benchmark item by overlap
test = load_dataset("openai/gsm8k", "main", split="test")
flagged = []
for i, ex in enumerate(test):
    g = ngrams(ex["question"])
    if g and len(g & train_grams) / len(g) > 0.5:      # >50% of its 13-grams seen in training
        flagged.append(i)

print(f"{len(flagged)}/{len(test)} test items look contaminated")
# Report the score on the clean subset too — if it is much lower, the headline number is inflated.` },
        { t: 'callout', kind: 'warn', title: 'Decontaminate before you train, not after you publish', x: 'Run overlap checks between your SFT/synthetic data and every benchmark you plan to report, and drop matches before training. Afterwards, the honest move is to report scores on the clean subset alongside the full score.' },

        { t: 'h2', x: 'Evaluation at scale: NeMo Evaluator' },
        { t: 'p', x: 'Running a handful of tasks on one GPU is fine for a lab. For a team that customises models continuously, NVIDIA provides [[nemo-evaluator|NeMo Evaluator]] — a microservice and open-source SDK/launcher that runs academic benchmarks (wrapping harnesses such as lm-evaluation-harness), custom datasets, [[llm-as-judge|LLM-as-a-judge]] and retriever/[[rag|RAG]] evaluations against any model behind an OpenAI-compatible endpoint, such as a [[nim|NIM]] or a [[tensorrt-llm|TensorRT-LLM]] / [[vllm|vLLM]] server. Because it talks to an endpoint, you evaluate exactly what you deploy — including quantization and serving settings.' },
        { t: 'flow', steps: [
          { b: 'Customise', s: 'SFT / LoRA with NeMo or TRL' },
          { b: 'Serve', s: 'Deploy candidate behind an OpenAI-compatible endpoint (NIM, TRT-LLM, vLLM)' },
          { b: 'Evaluate', s: 'NeMo Evaluator / lm-eval: benchmarks + domain set + judge' },
          { b: 'Compare', s: 'Diff against the previous release; gate on thresholds' },
          { b: 'Promote', s: 'Ship or iterate on data' },
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Match benchmark to capability: knowledge → MMLU, commonsense → HellaSwag, science reasoning → ARC, maths → GSM8K, code → HumanEval (pass@k), instruction following → IFEval, multi-turn chat → MT-Bench (judge), misconceptions → TruthfulQA.',
          'Expect a scenario where scores jump suspiciously after fine-tuning: the answer is contamination checking (n-gram overlap, rephrased test sets), not "the model got smarter".',
          'Know that few-shot count, prompt/chat template and normalisation must be held constant for a fair comparison.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'lm-evaluation-harness (`lm_eval` CLI or `simple_evaluate`) is the standard way to produce reproducible benchmark numbers for HF, vLLM and API models.',
          'Benchmarks differ in scoring style: likelihood (MMLU, HellaSwag, ARC), generate-and-check (GSM8K, HumanEval, IFEval) and judge (MT-Bench).',
          'Shots, templates, normalisation, decoding and harness version all move scores — hold them fixed and report them.',
          'Contamination inflates benchmarks; detect it with n-gram overlap, canaries, membership signals and perturbed re-tests, and decontaminate training data up front.',
          'At scale, NeMo Evaluator evaluates the deployed endpoint with benchmarks, custom sets and judges.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'llm-judge',
      title: 'LLM-as-a-judge, human eval, error analysis and RAG evaluation',
      minutes: 24,
      outcomes: [
        'Design pointwise and pairwise LLM-as-a-judge evaluations with rubrics, and mitigate position, verbosity and self-preference bias',
        'Design a human evaluation and measure inter-annotator agreement with Cohen\'s kappa',
        'Run an error-analysis loop: slice, cluster, build a taxonomy, fix, add regression tests',
        'Evaluate RAG systems with faithfulness, answer relevance and context precision/recall',
        'Build a domain evaluation set and wire evaluation into CI',
      ],
      blocks: [
        { t: 'lede', x: 'For open-ended generation there is no single reference answer, so overlap metrics fail. The modern toolkit is a combination: an LLM judge for scale, human evaluation to keep the judge honest, error analysis to turn scores into fixes, and stage-specific metrics for [[rag|RAG]]. This lesson shows how to build each so you can trust the result.' },

        { t: 'h2', x: 'LLM-as-a-judge: pointwise vs pairwise' },
        { t: 'vs', cards: [
          { h: 'Pointwise (absolute score)', items: [
            'Judge sees one answer + rubric, outputs e.g. 1–5 or pass/fail',
            'Cheap: N answers → N calls; easy to track over time',
            'Scores drift with judge version and prompt; poorly calibrated in the middle of the scale',
            'Best with a reference answer or binary, checkable criteria',
          ]},
          { h: 'Pairwise (A vs B)', items: [
            'Judge sees two answers and picks the better (or tie)',
            'More reliable for subtle quality differences; matches how humans compare',
            'Needs a baseline; cost grows with comparisons; gives win-rate, not absolute quality',
            'Must swap order to cancel position bias',
          ]},
        ]},
        { t: 'p', x: 'Good judge prompts share a pattern: a clear **rubric** with defined levels, an optional **reference answer** (big gains for maths and factual questions, where judges are weak), a request to **reason before scoring**, a strict **output format**, and temperature 0. Binary or 3-level criteria ("Does the answer cite the policy section? yes/no") are far more reliable than a 10-point "quality" score.' },
        { t: 'code', lang: 'python', title: 'pairwise_judge.py — swap order to cancel position bias', x: `import json
from openai import OpenAI   # works with any OpenAI-compatible endpoint: NIM, vLLM, TRT-LLM, hosted APIs

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed-for-local")
JUDGE = "meta/llama-3.1-70b-instruct"   # ideally a different family from the models being judged

RUBRIC = """You are grading two answers to the same user question.
Criteria, in priority order: (1) factual correctness, (2) completeness w.r.t. the question,
(3) clarity. Length is NOT a criterion; do not prefer the longer answer.
Think step by step, then output JSON: {"reason": "...", "winner": "A" | "B" | "tie"}"""

def judge_once(question, a, b):
    msg = f"Question:\\n{question}\\n\\n[Answer A]\\n{a}\\n\\n[Answer B]\\n{b}"
    r = client.chat.completions.create(
        model=JUDGE, temperature=0,
        messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": msg}],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)["winner"]

def judge_pair(question, ans_model, ans_baseline):
    first = judge_once(question, ans_model, ans_baseline)      # model shown as A
    second = judge_once(question, ans_baseline, ans_model)     # model shown as B
    if first == "A" and second == "B":
        return "win"
    if first == "B" and second == "A":
        return "loss"
    return "tie"            # inconsistent verdicts = position bias or a genuine tie

# win_rate = (wins + 0.5 * ties) / n   — report it with a confidence interval` },
        { t: 'table', head: ['Bias', 'What happens', 'Mitigation'], rows: [
          ['[[position-bias|Position bias]]', 'Judge favours the first (or second) answer', 'Evaluate both orders; count only consistent wins; randomise order'],
          ['Verbosity bias', 'Longer, more detailed-looking answers win', 'Rubric says length is not a criterion; length-controlled win rates; compare at similar lengths'],
          ['Self-preference', 'Judge prefers outputs from its own model family / style', 'Use a judge from a different family, or a panel of judges'],
          ['Weak at maths / code', 'Judge accepts plausible but wrong reasoning', 'Provide reference answers; use execution or exact match where possible'],
          ['Prompt sensitivity', 'Small wording changes flip verdicts', 'Fix and version the judge prompt; validate it against human labels'],
        ]},
        { t: 'callout', kind: 'warn', title: 'A judge is a model — evaluate it', x: 'Before trusting a judge, label 100–200 items by hand and measure agreement between judge and humans (accuracy or Cohen\'s κ). The MT-Bench study found strong judges agree with humans about as often as humans agree with each other (~80% on pairwise preferences) — but that is a property of that judge, prompt and task, not a guarantee for yours.' },

        { t: 'h2', x: 'Human evaluation that you can trust' },
        { t: 'p', x: '[[human-eval|Human evaluation]] remains the reference for helpfulness, tone and domain correctness. Make it reliable by design: write guidelines with examples for each score level, blind annotators to which model produced which answer, randomise order, use pairwise comparisons or short (3–5 point) scales, collect at least 2–3 labels per item for a calibration subset, and measure [[inter-annotator-agreement|inter-annotator agreement]].' },
        { t: 'specs', rows: [
          ['Cohen\'s κ (2 raters)', 'κ = (p_o − p_e) / (1 − p_e); p_o = observed agreement, p_e = chance agreement from each rater\'s label frequencies'],
          ['Worked example', 'Raters agree on 80 of 100 items (p_o = 0.80); both say "good" 50% of the time → p_e = 0.5·0.5 + 0.5·0.5 = 0.50 → κ = 0.30/0.50 = **0.60**'],
          ['Interpretation (Landis–Koch)', '< 0.2 slight · 0.21–0.40 fair · 0.41–0.60 moderate · 0.61–0.80 substantial · > 0.80 near-perfect'],
          ['More than 2 raters', 'Fleiss\' κ; Krippendorff\'s α also handles missing labels and ordinal scales'],
          ['Low agreement means', 'Ambiguous guidelines or an inherently subjective task — revise the rubric before collecting more labels'],
        ]},
        { t: 'callout', kind: 'trap', title: 'Raw agreement is not agreement', x: 'Two raters who each label 90% of items "acceptable" will agree ~82% of the time by pure chance (0.9² + 0.1²). Always report a chance-corrected statistic such as κ, not raw percent agreement.' },

        { t: 'h2', x: 'Error analysis: from a score to a fix' },
        { t: 'p', x: 'An aggregate score tells you *that* something is wrong, not *what*. [[error-analysis|Error analysis]] is the disciplined process of reading failures until you can name them, count them and fix the biggest one.' },
        { t: 'flow', steps: [
          { b: 'Collect', s: 'Log inputs, outputs, scores and metadata (lm-eval --log_samples, judge reasons, production traces)' },
          { b: 'Slice', s: 'Break the metric down by topic, length, language, template, difficulty — find where it is low' },
          { b: 'Read & cluster', s: 'Read 50–100 failures; cluster by embedding or by hand into recurring patterns' },
          { b: 'Taxonomy', s: 'Name categories (format error, refusal, wrong retrieval, arithmetic, hallucinated entity…) and count them' },
          { b: 'Fix the biggest', s: 'Prompt, data, retrieval or training change targeted at the top category' },
          { b: 'Regress', s: 'Add the failing examples to the eval set so the bug cannot silently return' },
        ]},
        { t: 'p', x: 'Typical discovery: "accuracy fell 6 points after fine-tuning" becomes "70% of new failures are on inputs longer than 2k tokens, because our SFT data was truncated at 1,024 tokens". That points at a data fix, not a hyperparameter search.' },

        { t: 'h2', x: 'Evaluating RAG: separate retrieval from generation' },
        { t: 'p', x: 'A [[rag|RAG]] answer can be wrong because the retriever fetched the wrong chunks or because the generator ignored or contradicted good chunks. Evaluate each stage. [[ragas|RAGAS]] popularised four reference-light metrics, mostly computed with an LLM judge:' },
        { t: 'table', head: ['Metric', 'Question it answers', 'Stage diagnosed', 'Needs reference answer?'], rows: [
          ['[[faithfulness|Faithfulness]]', 'Are the answer\'s claims supported by the retrieved context? (supported claims ÷ total claims)', 'Generator ([[hallucination]])', 'No'],
          ['Answer relevance', 'Does the answer address the question? (e.g. similarity between the question and questions regenerated from the answer)', 'Generator', 'No'],
          ['Context precision', 'Are the relevant chunks ranked near the top of what was retrieved?', 'Retriever / reranker', 'Usually (or judge-labelled relevance)'],
          ['Context recall', 'Does the retrieved context contain everything needed for the reference answer?', 'Retriever', 'Yes'],
        ]},
        { t: 'code', lang: 'python', title: 'faithfulness.py — the idea behind RAGAS faithfulness, in plain code', x: `import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
JUDGE = "meta/llama-3.1-70b-instruct"

def ask(prompt):
    r = client.chat.completions.create(model=JUDGE, temperature=0,
        messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    return json.loads(r.choices[0].message.content)

def faithfulness(answer: str, context: str) -> float:
    claims = ask(f'Split this answer into atomic factual claims. JSON: {{"claims": [...]}}\\n\\n{answer}')["claims"]
    if not claims:
        return 1.0
    verdicts = ask(
        "For each claim, answer 1 if it can be inferred from the context, else 0.\\n"
        f"Context:\\n{context}\\n\\nClaims:\\n{json.dumps(claims)}\\n"
        'JSON: {"verdicts": [0 or 1, ...]}'
    )["verdicts"]
    return sum(verdicts) / len(claims)

ctx = "The warranty covers manufacturing defects for 24 months from purchase."
print(faithfulness("The warranty lasts 24 months and covers accidental damage.", ctx))  # ~0.5` },
        { t: 'callout', kind: 'tip', title: 'Reading RAG metrics together', x: [
          'Low context recall, high faithfulness → the generator is honest but starved: fix chunking, embeddings, top-k or add a reranker.',
          'High context recall, low faithfulness → good documents, but the model hallucinates or uses parametric knowledge: tighten the prompt ("answer only from the context"), fine-tune for grounding, or lower temperature.',
        ]},

        { t: 'h2', x: 'Building a domain eval set and running it in CI' },
        { t: 'p', x: 'Public benchmarks rarely match your use case. The most valuable evaluation asset you can build is a [[eval-set|domain evaluation set]]: 200–1,000 real (anonymised) inputs, each with a reference answer or a checkable rubric, stratified across your traffic — common cases, hard cases, adversarial and safety cases, and every production failure you have fixed. Version it like code, and never let it leak into training data.' },
        { t: 'ul', items: [
          '**Tiered suites**: a fast smoke suite (~50 items, deterministic checks, minutes) on every pull request; the full suite with judges and benchmarks nightly or before release.',
          '**Prefer deterministic checks** (exact match, JSON-schema validity, regex, unit tests) and use LLM judges only where needed — they cost money and add variance.',
          '**Gate on deltas with uncertainty**: fail the build if a metric drops by more than its noise band (e.g. 2 standard errors or a bootstrap CI), not on any decrease.',
          '**Pin everything**: model version, judge model and prompt, decoding params, dataset version, harness version — store them next to every result.',
          '**Track over time** on a dashboard so you can see slow drift between releases.',
        ]},
        { t: 'code', lang: 'python', title: 'test_eval_gate.py — a pytest regression gate', x: `import json, math, pytest

BASELINE = json.load(open("eval/baseline.json"))      # committed results of the current release
CANDIDATE = json.load(open("eval/candidate.json"))    # produced by the CI eval job

def stderr(p, n):
    return math.sqrt(p * (1 - p) / n)

@pytest.mark.parametrize("metric", ["domain_em", "json_valid", "judge_win_rate"])
def test_no_significant_regression(metric):
    b, c, n = BASELINE[metric], CANDIDATE[metric], CANDIDATE["n_items"]
    assert c >= b - 2 * stderr(b, n), f"{metric}: {c:.3f} < baseline {b:.3f} beyond noise"

def test_safety_floor():
    assert CANDIDATE["refusal_on_harmful"] >= 0.98     # hard floor, no tolerance` },
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Judge questions: pairwise vs pointwise, and which bias a mitigation addresses (swap order → position bias; length control → verbosity bias; different judge family → self-preference).',
          'RAG questions: map a symptom to faithfulness, answer relevance, context precision or context recall, and to the retriever vs the generator.',
          'Process questions: "scores dropped, what first?" → error analysis (slice and read failures); "how do we keep quality as we iterate?" → versioned domain eval set + automated CI gates.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'LLM judges scale open-ended evaluation; use rubrics, references, temperature 0, order swapping and a different model family, and validate against human labels.',
          'Human evaluation needs guidelines, blinding and chance-corrected agreement (Cohen\'s κ = (p_o − p_e)/(1 − p_e)).',
          'Error analysis — slice, read, cluster, taxonomy, fix the biggest category, add regressions — turns scores into actions.',
          'Evaluate RAG per stage: context precision/recall for retrieval, faithfulness and answer relevance for generation.',
          'A versioned in-domain eval set with tiered CI gates on statistically meaningful drops is the backbone of scalable evaluation.',
        ]},
      ],
    },
  ],

  quiz: [
    { id: 'e01', domain: 'evaluation',
      scenario: 'You fine-tune a 0.5B model on 3,000 support conversations for 6 epochs. Training loss falls steadily from 1.9 to 0.4; validation loss reaches 1.2 at epoch 2 and climbs to 1.7 by epoch 6.',
      q: 'What is the best action?',
      opts: ['Increase the learning rate so validation loss catches up', 'Keep training; validation loss will fall again once training loss reaches zero', 'Use the epoch-2 checkpoint and reduce epochs or regularise, because the model is overfitting', 'Switch the metric to perplexity, which is not affected by overfitting'],
      a: 2,
      why: 'Falling training loss with rising validation loss after a minimum is classic [[overfitting]]. Early-stop at the validation minimum (epoch 2) and reduce epochs, LR or adapter capacity, or add data.',
      distract: 'A higher LR makes overfitting and forgetting worse. Validation loss does not recover by driving training loss to zero — that is memorisation. Perplexity is exp(loss), so it shows exactly the same overfitting.' },
    { id: 'e02', domain: 'evaluation',
      q: 'Model X (32k vocabulary) has WikiText perplexity 9.1; model Y (128k vocabulary) has 11.4 on the same text. What can you conclude?',
      opts: ['Nothing directly — per-token perplexity is not comparable across tokenizers; compare bits per byte instead', 'X is the better language model because its perplexity is lower', 'Y is better because a larger vocabulary always means lower loss per character', 'X is better, but only if both were evaluated with a stride equal to the window size'],
      a: 0,
      why: '[[perplexity|Perplexity]] is per token, and different tokenizers split the same text into different numbers of tokens. Normalising total NLL by bytes or characters gives a comparable number.',
      distract: 'Lower perplexity only means better when the tokenizer is shared. Vocabulary size does not guarantee anything about per-character loss. Stride choice matters for correctness but does not fix the tokenizer mismatch.' },
    { id: 'e03', domain: 'evaluation',
      q: 'For a code-generation model you sample n = 20 completions per HumanEval problem and 4 pass the tests. What is the unbiased pass@1 estimate for that problem?',
      opts: ['0.59, computed as 1 − (16/20)^4 with replacement', '0.20, since pass@1 = 1 − C(16,1)/C(20,1) = c/n', '0.80, because 16 of 20 samples failed', '1.0, because at least one sample passed'],
      a: 1,
      why: 'pass@k = 1 − C(n−c, k)/C(n, k). For k = 1 this reduces to c/n = 4/20 = 0.20.',
      distract: '1 − 0.8^4 ≈ 0.59 is a biased with-replacement estimate of pass@4, not pass@1. 0.80 is the failure rate. 1.0 would be pass@20, not pass@1.' },
    { id: 'e04', domain: 'evaluation',
      scenario: 'Your team reports that a newly fine-tuned 1B model jumped from 41% to 63% on GSM8K, while its MMLU and ARC scores are unchanged. The SFT data includes 40k synthetic maths problems generated by a large public model.',
      q: 'What should you check first?',
      opts: ['Whether ARC was run with acc_norm instead of acc', 'Whether the learning rate warmup was long enough', 'Whether HellaSwag also improved, which would confirm genuine reasoning gains', 'Whether the synthetic maths data overlaps GSM8K test items, using n-gram overlap and a rephrased test set'],
      a: 3,
      why: 'An isolated, large jump on one benchmark after training on synthetic data from a model that may have seen that benchmark is a classic [[contamination-detection|contamination]] signal. Check n-gram overlap and re-test on rephrased problems.',
      distract: 'ARC normalisation is unrelated to the GSM8K jump. Warmup length does not explain a benchmark-specific gain. HellaSwag measures commonsense completion, not maths, so it neither confirms nor rules out contamination.' },
    { id: 'e05', domain: 'evaluation',
      q: 'An LLM judge comparing two answers prefers whichever answer is shown first in 68% of cases, even when the two answers are identical. Which mitigation directly addresses this?',
      opts: ['Raise the judge temperature to 1.0 to add diversity', 'Ask the judge for a 1–10 score instead of a preference', 'Run each comparison in both orders and count only consistent verdicts', 'Use a longer rubric that lists more criteria'],
      a: 2,
      why: 'This is [[position-bias|position bias]]. Evaluating both orders and treating inconsistent verdicts as ties cancels it; randomising order also helps.',
      distract: 'Higher temperature adds noise, not fairness. Switching to pointwise scores avoids the pair format but loses pairwise sensitivity and does not correct the measured bias in your pairwise data. A longer rubric does not remove order effects.' },
    { id: 'e06', domain: 'evaluation',
      scenario: 'A RAG assistant over HR policies gives fluent answers. Evaluation shows context recall 0.45 and faithfulness 0.93.',
      q: 'Where should you focus improvements?',
      opts: ['Retrieval: chunking, embeddings, top-k or a reranker, because the needed information often is not retrieved', 'The generator prompt, because faithfulness is the main problem', 'The LLM judge, because a faithfulness of 0.93 is impossibly high', 'Decoding temperature, because recall measures sampling diversity'],
      a: 0,
      why: 'High [[faithfulness]] means the model sticks to what it is given; low context recall means what it is given often lacks the information needed. The bottleneck is the retriever.',
      distract: 'Faithfulness is already high, so the generator is not the issue. 0.93 is a plausible faithfulness score. Context recall measures retrieval coverage, not sampling diversity.' },
    { id: 'e07', domain: 'evaluation',
      q: 'Two annotators labelled 200 responses as acceptable/unacceptable and agreed on 180 (90%). Each labelled 90% of responses as acceptable. What does this indicate?',
      opts: ['Excellent reliability, since 90% raw agreement is above any common threshold', 'Agreement barely above chance: p_e = 0.82, so κ ≈ 0.44, and the guidelines need work', 'Perfect reliability, since both raters used the same label distribution', 'Nothing, because agreement cannot be computed for binary labels'],
      a: 1,
      why: 'Chance agreement p_e = 0.9² + 0.1² = 0.82, so κ = (0.90 − 0.82)/(1 − 0.82) ≈ 0.44 — only moderate. Raw agreement is misleading when one label dominates; see [[inter-annotator-agreement|inter-annotator agreement]].',
      distract: 'Raw percent agreement ignores chance. Matching label distributions is exactly what inflates chance agreement. Cohen\'s κ is defined for binary labels.' },
    { id: 'e08', domain: 'evaluation',
      q: 'Your team wants every pull request that changes prompts, adapters or quantization settings to be checked for quality regressions without slowing development. What design fits best?',
      opts: ['Run the full MMLU, HellaSwag and MT-Bench suite on every commit and block on any decrease', 'Rely on production user feedback and roll back if complaints rise', 'Run human evaluation of 500 samples for every pull request', 'A versioned domain eval set: a fast deterministic smoke suite per PR, the full suite with judges nightly, gating on drops beyond the noise band'],
      a: 3,
      why: 'Scalable evaluation uses tiers: cheap deterministic checks on every change, expensive judges and benchmarks less often, and statistical thresholds so noise does not block merges. The [[eval-set|domain eval set]] reflects real traffic.',
      distract: 'Full benchmark suites per commit are slow, and blocking on any decrease fails on noise. Production feedback detects regressions only after users are hurt. Human eval per PR does not scale; it is for validating automated metrics.' },
  ],
}
