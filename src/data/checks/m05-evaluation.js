// End-of-lesson comprehension checks for Module 5: Evaluation. Keyed by lesson id.
export default {
  metrics: [
    {
      q: 'A checkpoint has a mean validation cross-entropy of 2.0 nats per token. What is its perplexity?',
      opts: ['About 4.0', 'About 7.4', 'About 2.0', 'About 100'],
      a: 1,
      why: '[[perplexity|Perplexity]] = exp(mean [[cross-entropy]]) = e² ≈ 7.39.',
      distract: '4.0 is 2² (base-2 exponent applied to a natural-log loss). 2.0 is the loss itself. 100 is 10², a base-10 confusion.',
    },
    {
      q: 'Why does a sliding-window perplexity evaluation (stride smaller than the window) give a lower and more faithful number than scoring disjoint chunks?',
      opts: [
        'Because overlapping windows score each token several times and average the results',
        'Because it uses a smaller learning rate during evaluation',
        'Because it removes the brevity penalty applied to short chunks',
        'Because every scored token keeps substantial preceding context instead of restarting at each chunk boundary',
      ],
      a: 3,
      why: 'With disjoint chunks, tokens at the start of each chunk are predicted with almost no context, inflating loss. A sliding window only scores the new tokens of each window, each of which sees up to max_len − stride tokens of history.',
      distract: 'Each token is scored exactly once in the correct implementation (context tokens are masked with −100). Evaluation has no learning rate. The brevity penalty belongs to BLEU, not perplexity.',
    },
    {
      q: 'Which metric is most appropriate for scoring a model that writes Python functions to satisfy a specification?',
      opts: [
        'pass@k, by executing the generated code against unit tests',
        'BLEU against a reference implementation',
        'ROUGE-L against a reference implementation',
        'Perplexity of the reference implementation under the model',
      ],
      a: 0,
      why: 'Code correctness is checkable by execution. [[pass-at-k|pass@k]] measures the probability that at least one of k samples passes the tests.',
      distract: 'BLEU and ROUGE-L reward textual overlap — a correct solution with different variable names scores low and a near-copy with a bug scores high. Reference perplexity measures likelihood, not whether generated code works.',
    },
    {
      q: 'A summary says "Revenue grew 21%" while the reference says "Revenue grew 12%". Which statement about the metrics is true?',
      opts: [
        'BERTScore will flag the error because it compares meaning rather than tokens',
        'ROUGE-2 will be zero because the numbers differ',
        'BLEU, ROUGE and BERTScore will all score it highly, because they measure overlap or similarity rather than factual correctness',
        'Exact match with normalisation will give partial credit for the shared words',
      ],
      a: 2,
      why: 'Overlap and embedding-similarity metrics see almost identical text and reward it. None of them verifies facts; you need a faithfulness or factuality check.',
      distract: 'BERTScore embeddings of "21%" and "12%" are very similar, so the score stays high. ROUGE-2 still matches most bigrams ("revenue grew"). Exact match is binary — it gives 0, not partial credit; token F1 is the partial-credit variant.',
    },
    {
      q: 'In lm-evaluation-harness, what does `acc_norm` change compared with `acc` on a multiple-choice task?',
      opts: [
        'It normalises the prompt text (lower-case, strip punctuation) before scoring',
        'It divides each answer\'s log-likelihood by its length so longer answers are not penalised',
        'It rescales accuracy so random guessing scores 0',
        'It averages accuracy across all few-shot settings',
      ],
      a: 1,
      why: 'Summed log-probabilities get more negative with more tokens. `acc_norm` length-normalises (by bytes in lm-eval) before choosing the best answer — see [[log-likelihood-scoring|log-likelihood scoring]].',
      distract: 'Text normalisation is an exact-match concept, not acc_norm. No chance-correction is applied. Few-shot count is a fixed run setting, not averaged.',
    },
  ],

  benchmarks: [
    {
      q: 'Which benchmark specifically measures whether a model follows verifiable formatting instructions such as "answer in exactly three bullet points"?',
      opts: ['TruthfulQA', 'HellaSwag', 'IFEval', 'MMLU'],
      a: 2,
      why: 'IFEval contains ~500 prompts with ~25 types of automatically verifiable instructions and reports strict/loose accuracy at prompt and instruction level.',
      distract: 'TruthfulQA targets misconceptions. HellaSwag is commonsense sentence completion. [[mmlu|MMLU]] is multiple-choice knowledge.',
    },
    {
      q: 'You run `lm_eval` on your instruct model without `--apply_chat_template` and get much lower GSM8K than the model card reports. What is the most likely explanation?',
      opts: [
        'The model was evaluated without its chat template (and maybe with different shots), so the prompt format differs from the reported setting',
        'lm-eval cannot evaluate instruct models',
        'GSM8K must always be evaluated with log-likelihood scoring',
        'The batch size was set to auto, which changes the model weights',
      ],
      a: 0,
      why: 'Prompt format, chat template and [[few-shot-eval|shot count]] strongly affect generative benchmarks. Match the reported setting before comparing numbers.',
      distract: 'lm-eval supports chat models through `--apply_chat_template`. GSM8K is generate-then-exact-match, not log-likelihood. Batch size does not change weights; at most it causes tiny numerical differences.',
    },
    {
      q: 'What does the `--limit 50` flag do, and when should you use it?',
      opts: [
        'It sets the number of few-shot examples to 50, for long-context models',
        'It caps generation at 50 new tokens, for multiple-choice tasks',
        'It limits GPU memory to 50%, for shared machines',
        'It evaluates only the first 50 items of each task — for smoke tests, never for reported results',
      ],
      a: 3,
      why: '`--limit` subsamples each task so you can verify the pipeline quickly. Scores on a 50-item subset have huge standard errors and are not comparable to full runs.',
      distract: 'Shots are set with `--num_fewshot`. Generation length is controlled in task configs/gen_kwargs. lm-eval has no GPU-memory percentage flag of that kind.',
    },
    {
      q: 'You have access to your SFT dataset but not to the base model\'s pretraining corpus. Which contamination check can you run on your own data before training?',
      opts: [
        'Compare the base model\'s MMLU score with the leaderboard',
        'Compute long n-gram (e.g. 13-gram) overlap between your SFT examples and each benchmark\'s test items, and drop matches',
        'Check whether the benchmark uses acc_norm',
        'Evaluate at temperature 0 so memorised answers cannot appear',
      ],
      a: 1,
      why: 'N-gram overlap between your training data and the test set is the standard [[contamination-detection|decontamination]] step — and you control that data, so you can remove matches before training.',
      distract: 'Leaderboard comparison says nothing about your SFT data. acc_norm is a scoring choice. Greedy decoding makes memorised answers more likely to appear, not less.',
    },
    {
      q: 'Why does NVIDIA position NeMo Evaluator to run against a model served behind an OpenAI-compatible endpoint (e.g. a NIM)?',
      opts: [
        'So the evaluation covers exactly what is deployed, including quantization and serving configuration, and scales across many models and suites',
        'Because benchmarks cannot be computed from local Hugging Face checkpoints',
        'Because OpenAI-compatible endpoints return log-probabilities that are more accurate than local ones',
        'Because endpoints remove the need for few-shot examples',
      ],
      a: 0,
      why: '[[nemo-evaluator|NeMo Evaluator]] evaluates the deployed artefact — the same container, precision and settings users hit — and runs benchmarks, custom sets and judges at scale.',
      distract: 'lm-eval and NeMo tooling can run local checkpoints too. Endpoint log-probabilities are not inherently more accurate. Few-shot settings are part of the task, regardless of serving method.',
    },
  ],

  'llm-judge': [
    {
      q: 'Which judge setup is most reliable for detecting a small quality difference between a new model and the current production model?',
      opts: [
        'Pointwise 1–10 scores for each model at temperature 1.0',
        'Pointwise scores from the new model judging its own answers',
        'BLEU between the two models\' answers',
        'Pairwise comparison against the production model\'s answers, in both orders, with a rubric and a judge from a different model family',
      ],
      a: 3,
      why: 'Pairwise judging is more sensitive to subtle differences; evaluating both orders cancels [[position-bias|position bias]], a rubric reduces variance, and a different-family judge avoids self-preference.',
      distract: 'High-temperature pointwise scores are noisy and coarse. Self-judging introduces self-preference bias. BLEU between two answers measures similarity, not quality.',
    },
    {
      q: 'Your judge consistently prefers longer answers even when they add only filler. What is this called and how do you mitigate it?',
      opts: [
        'Position bias — swap the answer order',
        'Verbosity bias — state in the rubric that length is not a criterion and compare length-controlled win rates',
        'Self-preference — use the same model as judge and candidate',
        'Contamination — remove long answers from training data',
      ],
      a: 1,
      why: 'Favouring length independent of quality is verbosity bias. Rubric instructions and length-controlled metrics counteract it.',
      distract: 'Position bias concerns order, not length. Using the same model as judge and candidate causes self-preference rather than fixing anything. Contamination is about test data in training.',
    },
    {
      q: 'Two raters agree on 70% of items; chance agreement is 50%. What is Cohen\'s kappa?',
      opts: ['0.70', '0.20', '0.40', '0.35'],
      a: 2,
      why: 'κ = (p_o − p_e)/(1 − p_e) = (0.70 − 0.50)/(0.50) = 0.40 — "fair to moderate" [[inter-annotator-agreement|agreement]].',
      distract: '0.70 is raw agreement. 0.20 is p_o − p_e without dividing by 1 − p_e. 0.35 is 0.70 × 0.50, which has no meaning here.',
    },
    {
      q: 'After a fine-tune, your domain accuracy drops from 82% to 76%. What is the best first step in an error-analysis workflow?',
      opts: [
        'Slice the results by metadata and read the new failures to find and count recurring error categories',
        'Launch a hyperparameter sweep over learning rate and epochs',
        'Replace the eval set with a public benchmark',
        'Increase the LLM judge temperature to reduce variance',
      ],
      a: 0,
      why: '[[error-analysis|Error analysis]] starts by locating and naming the failures; only then do you know whether the fix is data, prompt, retrieval or training.',
      distract: 'A sweep before diagnosis may waste compute on the wrong cause. Public benchmarks do not reflect your domain. Higher temperature increases variance.',
    },
    {
      q: 'In RAGAS-style RAG evaluation, which metric requires a reference (ground-truth) answer?',
      opts: ['Faithfulness', 'Answer relevance', 'Judge agreement', 'Context recall'],
      a: 3,
      why: 'Context recall asks whether the retrieved context contains the information in the reference answer, so it needs that reference. [[faithfulness|Faithfulness]] and answer relevance are computed from question, context and answer alone.',
      distract: 'Faithfulness checks answer claims against the retrieved context. Answer relevance compares the answer with the question. Judge agreement is not a RAGAS metric.',
    },
  ],
}
