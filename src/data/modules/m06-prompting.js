export default {
  id: 'prompting',
  n: 6,
  domain: 'prompting',
  weight: 13,
  title: 'Prompt Engineering',
  blurb: 'Get the most out of a model without touching its weights: zero/few-shot prompting, system prompts and chat templates, chain-of-thought and self-consistency, every decoding knob from temperature to min-p, schema-constrained JSON, and a working RAG pipeline with hybrid search and reranking.',
  hours: '4–5 h',
  lessons: [
    /* ------------------------------------------------------------------ */
    {
      id: 'prompting-basics',
      title: 'Zero-, one- and few-shot prompting, system prompts and in-context learning',
      minutes: 20,
      outcomes: [
        'Explain how in-context learning conditions a frozen model without any gradient update',
        'Write zero-, one- and few-shot prompts and choose, balance and order the demonstrations',
        'Structure a chat request with system, user and assistant roles and render it with the model\'s chat template',
        'Distinguish base from instruct models and write a completion-style prompt for a base model',
        'Call a locally served model through the transformers pipeline and an OpenAI-compatible client',
      ],
      blocks: [
        { t: 'lede', x: 'Prompting is the cheapest adaptation method you have: no data pipeline, no GPU hours, instant iteration. It is also 13% of NCP-GENL. This lesson covers the mechanics — what [[in-context-learning|in-context learning]] really does, how to build [[zero-shot|zero-shot]] and [[few-shot|few-shot]] prompts that work, how roles and the [[chat-template|chat template]] turn a message list into [[token|tokens]], and why a base model needs a completely different prompt from an instruct model.' },

        { t: 'h2', x: 'In-context learning: learning without gradients' },
        { t: 'p', x: 'An [[llm|LLM]] is a next-token predictor. When you put examples of a task into the prompt, nothing about the weights changes — the examples simply become part of the conditioning context, and the most probable continuation is "more of the same pattern". That is [[in-context-learning|in-context learning]] (ICL): the model infers the task from the prompt at [[inference|inference]] time.' },
        { t: 'p', x: 'Research on ICL (Min et al., 2022) found something counter-intuitive: demonstrations help mostly by showing the **format**, the **label space** and the **input distribution**. Randomly shuffling the gold labels in a few-shot prompt often hurts far less than you would expect. Practical consequence: get the format and the set of allowed answers exactly right first; then worry about which examples to use.' },
        { t: 'ul', items: [
          'ICL lives inside the [[context-window|context window]] — every demonstration costs tokens on every request, which costs latency and money.',
          'ICL is ephemeral: nothing persists between requests. If you need the same 40 examples on every call, that is a signal to [[fine-tuning|fine-tune]] instead.',
          'ICL ability grows with model size. A 135M model barely does it; a 7B+ instruct model does it well. Expect weaker few-shot gains on the tiny models you trained earlier in the course.',
        ]},

        { t: 'h2', x: 'Zero-, one- and few-shot' },
        { t: 'table', head: ['Technique', 'What the prompt contains', 'Best when', 'Cost / risk'], rows: [
          ['[[zero-shot|Zero-shot]]', 'An instruction only: "Classify the sentiment of this review as positive or negative."', 'Instruct models on common tasks; you want the shortest prompt', 'Model may pick its own output format or label wording'],
          ['One-shot', 'Instruction + a single worked example', 'You mainly need to pin down the output format', 'Model may over-copy the one example (length, wording, label)'],
          ['[[few-shot|Few-shot]]', 'Instruction + k examples (typically 3–8)', 'Unusual formats, domain labels, base models, small models', 'More tokens per request; sensitive to choice and order of examples'],
        ]},
        { t: 'code', lang: 'python', title: 'A few-shot classifier as a chat request', x: `# Few-shot via the messages list: each demonstration is a user/assistant turn pair.
# This teaches the exact output format ("label only") better than any instruction.
messages = [
    {"role": "system", "content": "You label IT support tickets. Answer with exactly one label: "
                                  "HARDWARE, SOFTWARE, NETWORK or ACCESS."},
    # --- demonstrations (balanced labels, same format every time) ---
    {"role": "user", "content": "Ticket: My laptop fan is grinding and it shuts down."},
    {"role": "assistant", "content": "HARDWARE"},
    {"role": "user", "content": "Ticket: VPN drops every 10 minutes from home."},
    {"role": "assistant", "content": "NETWORK"},
    {"role": "user", "content": "Ticket: I can't log in to the HR portal since my password reset."},
    {"role": "assistant", "content": "ACCESS"},
    {"role": "user", "content": "Ticket: Excel crashes when I open a pivot table."},
    {"role": "assistant", "content": "SOFTWARE"},
    # --- the real query ---
    {"role": "user", "content": "Ticket: The printer on floor 3 shows 'paper jam' but has no paper in it."},
]` },

        { t: 'h3', x: 'Choosing and ordering demonstrations' },
        { t: 'ul', items: [
          '**Relevance beats quantity.** Retrieve the k most similar labelled examples for each query (embed the query, nearest-neighbour search over your example pool) — "dynamic few-shot" usually beats a fixed list.',
          '**Cover the label space and balance it.** If 4 of 5 examples are POSITIVE, the model drifts toward POSITIVE (majority-label bias).',
          '**Mind the order.** Models over-weight the last examples (recency bias). Shuffle or interleave labels; never put all of one class at the end.',
          '**Keep the format identical** across examples — same delimiters, same casing, same answer position. The model copies inconsistency too.',
          '**Include hard cases** close to the decision boundary rather than only obvious ones.',
          '**Stop adding examples** when a held-out eval set stops improving — usually well before you run out of context.',
        ]},
        { t: 'callout', kind: 'trap', title: 'Few-shot is not free accuracy', x: [
          'Order and label balance can swing few-shot accuracy by many points on the same model. If a scenario says "results changed a lot when we reordered the examples", the answer is about recency / majority-label bias, not about the model being broken.',
          'And every example is re-processed on every call — 8 examples × 150 tokens = 1,200 extra prompt tokens per request.',
        ]},

        { t: 'h2', x: 'System prompts, roles and the chat template' },
        { t: 'p', x: 'Instruct models are trained on conversations with roles. The [[system-prompt|system prompt]] sets persistent behaviour (persona, rules, output format, safety constraints); `user` turns carry the request; `assistant` turns carry the model\'s earlier replies — or demonstrations you wrote. Some APIs add a `tool` role for function results.' },
        { t: 'flow', steps: [
          { b: 'Message list', s: '[{role: system}, {role: user}, …] — what your code builds' },
          { b: 'Chat template', s: 'A Jinja template shipped in the tokenizer config wraps each turn in the model\'s special tokens' },
          { b: 'Generation prompt', s: 'add_generation_prompt=True appends the "assistant is speaking" header' },
          { b: 'Token IDs', s: 'The [[tokenizer|tokenizer]] encodes the rendered string; the model continues it' },
        ]},
        { t: 'code', lang: 'python', title: 'See exactly what the model receives', x: `from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [
    {"role": "system", "content": "You are a concise assistant. Answer in one sentence."},
    {"role": "user", "content": "What is a KV cache?"},
]
# tokenize=False returns the rendered string so you can inspect it
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
# <|im_start|>system
# You are a concise assistant. Answer in one sentence.<|im_end|>
# <|im_start|>user
# What is a KV cache?<|im_end|>
# <|im_start|>assistant
#   <- the model generates from here and should end with <|im_end|>` },
        { t: 'callout', kind: 'warn', title: 'Never hand-roll the template', x: 'Every model family uses different special tokens (ChatML `<|im_start|>`, Llama 3 `<|start_header_id|>`, Gemma `<start_of_turn>` …). Building the string yourself — or using the wrong template — silently degrades quality: the model sees a format it was never trained on. Always use `apply_chat_template`, or a server (vLLM, NIM) that applies it for you on `/v1/chat/completions`. If you fine-tuned with a template, serve with the **same** one.' },

        { t: 'h2', x: 'Base models vs instruct models' },
        { t: 'vs', cards: [
          { h: 'Base (pretrained) model', items: [
            'Trained only on next-token prediction over raw text',
            'Does not "follow instructions" — it continues documents',
            'Prompt it as the beginning of a document whose natural continuation is your answer',
            'Few-shot completion patterns work well ("Q: … A: … Q: … A:")',
            'No chat template; use the plain completions endpoint',
          ]},
          { h: 'Instruct / chat model', items: [
            'Base model + [[sft|SFT]] on instruction data (+ often [[rlhf|RLHF]] / DPO)',
            'Follows instructions zero-shot; respects a system prompt',
            'Expects its chat template with roles',
            'Use `/v1/chat/completions` or `apply_chat_template`',
            'Usually what you deploy to users',
          ]},
        ]},
        { t: 'code', lang: 'python', title: 'Completion-style prompt for a base model', x: `from transformers import pipeline

# A BASE model: no chat template, no instruction following. Give it a document to continue.
gen = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B", device_map="auto")

prompt = """Translate English to French.

English: Where is the train station?
French: Où est la gare ?

English: I would like a coffee, please.
French: Je voudrais un café, s'il vous plaît.

English: The meeting starts at nine.
French:"""

out = gen(prompt, max_new_tokens=20, do_sample=False, return_full_text=False)
print(out[0]["generated_text"].split("\\n")[0])   # cut at the newline: the pattern's natural stop` },
        { t: 'p', x: 'Asking the same base model "Please translate \'The meeting starts at nine\' into French" often yields a continuation of the request — more questions, a forum post — rather than an answer. The few-shot document above makes the answer the most probable next text.' },

        { t: 'h2', x: 'Writing instructions that work' },
        { t: 'table', head: ['Weak prompt', 'Better', 'Why'], rows: [
          ['"Summarise this."', '"Summarise the incident report below in 3 bullet points for an on-call engineer. Include root cause and fix."', 'States audience, length, format and required content'],
          ['"Don\'t be verbose."', '"Answer in at most 2 sentences."', 'Positive, measurable instructions beat vague negatives'],
          ['Instructions mixed into pasted text', 'Put data inside clear delimiters: `<document>…</document>` or triple quotes', 'The model can tell instructions from data; reduces prompt-injection risk'],
          ['"Return JSON."', 'Return JSON matching `{"label": string, "confidence": number}` — or use constrained decoding', 'Format is specified, not guessed (see next lesson)'],
          ['One giant prompt doing 5 tasks', 'Chain smaller prompts: extract → classify → draft', 'Each step is simpler, testable and cheaper to debug'],
        ]},
        { t: 'p', x: 'A reliable skeleton: **role** (who the model is) → **task** → **context/data in delimiters** → **constraints** (length, tone, what not to do) → **output format** → optional **examples**. Put long documents first and the question last — models attend well to the end of the prompt.' },

        { t: 'h2', x: 'Calling a local model the way production does' },
        { t: 'p', x: 'Engines such as [[vllm|vLLM]] and [[nim|NVIDIA NIM]] expose an [[openai-api|OpenAI-compatible API]], so the same client code works against a laptop, a GPU server or a hosted endpoint — only `base_url` and the model name change.' },
        { t: 'code', lang: 'bash', title: 'Start a local OpenAI-compatible server (any GPU, ~2 GB for 0.5B)', x: `pip install vllm openai
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --max-model-len 4096` },
        { t: 'code', lang: 'python', title: 'The same few-shot request through the OpenAI client', x: `from openai import OpenAI

# vLLM / NIM ignore the key unless you configured one, but the client requires a value
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=messages,          # the few-shot list from above
    temperature=0,              # deterministic: classification wants the top label
    max_tokens=5,
)
print(resp.choices[0].message.content)   # -> HARDWARE
print(resp.usage)                        # prompt_tokens shows what your demonstrations cost` },
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Expect short scenarios: "the model ignores the requested format" → add few-shot examples or constrain decoding; "a base model rambles instead of answering" → use a completion-style few-shot prompt or switch to the instruct variant; "accuracy changed when examples were reordered" → order / label-balance bias.',
          'Know the definitions cold: zero-shot = instruction only, one-shot = one example, few-shot = several; none of them updates weights. Few-shot ≠ fine-tuning.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'In-context learning conditions a frozen model; demonstrations mainly teach format, label space and input distribution.',
          'Choose relevant, balanced, consistently formatted examples and watch order effects; stop when held-out accuracy plateaus.',
          'System prompt = persistent rules; the chat template turns roles into the model\'s special tokens — always use the model\'s own template.',
          'Base models continue documents — prompt them completion-style. Instruct models follow instructions with their chat template.',
          'Clear prompts state role, task, delimited data, constraints and output format.',
          'OpenAI-compatible servers (vLLM, NIM) let one client serve every environment.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'reasoning-decoding',
      title: 'Chain-of-thought, self-consistency and decoding controls (temperature, top-k/top-p, structured output)',
      minutes: 25,
      outcomes: [
        'Apply zero-shot and few-shot chain-of-thought, and improve answers with self-consistency voting',
        'Compare tree-of-thought, ReAct and reasoning models and pick one for a task',
        'Predict how temperature, top-k, top-p, min-p, beam search and penalties change generated text',
        'Use max tokens, stop sequences and seeds to bound and reproduce outputs',
        'Guarantee schema-valid JSON with constrained (guided) decoding in vLLM',
      ],
      blocks: [
        { t: 'lede', x: 'Two levers control what comes out of a model at inference time: **what you ask it to produce** (a direct answer, or a reasoning chain first) and **how each token is picked** from the [[logits|logits]]. This lesson covers [[chain-of-thought|chain-of-thought]] and its descendants, then every decoding knob you will see on the exam — and finishes with [[constrained-decoding|constrained decoding]], which makes "return valid JSON" a guarantee instead of a hope.' },

        { t: 'h2', x: 'Chain-of-thought prompting' },
        { t: 'p', x: 'A transformer spends a fixed amount of compute per generated token. Asking for intermediate steps before the answer gives the model more tokens — more computation — and lets later tokens attend to earlier partial results. That is why [[chain-of-thought|chain-of-thought]] (CoT) improves arithmetic, logic and multi-step questions.' },
        { t: 'vs', cards: [
          { h: 'Zero-shot CoT (Kojima et al., 2022)', items: [
            'Append a trigger such as "Let\'s think step by step."',
            'No examples needed — one line of prompt',
            'Often followed by a second call / a final line: "Therefore, the answer is"',
            'Works best on larger instruct models',
          ]},
          { h: 'Few-shot CoT (Wei et al., 2022)', items: [
            'Demonstrations include the worked reasoning, not just the answer',
            'Teaches the style and granularity of the steps',
            'Stronger and more controllable than zero-shot CoT',
            'Costs more prompt tokens per request',
          ]},
        ]},
        { t: 'code', lang: 'python', title: 'Few-shot CoT with a fixed answer line you can parse', x: `cot_prompt = """Q: A GPU node has 8 GPUs. A job uses 3 nodes and 2 GPUs per node are reserved for evaluation. How many GPUs train?
A: 3 nodes x 8 GPUs = 24 GPUs. Evaluation reserves 3 x 2 = 6. 24 - 6 = 18.
Answer: 18

Q: A dataset has 1.2M examples. You keep 85% after dedup and then 60% of those after quality filtering. How many remain?
A: 1,200,000 x 0.85 = 1,020,000. 1,020,000 x 0.60 = 612,000.
Answer: 612000

Q: {question}
A:"""
# The consistent "Answer: <number>" line makes the result machine-parseable
# and is what the self-consistency vote below extracts.` },
        { t: 'callout', kind: 'warn', title: 'CoT has costs', x: [
          'More output tokens = higher latency and cost (decode is the slow phase). A 300-token chain before a 1-token label can make a classifier 50× slower.',
          'CoT gains are weak or negative on small models (historically below ~10B params): they produce fluent but wrong chains. Measure on your eval set before shipping.',
          'Do not show raw chains to end users if they could leak the system prompt or retrieved data; strip them before display.',
        ]},

        { t: 'h2', x: 'Beyond one chain: self-consistency, tree-of-thought, ReAct, reasoning models' },
        { t: 'table', head: ['Technique', 'Idea', 'Cost', 'Use it for'], rows: [
          ['[[self-consistency|Self-consistency]]', 'Sample N CoT chains at temperature > 0, majority-vote the final answers', 'N× the generation tokens', 'Questions with a single checkable answer (math, classification, extraction)'],
          ['[[tree-of-thought|Tree-of-thought]]', 'Explore a tree of partial solutions; the model scores branches; search (BFS/DFS) with backtracking', 'Many calls per problem', 'Planning and puzzles where early choices matter'],
          ['[[react|ReAct]]', 'Interleave Thought → Action (tool call) → Observation in a loop', 'One call per step + tool latency', 'Agents that must look things up, compute or act'],
          ['[[reasoning-model|Reasoning models]]', 'Models trained (often with RL) to produce long internal reasoning before answering', 'Many "thinking" tokens', 'Hard math/code/analysis; you trade latency for accuracy'],
        ]},
        { t: 'code', lang: 'python', title: 'Self-consistency in a dozen lines', x: `import re
from collections import Counter
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def self_consistent_answer(question: str, n: int = 8) -> tuple[str, float]:
    resp = client.completions.create(          # completion endpoint: the CoT prompt is raw text
        model="Qwen/Qwen2.5-0.5B-Instruct",
        prompt=cot_prompt.format(question=question),
        n=n,                                    # N independent samples in one request
        temperature=0.7,                        # MUST be > 0, otherwise all N chains are identical
        max_tokens=256,
        stop=["\\nQ:"],                          # stop before the model invents the next question
    )
    answers = []
    for choice in resp.choices:
        m = re.search(r"Answer:\\s*([-\\d.,]+)", choice.text)
        if m:
            answers.append(m.group(1).replace(",", ""))
    if not answers:
        return "", 0.0
    best, votes = Counter(answers).most_common(1)[0]
    return best, votes / len(answers)           # agreement doubles as a confidence score

print(self_consistent_answer("A model has 500M parameters in bf16. How many GB are the weights?"))` },
        { t: 'h3', x: 'Reasoning models' },
        { t: 'p', x: 'Models such as DeepSeek-R1, Qwen3 in thinking mode and NVIDIA\'s Nemotron reasoning models were trained to emit a long reasoning trace (often inside `<think>…</think>`) before the answer. You prompt them differently: give the task plainly, skip [[few-shot|few-shot]] CoT examples (they can hurt), budget a large `max_tokens`, and strip the trace before showing or parsing the answer. Several families let you toggle reasoning per request — for example via a system-prompt switch or a chat-template flag — so one deployment can serve both quick and deliberate answers.' },

        { t: 'h2', x: 'How a token is picked: decoding' },
        { t: 'p', x: 'At each step the model outputs a logit for every vocabulary entry. The decoder then transforms and samples from them, roughly in this order (exact ordering varies slightly between engines):' },
        { t: 'flow', steps: [
          { b: 'Logits', s: 'One score per vocabulary token' },
          { b: 'Penalties', s: 'Repetition / frequency / presence penalties lower scores of tokens already used' },
          { b: 'Temperature', s: 'Divide logits by T, then [[softmax|softmax]]' },
          { b: 'Truncation', s: 'top-k, top-p and/or min-p remove the unlikely tail' },
          { b: 'Choose', s: 'Sample from what is left — or take the argmax (greedy)' },
        ]},
        { t: 'p', x: '**Temperature in numbers.** Logits [2, 1, 0]: at T = 1, softmax gives [0.665, 0.245, 0.090]. At T = 0.5 the logits become [4, 2, 0] and the probabilities [0.867, 0.117, 0.016] — sharper. At T = 2 they become [1, 0.5, 0] → [0.506, 0.307, 0.186] — flatter. T → 0 approaches greedy decoding.' },
        { t: 'table', head: ['Control', 'What it does', 'Typical values'], rows: [
          ['[[greedy-decoding|Greedy]]', 'Always take the argmax token. Deterministic, can loop on repeated phrases', '`do_sample=False` / `temperature=0`'],
          ['[[beam-search|Beam search]]', 'Keep the `num_beams` highest-probability partial sequences; return the best full sequence', '4–5 beams; translation, short outputs'],
          ['[[temperature|Temperature]]', 'Scales logits: < 1 sharper, > 1 flatter', '0–0.3 extraction/code · 0.7–1.0 creative'],
          ['[[top-k|Top-k]]', 'Keep only the k most likely tokens', '20–50 (0 or -1 = off, engine-dependent)'],
          ['[[top-p|Top-p (nucleus)]]', 'Keep the smallest set whose cumulative probability ≥ p', '0.9–0.95'],
          ['[[min-p|Min-p]]', 'Keep tokens with p ≥ min_p × p(top token) — adapts to the model\'s confidence', '0.05–0.1'],
          ['[[repetition-penalty|Repetition penalty]]', 'Multiplicative down-weighting of any token already in the context (HF/vLLM)', '1.0 (off) – 1.2'],
          ['Frequency penalty', 'Subtract α × (count of the token so far) from its logit', '0–1 (OpenAI range −2…2)'],
          ['Presence penalty', 'Subtract β once if the token has appeared at all — pushes toward new topics', '0–1 (OpenAI range −2…2)'],
          ['Max tokens', 'Hard cap on generated tokens (`max_new_tokens` in HF, `max_tokens` in APIs)', 'Size to the task; it also bounds latency'],
          ['[[stop-sequence|Stop sequences]]', 'End generation when a string appears (not included in output)', '`"\\nQ:"`, `"</answer>"`, `"###"`'],
          ['Seed', 'Fixes the random stream for sampling', 'Reproducible on the same engine, hardware and batch'],
        ]},
        { t: 'vs', cards: [
          { h: 'Greedy / beam search', items: [
            'Maximise likelihood — deterministic',
            'Good for short, closed answers: classification, extraction, translation',
            'Long open-ended text gets bland and repetitive ("likelihood trap")',
            'Beam search costs ~num_beams× compute and memory',
          ]},
          { h: 'Sampling (T, top-k, top-p, min-p)', items: [
            'Diverse, more human-like long text',
            'Needed for self-consistency and for generating varied synthetic data',
            'Too hot (T > 1.2, top-p 1.0) → incoherence and invented facts',
            'Not reproducible without a fixed seed and identical setup',
          ]},
        ]},
        { t: 'callout', kind: 'trap', title: 'Classic decoding mistakes', x: [
          'Setting `temperature=0.7` with `do_sample=False` in transformers does nothing — greedy ignores temperature (and recent versions warn about it).',
          'Setting `top_p=0.1` is not "a bit more focused": it usually leaves only 1–2 tokens — effectively greedy.',
          'A repetition penalty that is too high (≥ 1.5) damages output: the model avoids necessary words such as "the", variable names, or JSON keys.',
          'A seed alone does not guarantee identical outputs across GPUs, batch sizes or engine versions — floating-point reductions differ.',
        ]},

        { t: 'lab', title: 'Feel the sampling parameters', x: 'Generate the same prompt under different decoding settings with `model.generate` and observe determinism, diversity, repetition and early stopping. You will see the table above come to life in five minutes.', gpu: 'CPU / Apple Silicon OK (Free Colab T4 faster)', time: '20 min', steps: [
          'Install `transformers` and `torch`, and load `Qwen/Qwen2.5-0.5B-Instruct`.',
          'Run greedy decoding twice and confirm the outputs are identical.',
          'Sample at T = 0.3, 1.0 and 1.5 with three seeds each; compare diversity and coherence.',
          'Compare top-k = 5, top-p = 0.9 and min-p = 0.1 at T = 1.0.',
          'Turn on `repetition_penalty=1.3`, then add a stop string, and note how the output changes.',
        ]},
        { t: 'code', lang: 'python', title: 'lab_decoding.py', x: `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "Write one sentence describing a GPU to a child."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)

def run(label, seed=0, **gen_kwargs):
    set_seed(seed)                                     # seeds python, numpy and torch
    out = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                         max_new_tokens=40, pad_token_id=tok.eos_token_id, **gen_kwargs)
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"{label:<28} | {text.strip()}")

# 1) Greedy: deterministic, same answer every time
run("greedy", do_sample=False)
run("greedy (again)", do_sample=False)

# 2) Temperature sweep with several seeds
for T in (0.3, 1.0, 1.5):
    for s in range(3):
        run(f"T={T} seed={s}", seed=s, do_sample=True, temperature=T, top_k=0, top_p=1.0)

# 3) Truncation strategies at T=1.0 (top_k=0 disables top-k so only one filter is active)
run("top-k=5",  do_sample=True, temperature=1.0, top_k=5, top_p=1.0)
run("top-p=0.9", do_sample=True, temperature=1.0, top_k=0, top_p=0.9)
run("min-p=0.1", do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=0.1)

# 4) Beam search: 4 beams, deterministic, often blander
run("beam=4", do_sample=False, num_beams=4, early_stopping=True)

# 5) Repetition penalty and a stop string (stop_strings needs the tokenizer passed in)
run("rep_penalty=1.3", do_sample=False, repetition_penalty=1.3)
run("stop at first comma", do_sample=False, stop_strings=[","], tokenizer=tok)` },
        { t: 'p', x: 'What to notice: the two greedy runs match exactly; T = 0.3 samples barely differ from greedy while T = 1.5 drifts into odd word choices; top-k = 5 and min-p = 0.1 rein in a hot temperature; beam search tends to produce the safest phrasing; the stop string cuts output mid-sentence (it is included in HF output here, while APIs usually exclude it).' },

        { t: 'h2', x: 'Structured output: from "please return JSON" to guarantees' },
        { t: 'p', x: '[[structured-output|Structured output]] comes in three strengths. **Prompting** ("return JSON with keys …") usually works on good models but can fail — a trailing comment, a missing brace, an extra key. **JSON mode** guarantees syntactically valid JSON but not your schema. **Schema-constrained decoding** guarantees the output matches a JSON Schema, regex, choice list or grammar.' },
        { t: 'p', x: '[[constrained-decoding|Constrained (guided) decoding]] works at the logit level: the grammar is compiled into a state machine, and at every step the engine masks the logits of all tokens that would make the output invalid, so only legal tokens can be sampled. Libraries: **XGrammar** ([[vllm|vLLM]]\'s default backend and also used by other engines), **Outlines**, **llguidance** and **lm-format-enforcer**. NVIDIA [[nim|NIM]] for LLMs exposes the same capability for its models.' },
        { t: 'code', lang: 'python', title: 'Schema-valid JSON from vLLM via the OpenAI client', x: `from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

class Ticket(BaseModel):
    reasoning: str                                   # put free-text reasoning FIRST ...
    label: Literal["HARDWARE", "SOFTWARE", "NETWORK", "ACCESS"]
    urgency: int                                     # 1-5

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Ticket: Whole office lost Wi-Fi 10 minutes before a client demo."}],
    temperature=0,
    max_tokens=300,                                  # too small a cap truncates -> invalid JSON
    # OpenAI-style structured outputs; vLLM enforces it with its guided-decoding backend
    response_format={"type": "json_schema",
                     "json_schema": {"name": "ticket", "schema": Ticket.model_json_schema()}},
)
ticket = Ticket.model_validate_json(resp.choices[0].message.content)   # ... so the label is conditioned on it
print(ticket.label, ticket.urgency)

# Simpler constraint: force one of a fixed set of strings (vLLM-specific extra parameter).
# Older vLLM releases used extra_body={"guided_choice": [...]} / {"guided_json": schema};
# newer releases group these under a structured-outputs parameter - check your version's docs.
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Is 'great product, broke in a day' positive or negative?"}],
    extra_body={"guided_choice": ["positive", "negative"]},
)` },
        { t: 'callout', kind: 'warn', title: 'What constrained decoding does not do', x: [
          'It guarantees **shape**, not **truth**. The model can still put a wrong label in a perfectly valid field.',
          'Field order matters: a `reasoning` field before `answer` lets the answer depend on the reasoning; the reverse forces a snap decision.',
          'If `max_tokens` runs out mid-object, you get truncated (invalid) JSON. Check `finish_reason == "length"`.',
          'Very tight grammars can push the model into low-probability tokens and degrade quality; keep schemas as simple as the task allows.',
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Map the requirement to the knob: deterministic extraction → temperature 0 / greedy; diverse candidates or synthetic data → higher temperature with top-p; repetitive loops → repetition/frequency penalty; runaway outputs → max tokens + stop sequences; guaranteed JSON → guided decoding with a schema.',
          'Self-consistency requires sampling (T > 0) and a majority vote; with greedy decoding all N samples are identical.',
          'Zero-shot CoT = a trigger phrase; few-shot CoT = examples with worked reasoning. ReAct = reasoning interleaved with tool actions.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'CoT buys accuracy on multi-step problems with extra output tokens; self-consistency buys more with N samples and a vote.',
          'Tree-of-thought searches over branches; ReAct interleaves reasoning with tool calls; reasoning models internalise long CoT.',
          'Temperature reshapes the distribution; top-k, top-p and min-p cut its tail; greedy and beam search are deterministic.',
          'Penalties fight repetition; max tokens and stop sequences bound output; seeds give reproducibility only on an identical setup.',
          'Use schema-constrained decoding (XGrammar/Outlines via vLLM or NIM) when downstream code must parse the output.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'rag-domain',
      title: 'Domain adaptation without training: RAG and prompt-based adaptation',
      minutes: 25,
      outcomes: [
        'Choose between prompting, RAG and fine-tuning for a domain-adaptation requirement',
        'Build a RAG pipeline: chunking, embeddings, a vector index, retrieval, reranking and prompt assembly with citations',
        'Combine BM25 and dense retrieval with reciprocal rank fusion',
        'Identify NVIDIA NeMo Retriever embedding and reranking NIMs and where they fit',
        'Apply prompt-level techniques that reduce hallucination and manage prompt templates as versioned code',
      ],
      blocks: [
        { t: 'lede', x: 'Your company\'s runbooks, product specs and tickets were not in any pretraining set, and they change every week. [[rag|Retrieval-augmented generation]] fixes that without training: find the relevant passages at query time and put them in the prompt. This lesson builds a RAG pipeline end to end — [[chunking|chunking]], [[embedding|embeddings]], a [[vector-database|vector index]], [[hybrid-search|hybrid search]], [[reranking|reranking]] and citation-grounded prompts — and shows where NVIDIA\'s [[nemo-retriever|NeMo Retriever]] fits.' },

        { t: 'h2', x: 'Three ways to teach a model your domain' },
        { t: 'table', head: ['', 'Prompting (instructions + few-shot)', 'RAG', 'Fine-tuning (SFT / LoRA)'], rows: [
          ['Changes', 'The context of each request', 'The context, filled from your documents', 'The weights'],
          ['Good for', 'Format, tone, simple domain rules', 'Facts, fresh or private knowledge, long-tail lookups', 'Behaviour, style, output format, domain jargon, latency'],
          ['Knowledge freshness', 'Manual edits', 'Instant — re-index the document', 'Needs retraining'],
          ['Citations / traceability', 'No', 'Yes — you know which passages were used', 'No'],
          ['Cost to start', 'Minutes', 'Days (pipeline + index)', 'Days–weeks (data + GPUs + eval)'],
          ['Per-request cost', 'Prompt tokens', 'Retrieval + many prompt tokens', 'Cheapest — knowledge is in the weights'],
        ]},
        { t: 'p', x: 'Rule of thumb: **prompting first; RAG for knowledge; fine-tuning for behaviour**. They combine well — a LoRA-tuned model that follows your citation format, fed by a RAG pipeline, is a common production pattern. Fine-tuning is a poor way to inject facts that change: it is expensive to update and the model still [[hallucination|hallucinates]] details it saw only a few times.' },

        { t: 'h2', x: 'The RAG pipeline' },
        { t: 'flow', steps: [
          { b: 'Ingest', s: 'Parse PDFs/HTML/tickets; keep metadata (source, date, section, ACL)' },
          { b: 'Chunk', s: 'Split into passages of ~200–800 tokens with some overlap' },
          { b: 'Embed', s: 'A bi-encoder turns each chunk into a vector' },
          { b: 'Index', s: 'Store vectors + metadata in a vector database (ANN index)' },
          { b: 'Retrieve', s: 'Embed the query; top-k nearest chunks (+ BM25 for hybrid)' },
          { b: 'Rerank', s: 'A cross-encoder rescores the candidates; keep the best few' },
          { b: 'Assemble', s: 'Numbered sources + instructions + question in a prompt template' },
          { b: 'Generate & cite', s: 'The LLM answers from the sources and cites [n]' },
        ]},
        { t: 'h3', x: 'Chunking' },
        { t: 'ul', items: [
          '**Too small** (a sentence): loses context — "it must be restarted" without saying what "it" is.',
          '**Too large** (a whole chapter): the embedding blurs many topics, retrieval gets less precise, and you burn context tokens.',
          'Start around 256–512 tokens with 10–20% overlap so facts that straddle a boundary appear whole in at least one chunk.',
          'Prefer structure-aware splitting (headings, paragraphs, code blocks, table rows) over fixed character counts.',
          'Respect the embedding model\'s max input length — text beyond it is silently truncated.',
          'Carry metadata (title, section path, date) into each chunk; prepending the document title to the chunk text often improves retrieval.',
        ]},
        { t: 'h3', x: 'Embeddings and vector indexes' },
        { t: 'p', x: 'An embedding model (bi-encoder) maps text to a fixed-size vector so that semantically similar texts are close. Normalise vectors to unit length and cosine similarity becomes a plain dot product. Query and passage are encoded **independently**, which is what makes it fast: passages are embedded once, offline. Many retrieval models expect different prefixes or an `input_type` for queries vs passages — follow the [[model-card|model card]].' },
        { t: 'table', head: ['Index type', 'How it works', 'Trade-off'], rows: [
          ['Flat (exact)', 'Compare the query with every vector', 'Perfect recall; O(N) — fine up to ~1M vectors on GPU'],
          ['IVF', 'Cluster vectors; search only the nearest clusters (nprobe)', 'Fast, needs training; recall depends on nprobe'],
          ['HNSW', 'Multi-layer proximity graph walked greedily', 'High recall and fast queries; memory-hungry; the common default'],
          ['IVF-PQ / quantised', 'Compress vectors with product quantisation', 'Billions of vectors in memory; some recall loss'],
          ['GPU graph (CAGRA, cuVS)', 'GPU-built and GPU-searched proximity graph', 'Very high throughput for build and batch queries (Milvus, FAISS GPU)'],
        ]},
        { t: 'p', x: 'A [[vector-database|vector database]] (Milvus, Qdrant, Weaviate, pgvector, Elasticsearch/OpenSearch…) adds what a bare library like FAISS lacks: persistence, metadata filtering ("only docs this user may see, newer than 2025"), updates and deletes, replication.' },

        { t: 'lab', title: 'A RAG pipeline in one file', x: 'Build a working RAG loop over a handful of runbook snippets: chunk, embed with sentence-transformers, index with FAISS, retrieve, rerank with a cross-encoder and answer with citations from a local OpenAI-compatible server.', gpu: 'CPU / Apple Silicon OK', time: '30 min', steps: [
          '`pip install sentence-transformers faiss-cpu openai rank_bm25` (use faiss-gpu builds on a CUDA box if you like).',
          'Run the indexing part and inspect the shapes: N chunks × 384 dims.',
          'Ask three questions; print the retrieved chunk ids before and after reranking.',
          'Start the vLLM server from lesson 1 and run the generation step; check that the answer cites [n].',
          'Ask something the documents do not cover and confirm the model says it does not know.',
        ]},
        { t: 'code', lang: 'python', title: 'rag_lab.py — index, retrieve, rerank, answer', x: `import numpy as np, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

docs = {
    "gpu-oom.md": "If training fails with CUDA out of memory, first lower the micro-batch size and raise "
                  "gradient accumulation to keep the effective batch. Then enable activation checkpointing.",
    "vllm-restart.md": "To restart the inference service, drain traffic from the pod, wait for in-flight "
                       "requests to finish, then delete the pod; the Deployment recreates it.",
    "nccl-timeout.md": "NCCL timeouts usually mean one rank crashed or is stuck. Check every rank's log, "
                       "then verify the network with nccl-tests before relaunching.",
    "kv-cache.md": "If vLLM logs preemptions, the KV cache is full. Lower max-num-seqs, shorten "
                   "max-model-len, or raise gpu-memory-utilization.",
}

# 1) Chunk: tiny docs here, so one chunk per sentence group; real code splits on headings/tokens.
chunks, sources = [], []
for name, text in docs.items():
    for part in text.split(". "):
        chunks.append(f"{name}: {part.strip()}")      # prepend the title -> better retrieval
        sources.append(name)

# 2) Embed (bi-encoder) with unit-length vectors so inner product == cosine similarity
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # 384-dim, CPU-friendly
emb = embedder.encode(chunks, normalize_embeddings=True).astype("float32")

# 3) Index: exact inner-product search (swap for HNSW/IVF at scale)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
print("indexed", index.ntotal, "chunks of dim", emb.shape[1])

# 4) Retrieve top-k candidates, 5) rerank with a cross-encoder (reads query+chunk together)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve(query: str, k: int = 6, keep: int = 3):
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    cands = [int(i) for i in ids[0] if i != -1]
    rr = reranker.predict([(query, chunks[i]) for i in cands])
    ranked = [cands[j] for j in np.argsort(-rr)][:keep]
    return ranked

# 6) Assemble a grounded prompt with numbered sources, 7) generate
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def answer(query: str) -> str:
    ids = retrieve(query)
    context = "\\n".join(f"[{n}] ({sources[i]}) {chunks[i]}" for n, i in enumerate(ids, 1))
    messages = [
        {"role": "system", "content":
            "Answer ONLY from the numbered sources. Cite sources like [1]. "
            "If the sources do not contain the answer, reply exactly: I don't know."},
        {"role": "user", "content": f"<sources>\\n{context}\\n</sources>\\n\\nQuestion: {query}"},
    ]
    r = client.chat.completions.create(model="Qwen/Qwen2.5-0.5B-Instruct",
                                       messages=messages, temperature=0, max_tokens=200)
    return r.choices[0].message.content

print(answer("vLLM keeps preempting requests, what should I change?"))
print(answer("What is our parental leave policy?"))   # not in the docs -> should refuse` },

        { t: 'h2', x: 'Hybrid search and reranking' },
        { t: 'p', x: 'Dense embeddings capture meaning ("GPU ran out of memory" ≈ "CUDA OOM") but are weak on exact strings: error codes, part numbers, function names, rare acronyms. [[bm25|BM25]] keyword search is the opposite. [[hybrid-search|Hybrid search]] runs both and fuses the ranked lists — most robustly with **reciprocal rank fusion** (RRF), which needs no score calibration: score(d) = Σ 1 / (k + rank_i(d)), with k ≈ 60.' },
        { t: 'code', lang: 'python', title: 'BM25 + dense with reciprocal rank fusion', x: `from rank_bm25 import BM25Okapi

bm25 = BM25Okapi([c.lower().split() for c in chunks])

def hybrid(query: str, k: int = 10, rrf_k: int = 60):
    # dense ranking
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    _, dense_ids = index.search(q, k)
    # sparse (keyword) ranking
    sparse_ids = np.argsort(-bm25.get_scores(query.lower().split()))[:k]
    fused = {}
    for ranking in (dense_ids[0], sparse_ids):
        for rank, i in enumerate(ranking):
            if i == -1:
                continue
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)

print([sources[i] for i in hybrid("NCCL timeout")][:3])` },
        { t: 'p', x: '[[reranking|Reranking]] is the second stage. A **cross-encoder** reads the query and a passage together through full attention, so it judges relevance far better than comparing two independent vectors — but it must run once per (query, passage) pair. The standard pattern: retrieve 20–100 candidates cheaply, rerank them, pass the top 3–8 to the LLM. Reranking usually improves answer quality more than switching to a bigger embedding model.' },

        { t: 'h2', x: 'The NVIDIA stack for retrieval' },
        { t: 'specs', rows: [
          ['NeMo Retriever', 'NVIDIA\'s family of retrieval microservices: extraction (parsing PDFs, tables, charts), embedding and reranking models packaged as NIMs'],
          ['Embedding NIM', 'OpenAI-compatible `/v1/embeddings`; retrieval-tuned models (e.g. the NV-EmbedQA family) take `input_type: "query"` or `"passage"`'],
          ['Reranking NIM', 'Cross-encoder reranker (e.g. the NV-RerankQA family) behind a `/v1/ranking` endpoint: query + passages in, relevance logits out'],
          ['cuVS / GPU indexes', 'GPU-accelerated vector search (CAGRA, IVF-PQ) used by Milvus and FAISS for fast index build and high-QPS search'],
          ['RAG Blueprint', 'NVIDIA reference architecture wiring ingestion, NeMo Retriever, a vector DB, an LLM NIM and optional guardrails'],
        ]},
        { t: 'code', lang: 'python', title: 'Calling an embedding NIM (same client, different model)', x: `from openai import OpenAI

# A locally running embedding NIM (or NVIDIA's hosted API with your key and its base_url)
emb_client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")

def embed(texts, kind):   # kind = "query" for questions, "passage" for documents
    r = emb_client.embeddings.create(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",    # check the NIM catalog for current model names
        input=texts,
        extra_body={"input_type": kind, "truncate": "END"},
    )
    return [d.embedding for d in r.data]

doc_vecs = embed(chunks, "passage")
q_vec = embed(["How do I fix NCCL timeouts?"], "query")[0]` },
        { t: 'callout', kind: 'trap', title: 'Asymmetric embedding models', x: 'Retrieval-tuned embedders are trained with different handling for queries and passages. Embedding queries as passages (wrong `input_type` or missing prefix) silently lowers recall. Also: you must re-embed the whole corpus when you change the embedding model — vectors from two models are not comparable.' },

        { t: 'h2', x: 'Grounding the answer and fighting hallucination' },
        { t: 'ul', items: [
          '**Instruct grounding explicitly**: "answer only from the sources; if they do not contain the answer, say you don\'t know". Give the model a permitted way out.',
          '**Number the sources and require citations** [n]; you can then verify programmatically that every cited id exists and show the passages to users.',
          '**Mind position**: models use information at the start and end of long contexts better than the middle ("lost in the middle"). Put the best chunks first, fewer is often better.',
          '**Use low temperature** for factual QA.',
          '**Filter by metadata** (product version, date, access rights) before ranking — the right answer for v2 is a hallucination for v3 users.',
          '**Evaluate retrieval separately** from generation: recall@k / MRR on retrieval, faithfulness and answer relevance on generation. If the right chunk is never retrieved, no prompt will fix the answer.',
          '**Add guardrails** for topic control and fact-checking against sources where the risk justifies it.',
        ]},
        { t: 'callout', kind: 'warn', title: 'Retrieved text is untrusted input', x: 'Documents can contain instructions ("ignore previous instructions and…"). Keep retrieved text inside clear delimiters, tell the model that sources are data, and never give a RAG agent tools with more privilege than the least-trusted document deserves.' },

        { t: 'h2', x: 'Managing prompts as code' },
        { t: 'p', x: 'Once prompts drive production behaviour they need the same discipline as code. A [[prompt-template|prompt template]] separates the fixed instructions from the variables filled at runtime.' },
        { t: 'ul', items: [
          'Store templates in version control (Jinja or plain format strings), not scattered through application code; give each a name and version.',
          'Log the template version with every request so a quality regression can be traced to a prompt change.',
          'Run an evaluation set on every prompt change, exactly like a unit test suite; a "small wording tweak" can move accuracy several points.',
          'Pin the model version together with the prompt — the same prompt behaves differently on a new model.',
          'Keep system prompts free of secrets; assume they can be extracted.',
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Scenario → method: "knowledge changes daily / needs citations / private documents" → RAG. "Must always answer in our house style and JSON format" → fine-tuning (or constrained decoding). "Quick improvement with no training budget" → prompt engineering / few-shot.',
          'Know the stages and their purpose: chunk → embed → index → retrieve → rerank → generate. Reranking = cross-encoder on top-k candidates; hybrid = BM25 + dense.',
          'NVIDIA mapping: NeMo Retriever = embedding/reranking/extraction NIMs; cuVS = GPU vector search; NIM for the LLM.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Prompting first, RAG for knowledge, fine-tuning for behaviour — and they compose.',
          'Chunk with structure and overlap; embed with a retrieval model; store in an index or vector DB that supports metadata filtering.',
          'Hybrid search (BM25 + dense, fused with RRF) handles both meaning and exact identifiers; a cross-encoder reranker sharpens the top results.',
          'NeMo Retriever NIMs provide production embedding and reranking behind OpenAI-style APIs.',
          'Ground with numbered sources, citations and an explicit "I don\'t know" path; evaluate retrieval and generation separately; version your prompts.',
        ]},
      ],
    },
  ],

  quiz: [
    { id: 'p01', domain: 'prompting',
      scenario: 'A team prompts a base (non-instruct) 7B model with: "Please summarise the following meeting notes in three bullets: …". The model responds by writing more meeting notes and then a question from another attendee.',
      q: 'What is the most effective fix without training?',
      opts: [
        'Raise the temperature so the model explores answers beyond the meeting notes',
        'Rewrite it as a completion-style document of notes and summaries that ends where a summary begins',
        'Wrap the request in the instruct model\'s chat template tokens to activate instruction following',
        'Add a repetition penalty of 2.0 so the model stops repeating the meeting notes',
      ],
      a: 1,
      why: 'A base model continues documents; it was never trained to follow instructions. A few-shot, completion-style prompt makes the summary the most probable continuation. Alternatively, switch to the instruct variant.',
      distract: 'Higher temperature makes the continuation more random, not more obedient. A base model never saw chat-template tokens during training, so they do not unlock instruction following. A large repetition penalty distorts text without changing the task the model believes it is doing.' },

    { id: 'p02', domain: 'prompting',
      scenario: 'A few-shot sentiment prompt has six examples: the first five are positive, the last one is negative. On a test set the model labels most ambiguous reviews as positive, and when the examples are reversed the bias shifts.',
      q: 'Which explanation and fix are most appropriate?',
      opts: [
        'The context window is overflowing; reduce to zero-shot to save tokens',
        'The model needs fine-tuning because few-shot cannot express binary labels',
        'Majority-label and recency biases; balance labels and interleave or shuffle them',
        'Greedy decoding is biased toward positive tokens; switch to beam search',
      ],
      a: 2,
      why: 'Few-shot predictions are pulled toward the most frequent label in the demonstrations and toward the labels of the last examples. Balancing and interleaving labels (and validating order on a held-out set) removes most of the effect.',
      distract: 'Six short examples do not overflow any modern context window. Few-shot handles binary labels well once demonstrations are balanced. Beam search also maximises likelihood and inherits the same prompt-induced bias.' },

    { id: 'p03', domain: 'prompting',
      q: 'You need N reasoning chains to disagree occasionally so that a majority vote is meaningful. Which setting is required for self-consistency?',
      opts: [
        'Sampling with temperature above 0 for each of the N generations',
        'Greedy decoding repeated N times with different seeds',
        'Beam search with num_beams = N and returning only the top beam',
        'A repetition penalty above 1.5 so later chains avoid earlier words',
      ],
      a: 0,
      why: '[[self-consistency|Self-consistency]] samples diverse reasoning paths (e.g. T ≈ 0.5–0.8) and votes on the final answers. Diversity is what lets correct reasoning paths outvote occasional wrong ones.',
      distract: 'Greedy decoding ignores the seed and returns the same chain N times. Returning one beam gives a single answer, not a vote. Repetition penalty acts within one sequence, not across independent samples, and degrades text.' },

    { id: 'p04', domain: 'prompting',
      scenario: 'A support classifier must return one of four labels and a confidence, parsed by downstream code. About 2% of responses fail JSON parsing because the model adds commentary.',
      q: 'What guarantees parseable, schema-valid output?',
      opts: [
        'Add "Return ONLY valid JSON" in capital letters to the system prompt',
        'Lower the temperature to 0 so the output format is always deterministic',
        'Post-process with a regex that extracts the first {...} block',
        'Use guided decoding with a JSON Schema that masks invalid tokens',
      ],
      a: 3,
      why: '[[constrained-decoding|Constrained decoding]] compiles the schema into a state machine and masks logits of tokens that would violate it — the output cannot fail to parse (unless truncated by max_tokens).',
      distract: 'Stronger wording reduces but does not eliminate failures. Temperature 0 makes the failure deterministic, not impossible. A regex can still extract malformed or schema-violating JSON.' },

    { id: 'p05', domain: 'prompting',
      q: 'With logits [2, 1, 0], what happens to the probability of the top token when temperature drops from 1.0 to 0.5?',
      opts: [
        'It falls from about 0.67 to about 0.51',
        'It rises from about 0.67 to about 0.87',
        'It stays at about 0.67 because temperature only affects sampling, not probabilities',
        'It rises to exactly 1.0, because T below 1 is the same as greedy',
      ],
      a: 1,
      why: 'Dividing by T = 0.5 gives logits [4, 2, 0]; softmax → [0.867, 0.117, 0.016]. Lower temperature sharpens the distribution toward the top token.',
      distract: '0.51 is the T = 2 value (flatter). Temperature changes the distribution itself, which is why it changes sampling. Only T → 0 approaches greedy; T = 0.5 still leaves real probability on other tokens.' },

    { id: 'p06', domain: 'prompting',
      scenario: 'A RAG assistant over internal docs answers poorly when users paste error codes such as "XID 79" or config keys such as "max_num_batched_tokens". Paraphrased questions work well.',
      q: 'What change most directly addresses this?',
      opts: [
        'Increase the chunk size so each chunk contains more error codes',
        'Switch the LLM to a larger model with better reasoning',
        'Add BM25 keyword retrieval and fuse it with dense results using reciprocal rank fusion',
        'Raise the generation temperature so the model considers more candidate answers',
      ],
      a: 2,
      why: 'Dense embeddings are weak on rare exact strings; BM25 matches them exactly. [[hybrid-search|Hybrid search]] with RRF keeps semantic recall and adds exact-match recall.',
      distract: 'Bigger chunks dilute embeddings and do not make exact matching better. A larger LLM cannot answer from passages that were never retrieved. Temperature affects generation, not retrieval.' },

    { id: 'p07', domain: 'prompting',
      scenario: 'Product pricing and policies change weekly; answers must cite the source document; the budget for GPU training is zero.',
      q: 'Which adaptation approach fits best?',
      opts: [
        'RAG over the policy documents, re-indexed whenever they change',
        'Continued pretraining of the model on the policy documents every week',
        'LoRA fine-tuning on question/answer pairs generated from the documents',
        'A longer system prompt that lists every current price and policy',
      ],
      a: 0,
      why: 'RAG handles fresh, private knowledge, gives citations (you know which passages were used) and needs no training — updating means re-indexing.',
      distract: 'Continued pretraining and LoRA need GPU training, must be repeated on every change and provide no citations. Pasting everything into the system prompt does not scale, costs tokens on every request and gives no traceability.' },

    { id: 'p08', domain: 'prompting',
      q: 'Which statement about cross-encoder rerankers versus bi-encoder embedding retrieval is correct?',
      opts: [
        'Cross-encoders precompute passage vectors offline, so they are cheaper per query than bi-encoders',
        'Bi-encoders read query and passage together, so they are more accurate but slower',
        'Rerankers replace the vector index entirely in most production RAG systems today',
        'Cross-encoders score each query–passage pair jointly: accurate, but too slow for a whole corpus',
      ],
      a: 3,
      why: 'A cross-encoder attends over query and passage together, which gives better relevance judgments, but requires one forward pass per pair. Hence the two-stage pattern: bi-encoder retrieves candidates, cross-encoder [[reranking|reranks]] them.',
      distract: 'Precomputed passage vectors are the bi-encoder property. The second option swaps the two architectures. Rerankers cannot scan millions of chunks per query, so they complement the index rather than replace it.' },

    { id: 'p09', domain: 'prompting',
      scenario: 'An agent must answer questions about current cluster status by calling a monitoring API, reading the result, deciding whether to call another API, and finally answering.',
      q: 'Which prompting pattern is designed for this?',
      opts: [
        'Self-consistency with eight samples and a majority vote',
        'ReAct: interleaved Thought, Action (tool call) and Observation steps',
        'Few-shot chain-of-thought with worked arithmetic examples',
        'Beam search with a length penalty to produce complete plans',
      ],
      a: 1,
      why: '[[react|ReAct]] alternates reasoning with actions and feeds each tool observation back into the context, which is exactly a look-up-then-decide loop.',
      distract: 'Self-consistency and few-shot CoT reason only over what is already in the prompt; they cannot fetch live data. Beam search is a decoding strategy, not an agent pattern.' },

    { id: 'p10', domain: 'prompting',
      scenario: 'A chatbot generates long answers that sometimes loop ("…and the GPU, and the GPU, and the GPU…") and occasionally run until the 4,096-token limit.',
      q: 'Which pair of controls addresses both symptoms?',
      opts: [
        'Set top-k to 1 and raise the temperature to 1.5 for more variety',
        'Switch to beam search with 4 beams and remove the max token limit',
        'A mild repetition or frequency penalty, plus max_tokens and stop sequences',
        'Remove the system prompt and add more few-shot examples of long answers',
      ],
      a: 2,
      why: 'Frequency/[[repetition-penalty|repetition penalties]] down-weight tokens already generated, breaking loops; max_tokens and [[stop-sequence|stop sequences]] bound runaway outputs.',
      distract: 'top-k = 1 is greedy, which makes loops more likely, and temperature has no effect once only one token remains. Beam search tends to repeat more, and removing the cap makes runaways worse. Removing the system prompt does not address either decoding problem.' },
  ],
}
