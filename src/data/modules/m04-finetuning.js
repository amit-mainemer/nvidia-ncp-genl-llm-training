export default {
  id: "finetuning", n: 4, domain: "finetuning", weight: 13,
  title: "Fine-Tuning",
  blurb: "Decide when fine-tuning is the right tool, then do it: full SFT with TRL, LoRA and QLoRA on a single GPU, preference alignment with DPO and GRPO, and the NeMo path to the same results at NVIDIA scale — without making the model forget what it already knew.",
  hours: "6–8 h",
  lessons: [
    /* ------------------------------------------------------------------ */
    {
      id: "when-to-finetune",
      title: "Choosing an adaptation method: prompting, RAG, PEFT, full FT, continued pretraining",
      minutes: 18,
      outcomes: [
        "Rank the six adaptation levers — prompting, RAG, PEFT, full fine-tuning, continued pretraining, pretraining — by cost and by what they change",
        "Distinguish problems of missing knowledge from problems of behaviour, format and style, and match each to a method",
        "Estimate the data and compute each method needs, using the 6·N·D rule for training FLOPs",
        "Choose between a base and an instruct checkpoint as the starting point for fine-tuning",
      ],
      blocks: [
        { t: "lede", x: "Most failed fine-tuning projects did not fail in training — they failed at the decision to fine-tune. Before you spend a single GPU-hour, you should be able to say what is wrong with the current model (it does not know something, or it does not behave the way you want), and pick the cheapest lever that fixes exactly that. This lesson gives you the decision framework the exam expects, with the numbers behind it." },

        { t: "h2", x: "Six levers, ordered by cost" },
        { t: "p", x: "Think of adaptation as a ladder. Each rung costs roughly an order of magnitude more data, compute or engineering than the one below it, and each rung changes something deeper in the model. Climb only as far as you need." },
        { t: "flow", steps: [
          { b: "1 · Prompting", s: "Zero/few-shot examples, a system prompt, output constraints. Changes nothing in the weights. Minutes to iterate." },
          { b: "2 · RAG", s: "Retrieve documents at query time and put them in the context. Adds knowledge without training; knowledge stays editable and citable." },
          { b: "3 · PEFT", s: "[[lora|LoRA]], [[qlora|QLoRA]], [[adapters|adapters]], [[p-tuning|p-tuning]]: train well under 1–2% of parameters. Changes behaviour, format, style, task skill. Hours on one GPU." },
          { b: "4 · Full fine-tuning", s: "[[full-fine-tuning|Update every weight]] with [[sft|SFT]] and/or preference alignment. Highest quality ceiling for a task; ~16 bytes/param of training memory." },
          { b: "5 · Continued pretraining", s: "[[continued-pretraining|Keep pretraining]] on billions of unlabeled in-domain tokens (legal, biomedical, a new language). Changes the model's knowledge and vocabulary use." },
          { b: "6 · Pretraining from scratch", s: "Trillions of tokens, thousands of GPUs. Only when no suitable base exists (new modality, new tokenizer, sovereign model)." },
        ]},

        { t: "h2", x: "Knowledge versus behaviour" },
        { t: "p", x: "The single most useful distinction: fine-tuning on a few thousand examples is very good at teaching **behaviour** — follow this format, use this tone, call this tool, refuse that request, solve this kind of task — and comparatively poor at reliably injecting **new facts**. A model that sees a fact a handful of times in SFT data may repeat it, paraphrase it wrongly, or start confidently inventing facts in the same style ([[hallucination|hallucination]])." },
        { t: "vs", cards: [
          { h: "Behaviour problems → fine-tune (PEFT first)", items: [
            "Output must always be valid JSON matching a schema",
            "Answers must follow a house style, length or persona",
            "A narrow task (classification, extraction, SQL generation) where a small tuned model must match a large prompted one",
            "Long prompts with many few-shot examples are too slow or expensive — bake them into the weights",
            "Teaching a base model to follow instructions at all ([[instruction-tuning|instruction tuning]])",
          ]},
          { h: "Knowledge problems → RAG or continued pretraining", items: [
            "Facts that change (prices, policies, inventory, tickets) → [[rag|RAG]]",
            "Answers must cite a source document → RAG",
            "Per-user or access-controlled data → RAG (never bake private data into shared weights)",
            "A whole domain's vocabulary and style of reasoning (clinical notes, chip design) → continued pretraining, then SFT",
            "Missing language coverage → continued pretraining (sometimes with tokenizer extension)",
          ]},
        ]},
        { t: "callout", kind: "trap", title: "“Fine-tune it on our docs so it knows them”", x: [
          "This is the classic wrong answer. Fine-tuning on a document dump gives no citations, cannot be updated without retraining, cannot enforce per-user access, and often increases hallucination.",
          "The right pattern is usually **RAG for knowledge + light fine-tuning for behaviour** (for example, teach the model to answer only from the retrieved context and to say “not in the documents” otherwise).",
        ]},

        { t: "h2", x: "The decision table" },
        { t: "table", head: ["Method", "What it changes", "Typical data", "Compute (7B-class)", "Use when", "Watch out for"], rows: [
          ["Prompting", "Nothing in weights — only the input", "0–20 examples", "None", "Always the first attempt and the baseline", "Prompt length = latency and cost on every call"],
          ["RAG", "What the model sees at inference", "A document corpus + retriever", "Embedding the corpus; no training", "Fresh, private, citable knowledge", "Retrieval quality caps answer quality"],
          ["PEFT (LoRA/QLoRA)", "Behaviour, format, task skill; small low-rank weight update", "500–50k labeled examples", "1 GPU, hours (QLoRA fits 7B in ~10 GB)", "Most production customisation; many tasks on one base", "Slightly lower ceiling than full FT on hard tasks"],
          ["Full fine-tuning", "All weights", "10k–1M examples", "Multi-GPU: ~16 B/param ⇒ ~112 GB + activations for 7B", "Maximum quality, large distribution shift, one model per use case", "[[catastrophic-forgetting|Catastrophic forgetting]], cost, a full model copy per task"],
          ["Continued pretraining", "Knowledge and domain language", "1B–100B+ unlabeled tokens", "Multi-node, days", "Big domain or language gap", "Needs SFT/alignment again afterwards; replay general data"],
          ["Pretraining", "Everything", "Trillions of tokens", "Thousands of GPUs, weeks", "No usable base model exists", "Almost never the right answer for an application team"],
        ]},
        { t: "ol", items: [
          "Write an evaluation set first (50–200 real inputs with expected outputs or a rubric). Without it you cannot tell whether any rung helped.",
          "Push prompting as far as it goes: system prompt, few-shot examples, structured output. Record the score.",
          "If failures are about missing or changing facts, add RAG and re-measure.",
          "If failures are about behaviour, format, latency or cost, try PEFT ([[lora|LoRA]] or [[qlora|QLoRA]]) on a few thousand high-quality examples.",
          "Move to full fine-tuning only if PEFT plateaus below target and you have the data and GPUs; move to continued pretraining only for a genuine domain/language gap.",
          "After any training, align preferences if needed ([[dpo|DPO]], [[grpo|GRPO]]) and re-check general capability for forgetting.",
        ]},
        { t: "callout", kind: "exam", title: "How this is tested", x: [
          "Scenario questions describe a symptom and ask for the most cost-effective fix. Map: changing facts → RAG; format/style/task → PEFT; domain language gap with lots of unlabeled text → continued pretraining; limited GPU memory → QLoRA; many customers on one base → LoRA adapters served together.",
          "“Least compute / fastest to iterate” almost always means prompting or PEFT, never full fine-tuning.",
        ]},

        { t: "h2", x: "Doing the numbers" },
        { t: "p", x: "Training compute is about **6 × N × D** FLOPs, where N is the number of trainable-path parameters the forward and backward pass touch (the whole model, even for LoRA — gradients still flow through frozen layers) and D is the number of training tokens. Divide by your *achieved* throughput, not the datasheet peak." },
        { t: "specs", rows: [
          ["SFT of a 0.5B model on 10M tokens", "6 × 0.5e9 × 1e7 = 3e16 FLOPs → ~5 min at an achieved 100 TFLOP/s"],
          ["LoRA SFT of a 7B model on 20M tokens", "≈ 6 × 7e9 × 2e7 = 8.4e17 FLOPs → ~40 min at 350 TFLOP/s (LoRA saves memory and optimizer work, not much forward/backward compute)"],
          ["Continued pretraining of 7B on 10B tokens", "6 × 7e9 × 1e10 = 4.2e20 FLOPs → ~36 h on 8× H100 at ~400 TFLOP/s achieved each"],
          ["Typical SFT dataset", "1k–100k conversations; quality beats quantity (≈1k curated examples can already change style dramatically)"],
          ["Typical preference dataset (DPO)", "5k–100k (prompt, chosen, rejected) triples"],
        ]},

        { t: "h2", x: "Base or instruct?" },
        { t: "p", x: "A **base** checkpoint (e.g. `Qwen/Qwen2.5-0.5B`) has only been pretrained: it continues text. An **instruct** checkpoint (`Qwen/Qwen2.5-0.5B-Instruct`) has already been through SFT and preference alignment, with a specific chat template. Which you start from depends on how far your target behaviour is from a general assistant." },
        { t: "table", head: ["Start from", "Choose it when", "Remember"], rows: [
          ["[[base-model|Base model]]", "You will do substantial SFT (≥10k examples), need a non-chat format (completion, classification, code infill), or are doing continued pretraining", "You must teach turn-taking and the end-of-turn token yourself; no built-in safety behaviour"],
          ["[[instruct-model|Instruct model]]", "Small dataset (hundreds to a few thousand examples), you want to keep chat skills and safety tuning, light style/format changes", "Use **its exact chat template**; a mismatched template silently degrades quality and can erode its safety behaviour"],
        ]},
        { t: "callout", kind: "warn", title: "Fine-tuning can remove safety behaviour", x: "Even benign fine-tuning of an instruct model can weaken the refusals it learned during alignment. If you fine-tune an aligned model, include refusal examples in your data, re-run your safety evaluation, and keep guardrails in front of it in production (Module 10)." },

        { t: "h2", x: "Key takeaways" },
        { t: "ul", items: [
          "Climb the ladder: prompting → RAG → PEFT → full FT → continued pretraining → pretraining. Stop at the first rung that meets your eval target.",
          "Fine-tuning teaches behaviour, format and skills; RAG supplies fresh, private, citable knowledge. Many real systems use both.",
          "Training compute ≈ 6·N·D FLOPs; full FT memory ≈ 16 bytes/param before activations; QLoRA puts a 7B model on one consumer GPU.",
          "Start from an instruct model for light tuning with small data; start from a base model for heavy SFT, non-chat formats or continued pretraining.",
          "Always build the evaluation set before training, and re-check general and safety behaviour afterwards.",
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: "full-sft",
      title: "Lab: supervised fine-tuning with TRL",
      minutes: 35,
      outcomes: [
        "Format an instruction dataset into the conversational prompt–completion shape TRL expects",
        "Configure SFTTrainer/SFTConfig with completion-only loss, sensible learning rate, warmup, epochs and evaluation",
        "Compare a base model's outputs before and after SFT and read the train/eval loss curves for overfitting",
        "Explain why end-of-turn tokens, chat templates and fp32 master weights matter for a successful full fine-tune",
        "Save a fine-tuned model and push it to the Hugging Face Hub",
      ],
      blocks: [
        { t: "lede", x: "Time to turn a base model into an assistant. You will take `Qwen/Qwen2.5-0.5B` — a pure next-token predictor — and run [[full-fine-tuning|full]] [[sft|supervised fine-tuning]] on a few thousand instruction–response pairs with TRL's `SFTTrainer`. Same objective as pretraining, different data and a loss mask: that is all [[instruction-tuning|instruction tuning]] is. The whole run takes 15–30 minutes on one GPU." },

        { t: "h2", x: "What SFT actually optimises" },
        { t: "p", x: "SFT minimises the same [[cross-entropy|cross-entropy]] as pretraining, but only over the response tokens. For a prompt x and response y = (y₁…y_T): **L = −(1/T) Σₜ log p_θ(yₜ | x, y₍<ₜ₎)**. The prompt tokens are still fed in — the model has to read them — but their labels are set to −100 so they contribute nothing to the [[loss|loss]]. Training on prompt tokens too wastes capacity on predicting user text and, with short answers, lets the prompt dominate the gradient." },
        { t: "table", head: ["Dataset format (TRL)", "Example row", "What the loss covers"], rows: [
          ["Language modelling", "`{\"text\": \"...\"}`", "Every token (use for continued pretraining)"],
          ["Prompt–completion", "`{\"prompt\": \"...\", \"completion\": \"...\"}`", "Only the completion (`completion_only_loss` defaults to on)"],
          ["Conversational", "`{\"messages\": [{\"role\": \"user\", ...}, {\"role\": \"assistant\", ...}]}`", "Every token, unless `assistant_only_loss=True` and the chat template supports it"],
          ["Conversational prompt–completion", "`{\"prompt\": [user msgs], \"completion\": [assistant msg]}`", "Only the assistant completion, rendered through the chat template — what this lab uses"],
        ]},
        { t: "callout", kind: "trap", title: "assistant_only_loss needs template support", x: [
          "TRL's `assistant_only_loss=True` finds assistant spans using `{% generation %}` markers inside the Jinja chat template. Many popular templates (including Qwen2.5's) do not contain them, so the option fails or cannot mask correctly.",
          "The portable route for single-turn data is the **conversational prompt–completion** format: the template is applied to both halves and only the completion is trained. For multi-turn data, use a template with generation markers or mask manually.",
        ]},

        { t: "h2", x: "Hyperparameters that matter" },
        { t: "table", head: ["Setting", "Full SFT (0.5B–8B)", "Why"], rows: [
          ["Learning rate", "1e-5 – 2e-5 (up to ~5e-5 for very small models)", "Pretrained weights are already good; large steps destroy them. LoRA uses ~10× higher (1e-4 – 2e-4) because it trains a tiny fresh update"],
          ["Epochs", "1–3", "Instruction data is small and repetitive; eval loss usually bottoms out by epoch 2–3"],
          ["Warmup", "3–10% of steps", "Adam's moment estimates are noisy at the start; warmup prevents an early loss spike"],
          ["Schedule", "Cosine or linear decay to ~0", "Lower LR at the end consolidates"],
          ["Effective batch", "32–128 sequences", "per_device × grad_accum × #GPUs; too small = noisy updates"],
          ["max_length", "1024–4096 tokens", "Cover your longest examples; truncation silently cuts answers"],
          ["Packing", "On for many short examples (needs position-aware attention)", "Removes padding waste; can 2–5× throughput"],
          ["Precision", "bf16 autocast with fp32 master weights (fp16 on T4)", "Pure-bf16 weights cannot represent tiny 1e-5-scale updates"],
        ]},
        { t: "callout", kind: "exam", title: "How this is tested", x: [
          "Expect “training loss falls but validation loss rises after epoch 2 — what now?” (stop earlier / fewer epochs / more data / lower LR), and “loss spikes in the first steps” (add warmup, lower LR).",
          "Know the order of magnitude: full fine-tuning LRs are ~1e-5, LoRA ~1e-4, DPO ~1e-6 – 1e-5 (full) — and that the prompt tokens are masked out of the SFT loss.",
        ]},

        { t: "h2", x: "The lab" },
        { t: "lab", title: "Full SFT of Qwen2.5-0.5B on Alpaca-cleaned", gpu: "1× 24 GB GPU (a free Colab T4 works with batch 4)", time: "45 min", x: "Instruction-tune the base `Qwen/Qwen2.5-0.5B` on 6,000 examples from `yahma/alpaca-cleaned` with TRL's SFTTrainer, training only on the assistant completions. Compare greedy outputs on the same prompts before and after, watch train and eval loss, then save and push the model.", steps: [
          "Install the libraries and load the tokenizer; confirm it ships a chat template.",
          "Convert Alpaca rows into the conversational prompt–completion format and hold out 200 examples for evaluation.",
          "Load the base model in fp32 and record its answers to three probe prompts.",
          "Configure SFTConfig (lr 2e-5, 2 epochs, warmup, cosine, eval every 50 steps) and train.",
          "Re-run the probe prompts, compare, and inspect the loss curves for overfitting.",
          "Save locally and push the model and tokenizer to the Hub.",
        ]},
        { t: "code", lang: "python", title: "1 · Setup and data", x: `# pip install -U "transformers>=4.56" "trl>=0.20" "datasets>=3" accelerate
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B"            # BASE model: it only continues text today
tok = AutoTokenizer.from_pretrained(MODEL_ID)
assert tok.chat_template is not None      # Qwen2.5 base tokenizers ship the ChatML template

raw = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42).select(range(6000))

def to_prompt_completion(ex):
    # Alpaca rows: instruction, optional input, output
    user = ex["instruction"] if not ex["input"] else f"{ex['instruction']}\\n\\n{ex['input']}"
    return {
        "prompt":     [{"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": ex["output"]}],
    }

ds = raw.map(to_prompt_completion, remove_columns=raw.column_names)
ds = ds.train_test_split(test_size=200, seed=42)

# Look at exactly what the model will be trained on. Note the <|im_start|>/<|im_end|> markers
# and the default system prompt the Qwen template inserts when you give none.
ex = ds["train"][0]
print(tok.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False))` },
        { t: "code", lang: "python", title: "2 · Baseline: what does the base model do?", x: `IM_END = tok.convert_tokens_to_ids("<|im_end|>")   # end-of-turn token in ChatML
PROBES = [
    "Give three tips for staying focused while studying.",
    "Rewrite in the passive voice: The committee approved the budget.",
    "What is the capital of Australia? Answer in one word.",
]

@torch.no_grad()
def chat(model, prompt, max_new_tokens=150):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens, do_sample=False,             # greedy: comparable runs
        eos_token_id=[IM_END, tok.eos_token_id], pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# Load in fp32: the Trainer keeps fp32 master weights and autocasts to bf16/fp16.
# (0.5B params x 16 bytes ~ 8 GB for weights + grads + Adam states.)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda")
before = {p: chat(model, p) for p in PROBES}
for p, a in before.items():
    print(f"### {p}\\n{a}\\n")   # expect rambling, repeated questions, or no stop` },
        { t: "code", lang: "python", title: "3 · Configure and train", x: `use_bf16 = torch.cuda.is_bf16_supported()      # Ampere+ -> bf16; T4 -> fp16

cfg = SFTConfig(
    output_dir="qwen2.5-0.5b-alpaca-sft",
    num_train_epochs=2,
    per_device_train_batch_size=8,           # T4: use 4 and grad_accum 8
    gradient_accumulation_steps=4,           # effective batch = 32 sequences
    learning_rate=2e-5,                      # full fine-tuning range: 1e-5 .. 2e-5
    lr_scheduler_type="cosine",
    warmup_steps=20,                         # ~5% of ~360 total steps
    weight_decay=0.0,
    max_length=1024,                         # longer examples are truncated
    packing=False,                           # see the note on packing below
    completion_only_loss=True,               # default for prompt-completion data; explicit here
    bf16=use_bf16, fp16=not use_bf16,
    gradient_checkpointing=True,             # trade ~30% compute for much less activation memory
    eval_strategy="steps", eval_steps=50,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",                        # or "wandb" / "tensorboard"
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    processing_class=tok,                    # trl>=0.12 name for the tokenizer argument
)

# Sanity-check the loss mask on one batch: prompt positions must be -100.
batch = next(iter(trainer.get_train_dataloader()))
print("masked fraction:", (batch["labels"] == -100).float().mean().item())

trainer.train()
print(trainer.evaluate())                    # eval_loss; perplexity = exp(eval_loss)` },
        { t: "code", lang: "python", title: "4 · Compare, save and push", x: `model = trainer.model.eval()
for p in PROBES:
    print(f"### {p}\\n--- before ---\\n{before[p][:300]}\\n--- after ---\\n{chat(model, p)}\\n")

# Loss curves: train vs eval
hist = trainer.state.log_history
train = [(h["step"], h["loss"]) for h in hist if "loss" in h]
evals = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
print("train:", train[-3:], "\\neval:", evals)

# Save weights + tokenizer (the tokenizer carries the chat template)
trainer.save_model("qwen2.5-0.5b-alpaca-sft/final")
tok.save_pretrained("qwen2.5-0.5b-alpaca-sft/final")

# Push (run \`hf auth login\` or huggingface_hub.login() first)
# trainer.model.push_to_hub("your-username/qwen2.5-0.5b-alpaca-sft")
# tok.push_to_hub("your-username/qwen2.5-0.5b-alpaca-sft")` },
        { t: "p", x: "What you should see: before training the base model often repeats the question, continues with another made-up “instruction”, or never stops. After two epochs it answers in the assistant turn and emits `<|im_end|>`. The masked fraction printed in step 3 is typically 30–60% — that is the prompt (and system prompt) being excluded from the loss." },

        { t: "h2", x: "Reading the curves" },
        { t: "table", head: ["What you see", "Likely cause", "What to do"], rows: [
          ["Train ↓, eval ↓ together, flattening", "Healthy", "Stop when eval flattens; keep the best checkpoint"],
          ["Train keeps ↓ (below ~0.5), eval ↑ after epoch 2", "[[overfitting|Overfitting]] / memorisation", "Fewer epochs, more or more diverse data, lower LR, keep the earlier checkpoint"],
          ["Loss spikes in the first 20 steps", "LR too high for the start", "More warmup, lower peak LR"],
          ["Loss barely moves", "LR too low, or labels all masked", "Check the masked fraction is < 100%, raise LR"],
          ["Loss is NaN on T4 (fp16)", "Overflow in fp16 activations", "Lower LR, keep fp32 master weights, or use a bf16-capable GPU"],
          ["Good loss, model never stops talking", "End-of-turn token not learned or not used at inference", "Check the template appends `<|im_end|>`; pass it as `eos_token_id` when generating"],
        ]},
        { t: "callout", kind: "warn", title: "Packing and cross-contamination", x: [
          "`packing=True` concatenates short examples into full `max_length` rows so no compute is spent on padding. Done naively, tokens of one example can attend to the previous example in the same row.",
          "TRL's packing is designed to be used with attention that respects per-example boundaries via position ids (e.g. `attn_implementation=\"flash_attention_2\"` or another padding-free kernel). Without such a kernel, leave packing off or accept some contamination.",
        ]},
        { t: "callout", kind: "trap", title: "bf16 weights + tiny learning rate = no learning", x: "bf16 has only 8 bits of mantissa: adding an update of 1e-5 × weight to a bf16 weight often rounds back to the same value. That is why mixed precision keeps fp32 **master weights** and only computes in bf16. Loading a model in bf16 and doing full fine-tuning with a 1e-5 LR without fp32 masters is a common reason for “the loss went down a bit and then nothing changed”." },

        { t: "h2", x: "The same thing at NVIDIA scale" },
        { t: "p", x: "Nothing changes conceptually for a 70B model — only the memory. Full SFT needs ~16 bytes/param (bf16 weights and grads, fp32 master copy, two Adam moments): 7B ⇒ ~112 GB, 70B ⇒ ~1.1 TB before activations. So you shard with FSDP/ZeRO-3 or use tensor + pipeline parallelism through [[nemo|NeMo]] / [[megatron|Megatron]] recipes (next lessons), and you run the same eval-before/eval-after discipline. The output checkpoint is then optimised with [[tensorrt-llm|TensorRT-LLM]] and served with [[nim|NIM]] or Triton." },

        { t: "h2", x: "Key takeaways" },
        { t: "ul", items: [
          "SFT is next-token cross-entropy on responses only; prompt tokens are fed in but masked with label −100.",
          "Use TRL's conversational prompt–completion format for portable completion-only loss; `assistant_only_loss` needs generation markers in the template.",
          "Full fine-tuning: LR 1e-5–2e-5, 1–3 epochs, 3–10% warmup, cosine decay, fp32 master weights with bf16/fp16 compute.",
          "Always evaluate before and after on the same prompts with greedy decoding, and watch eval loss for the overfitting turn.",
          "Teach and use the end-of-turn token; save the tokenizer with the model so the chat template travels with it.",
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: "lora-qlora",
      title: "Lab: LoRA and QLoRA on a single GPU",
      minutes: 35,
      outcomes: [
        "Derive LoRA's update ΔW = BA, its scaling α/r and its initialisation, and count LoRA parameters for a given model",
        "Choose rank, alpha and target modules for a LoRA run",
        "Explain the three QLoRA ingredients — NF4, double quantization and paged optimizers — and their memory effect",
        "Run a QLoRA fine-tune of Qwen2.5-1.5B with peft and TRL, then merge the adapter for deployment",
        "Compare the memory needed to fine-tune a 7B model with full FT, LoRA and QLoRA",
      ],
      blocks: [
        { t: "lede", x: "[[lora|LoRA]] is the reason fine-tuning stopped being a cluster-only activity. Instead of updating a d×k weight matrix, you learn a low-rank correction with a few thousand times fewer parameters; [[qlora|QLoRA]] additionally stores the frozen base in 4-bit. Result: a 7B model fine-tunes in about 10 GB of GPU memory, and the trained “model” is a 20–200 MB file you can hot-swap at serving time." },

        { t: "h2", x: "The math of LoRA" },
        { t: "p", x: "For a frozen pretrained weight W₀ ∈ ℝ^(d×k), LoRA learns **ΔW = B·A** with B ∈ ℝ^(d×r), A ∈ ℝ^(r×k) and [[lora-rank|rank]] r ≪ min(d, k). The forward pass becomes **h = W₀x + (α/r)·B·A·x**. Only A and B receive gradients and optimizer state; W₀ needs neither." },
        { t: "ul", items: [
          "**Initialisation:** A is random (Kaiming-uniform in peft), **B is zero**, so ΔW = 0 at step 0 and training starts exactly from the pretrained model.",
          "**Scaling:** the update is multiplied by [[lora-alpha|α/r]]. Keeping α proportional to r (common: α = 2r, or α = r) keeps the update magnitude roughly stable when you change r, so you do not have to retune the LR. rsLoRA uses α/√r instead, which behaves better at high ranks (`use_rslora=True`).",
          "**Why low rank works:** the weight change needed to adapt a pretrained model to a task has low “intrinsic rank” — a handful of directions carry most of it.",
          "**Zero inference overhead after merging:** W = W₀ + (α/r)BA can be folded into the base weight ([[merge-adapter|merge]]), giving a normal dense model with identical latency.",
        ]},
        { t: "h3", x: "Count the parameters" },
        { t: "p", x: "One 4096×4096 projection has 16.8M weights. LoRA with r = 16 adds r·(d + k) = 16 × 8192 = **131,072** — 0.78% of that matrix. For a Llama-2-7B-shaped model (32 layers, hidden 4096, MLP 11008) with LoRA on all seven linear layers:" },
        { t: "table", head: ["Module (per layer)", "Shape", "LoRA params at r = 16"], rows: [
          ["q, k, v, o projections (×4)", "4096 × 4096", "4 × 131,072 = 524,288"],
          ["gate, up projections (×2)", "4096 → 11008", "2 × 16 × (4096 + 11008) = 483,328"],
          ["down projection (×1)", "11008 → 4096", "16 × 15104 = 241,664"],
          ["Per layer", "", "1,249,280"],
          ["× 32 layers", "", "≈ 40.0M trainable ≈ 0.6% of 6.7B"],
        ]},
        { t: "callout", kind: "tip", title: "Choosing r, alpha and target modules", x: [
          "Start with r = 16, α = 32, dropout 0.05, `target_modules=\"all-linear\"` (attention and MLP). Targeting only q and v (the original paper) is cheaper but usually a bit worse.",
          "Raise r (32–64) for large distribution shifts or many tasks in one adapter; drop to r = 4–8 for simple format/style changes. Rank matters less than data quality and which modules you target.",
          "LoRA learning rate: 1e-4 – 2e-4 (about 10× full fine-tuning).",
        ]},

        { t: "h2", x: "QLoRA: three tricks" },
        { t: "vs", cards: [
          { h: "NF4 (4-bit NormalFloat)", items: [
            "[[nf4|NF4]] places its 16 levels at quantiles of a normal distribution — information-theoretically well matched to normally-distributed weights",
            "Block-wise: every 64 weights share one absmax scale",
            "Weights are dequantized to bf16 on the fly for each matmul (compute dtype)",
            "Only the frozen base is 4-bit; LoRA A and B stay in bf16/fp32",
          ]},
          { h: "Double quantization", items: [
            "[[double-quantization|Quantize the quantization constants]] too: the fp32 per-block scales are themselves stored in 8 bits, in blocks of 256",
            "Overhead drops from 32/64 = 0.5 bits/param to 8/64 + 32/(64·256) ≈ 0.127 bits/param",
            "Saves ≈ 0.37 bits/param ≈ 0.33 GB on 7B, ≈ 3 GB on 65B",
          ]},
          { h: "Paged optimizers", items: [
            "[[paged-optimizer|Paged optimizer]] states live in CUDA unified memory and are paged to CPU RAM when the GPU runs out",
            "Absorbs the memory spikes of long sequences with gradient checkpointing instead of OOM-ing",
            "In Transformers: `optim=\"paged_adamw_8bit\"` or `\"paged_adamw_32bit\"`",
          ]},
        ]},
        { t: "p", x: "All three are implemented in the [[bitsandbytes|bitsandbytes]] library and switched on through `BitsAndBytesConfig`. The QLoRA paper's headline: fine-tuning a 65B model on a single 48 GB GPU while matching 16-bit full fine-tuning quality on its benchmarks." },

        { t: "h2", x: "Memory: full FT vs LoRA vs QLoRA for 7B" },
        { t: "table", head: ["Component (7B)", "Full FT (mixed precision, AdamW)", "LoRA (bf16 base, r=16)", "QLoRA (NF4 base, r=16)"], rows: [
          ["Base weights", "bf16 14 GB + fp32 master 28 GB", "bf16 14 GB (frozen)", "NF4 ≈ 3.5 GB + ≈ 0.1 GB constants"],
          ["Gradients", "14 GB (bf16)", "≈ 0.08 GB (only 40M LoRA params)", "≈ 0.08 GB"],
          ["Optimizer states", "56 GB (Adam m, v in fp32)", "≈ 0.3 GB", "≈ 0.3 GB (or less, 8-bit paged)"],
          ["Subtotal before activations", "≈ 112 GB (16 B/param)", "≈ 15 GB", "≈ 4–5 GB"],
          ["Fits on", "Multi-GPU with FSDP/ZeRO-3 (2+ × 80 GB)", "1 × 24–40 GB", "1 × 12–16 GB (T4, RTX 4070 Ti, …)"],
        ]},
        { t: "callout", kind: "exam", title: "How this is tested", x: [
          "Memory arithmetic: full FT ≈ 16 bytes/param; LoRA ≈ 2 bytes/param (frozen bf16) + small; QLoRA ≈ 0.5 bytes/param + small. Activations come on top and scale with batch × sequence length.",
          "Know which knob does what: r (capacity), α (scale of the update), target_modules (where), NF4 (4-bit storage), double quant (smaller constants), paged optimizer (absorb spikes), compute dtype (matmul precision).",
          "“Many customers, one base model” → one base + many LoRA adapters (multi-LoRA serving) rather than many merged copies.",
        ]},

        { t: "h2", x: "The lab" },
        { t: "lab", title: "QLoRA fine-tune of Qwen2.5-1.5B, then merge", gpu: "Free Colab T4 (16 GB) or any 12 GB+ GPU", time: "45 min", x: "Load `Qwen/Qwen2.5-1.5B` in 4-bit NF4 with double quantization, attach a rank-16 LoRA to every linear layer, train on the same Alpaca prompt–completion data as the SFT lab with a paged 8-bit optimizer, measure peak memory, then reload the base in bf16 and merge the adapter into a standalone model.", steps: [
          "Install bitsandbytes and peft; load the base model with a 4-bit NF4 BitsAndBytesConfig and check its footprint.",
          "Prepare the model for k-bit training and define a LoraConfig (r=16, alpha=32, all-linear).",
          "Build the prompt–completion dataset and train with SFTTrainer (lr 2e-4, paged_adamw_8bit).",
          "Print trainable parameters (~18.5M, ~1.2%) and peak GPU memory.",
          "Save the adapter (tens of MB), test it, then merge into a bf16 base with merge_and_unload and save.",
        ]},
        { t: "code", lang: "python", title: "1 · Load the base in 4-bit and attach LoRA", x: `# pip install -U "transformers>=4.56" "trl>=0.20" "peft>=0.13" "datasets>=3" accelerate bitsandbytes
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B"
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat4 (vs plain "fp4")
    bnb_4bit_use_double_quant=True,       # quantize the per-block scales as well
    bnb_4bit_compute_dtype=compute_dtype, # dequantize to this dtype for each matmul
)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map={"": 0}, dtype=compute_dtype,
)
print(f"4-bit model footprint: {model.get_memory_footprint() / 1e9:.2f} GB")  # ~1.2 GB vs ~3.1 GB in bf16

# Casts norms to fp32, enables input grads so checkpointing works with frozen weights
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules="all-linear",          # q,k,v,o,gate,up,down (lm_head excluded)
    bias="none", task_type="CAUSAL_LM",
    # use_dora=True,                      # DoRA: magnitude/direction split, slower, often a bit better
)` },
        { t: "code", lang: "python", title: "2 · Data and training", x: `raw = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=0).select(range(6000))

def to_prompt_completion(ex):
    user = ex["instruction"] if not ex["input"] else f"{ex['instruction']}\\n\\n{ex['input']}"
    return {"prompt": [{"role": "user", "content": user}],
            "completion": [{"role": "assistant", "content": ex["output"]}]}

ds = raw.map(to_prompt_completion, remove_columns=raw.column_names).train_test_split(test_size=200, seed=0)

cfg = SFTConfig(
    output_dir="qwen2.5-1.5b-qlora",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,         # effective batch 32
    learning_rate=2e-4,                    # LoRA range: 1e-4 .. 2e-4
    lr_scheduler_type="cosine",
    warmup_steps=10,
    optim="paged_adamw_8bit",              # paged + 8-bit optimizer states
    max_length=1024,
    bf16=compute_dtype == torch.bfloat16, fp16=compute_dtype == torch.float16,
    gradient_checkpointing=True,
    eval_strategy="steps", eval_steps=50,
    logging_steps=10, save_strategy="no", report_to="none",
)

trainer = SFTTrainer(
    model=model, args=cfg, peft_config=lora,
    train_dataset=ds["train"], eval_dataset=ds["test"], processing_class=tok,
)
trainer.model.print_trainable_parameters()   # ~18.5M trainable of ~1.56B (~1.2%)

torch.cuda.reset_peak_memory_stats()
trainer.train()
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

trainer.save_model("qwen2.5-1.5b-qlora/adapter")   # adapter_config.json + adapter_model.safetensors` },
        { t: "code", lang: "python", title: "3 · Merge the adapter for deployment", x: `# Merge into a *bf16* copy of the base, not the 4-bit one: merging into NF4 would
# re-quantize the sum and lose the update. Do this on a machine with ~4 GB free (CPU is fine).
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "qwen2.5-1.5b-qlora/adapter")
merged = model.merge_and_unload()          # W = W0 + (alpha/r) * B @ A, LoRA layers removed

merged.save_pretrained("qwen2.5-1.5b-qlora-merged")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B").save_pretrained("qwen2.5-1.5b-qlora-merged")
# Re-run your probe prompts on the merged model: it was trained against dequantized NF4
# weights, so verify quality on the bf16 merge before shipping it.` },
        { t: "callout", kind: "trap", title: "Merge or keep separate?", x: [
          "**Merge** when you ship one fine-tuned model: no adapter overhead at inference, works with any engine (vLLM, TensorRT-LLM, NIM).",
          "**Keep adapters separate** when one base serves many tasks or tenants: a single base in GPU memory plus small adapters selected per request (multi-LoRA serving in vLLM, NIM, TensorRT-LLM). Merging here would mean N full model copies.",
          "You cannot merge two different adapters into the same base and expect both behaviours to survive intact — combining adapters is its own research area.",
        ]},

        { t: "h2", x: "Beyond vanilla LoRA" },
        { t: "table", head: ["Variant", "Idea", "When"], rows: [
          ["[[dora|DoRA]]", "Decompose W into magnitude × direction; LoRA updates the direction, a separate vector learns the magnitude", "Often closes part of the gap to full FT at the same rank; peft `use_dora=True`; slower training"],
          ["rsLoRA", "Scale by α/√r instead of α/r", "High ranks (64–256) where α/r makes updates too small"],
          ["LoRA+", "Higher LR for B than for A", "Faster convergence at the same budget"],
          ["QLoRA", "LoRA on a 4-bit NF4 frozen base", "Memory-limited GPUs; slightly slower steps than LoRA due to dequantization"],
        ]},

        { t: "h2", x: "Key takeaways" },
        { t: "ul", items: [
          "LoRA: h = W₀x + (α/r)BAx; B starts at zero so training starts from the base model; parameters per matrix = r·(d + k).",
          "Defaults that work: r = 16, α = 32, all-linear targets, LR 1e-4–2e-4, 1–3 epochs.",
          "QLoRA = NF4 4-bit frozen base + double quantization + paged optimizers; 7B fine-tuning drops from ~112 GB (full) to ~5 GB plus activations.",
          "Merge into a bf16 base for single-model deployment; keep adapters separate for multi-tenant multi-LoRA serving.",
          "DoRA and rsLoRA are drop-in peft options when plain LoRA plateaus.",
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: "alignment",
      title: "Preference alignment: RLHF, PPO, DPO and GRPO",
      minutes: 28,
      outcomes: [
        "Describe the three-stage RLHF pipeline — SFT, reward model, PPO with a KL penalty — and why each stage exists",
        "Explain the DPO loss and run DPOTrainer on a preference dataset",
        "Compare DPO with ORPO, KTO and SimPO by data requirements and reference-model needs",
        "Explain GRPO's group-relative advantage and set up GRPOTrainer with verifiable rewards",
        "Recognise reward hacking and the controls that limit it",
      ],
      blocks: [
        { t: "lede", x: "SFT teaches a model to imitate good answers. Alignment teaches it to **prefer** better answers over worse ones — more helpful, more honest, safer, or, for reasoning models, more often correct. This lesson walks from classic [[rlhf|RLHF]] with PPO to the methods most teams actually run today: [[dpo|DPO]] for preference pairs and [[grpo|GRPO]] for verifiable rewards." },

        { t: "h2", x: "Classic RLHF: SFT → reward model → PPO" },
        { t: "flow", steps: [
          { b: "1 · SFT", s: "Fine-tune a base model on demonstrations. This becomes the starting policy π and the frozen [[reference-model|reference]] π_ref." },
          { b: "2 · Reward model", s: "Collect human comparisons (A better than B). Train a [[reward-model|reward model]] r_φ(x, y) — usually the SFT model with a scalar head — with the Bradley–Terry loss −log σ(r_φ(x, y_w) − r_φ(x, y_l))." },
          { b: "3 · PPO", s: "Sample responses from π, score them with r_φ, and update π with [[ppo|PPO]] to maximise r_φ(x, y) − β·KL(π ‖ π_ref). A value (critic) model estimates advantages." },
        ]},
        { t: "p", x: "The [[kl-penalty|KL penalty]] is the crucial piece: without it, the policy drifts to whatever text exploits the reward model's blind spots, and fluency collapses. PPO's clipped objective additionally limits how far each update can move the policy. The cost is complexity: four models in memory (policy, reference, reward, value), on-policy generation inside the training loop, and many sensitive hyperparameters." },
        { t: "callout", kind: "exam", title: "How this is tested", x: [
          "Know the pipeline order (SFT → RM → PPO), what the KL term prevents (drift and reward hacking), and that PPO needs a value/critic model while GRPO and DPO do not.",
          "DPO needs (prompt, chosen, rejected) pairs and a reference model, but **no reward model and no sampling** during training. GRPO needs a reward function (often programmatic) and samples a group of answers per prompt.",
        ]},

        { t: "h2", x: "DPO: preference learning without RL" },
        { t: "p", x: "Direct Preference Optimization shows that the RLHF objective (reward minus β·KL) has a closed-form optimal policy, so the reward can be written in terms of the policy itself: the implicit reward of a response is **β·log(π_θ(y|x) / π_ref(y|x))**. Plug that into the Bradley–Terry preference loss and the reward model disappears:" },
        { t: "p", x: "**L_DPO = −log σ( β·[ log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x) ] )**" },
        { t: "ul", items: [
          "Intuition: raise the likelihood of the chosen answer y_w **relative to the reference**, and lower the rejected y_l relative to the reference, until the margin is large.",
          "β (typically 0.1; range ~0.01–0.5) plays the KL role: higher β keeps the policy closer to π_ref.",
          "It is just a classification-style loss over log-probs — stable, cheap, no generation in the loop. Two models: policy and frozen reference (with LoRA, the reference is the base with the adapter switched off).",
          "Watch the logged `rewards/chosen`, `rewards/rejected`, `rewards/margins` and `rewards/accuracies`: margins should grow; if chosen rewards also fall steeply, the model is just lowering the probability of everything.",
        ]},
        { t: "code", lang: "python", title: "DPO with TRL on UltraFeedback pairs (LoRA)", x: `# pip install -U "transformers>=4.56" "trl>=0.20" "peft>=0.13" "datasets>=3"
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # DPO starts from an SFT'd model, never a raw base
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

# Conversational preference pairs: chosen / rejected message lists
# (TRL extracts the shared prompt if it is not a separate column).
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:5000]")

cfg = DPOConfig(
    output_dir="qwen0.5b-dpo",
    beta=0.1,                          # strength of the implicit KL constraint
    learning_rate=5e-6,                # DPO uses small LRs (full: ~5e-7..5e-6; LoRA: up to ~5e-5)
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_length=1024,
    bf16=True, logging_steps=10, report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,                    # with peft_config, the reference = base with adapter disabled
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()                        # watch rewards/margins and rewards/accuracies rise` },

        { t: "h2", x: "The DPO family: ORPO, KTO, SimPO" },
        { t: "table", head: ["Method", "Data", "Reference model?", "Key idea"], rows: [
          ["[[dpo|DPO]]", "Pairs (chosen, rejected)", "Yes (frozen)", "Implicit reward = β·log-ratio to reference; logistic loss on the margin"],
          ["[[orpo|ORPO]]", "Pairs", "No", "Adds an odds-ratio penalty to the normal SFT loss — SFT and preference alignment in one stage"],
          ["[[kto|KTO]]", "Unpaired: each answer labeled good or bad", "Yes", "Prospect-theory-inspired loss; useful when you only have thumbs-up/down feedback"],
          ["[[simpo|SimPO]]", "Pairs", "No", "Reward = length-normalised average log-prob plus a target margin γ; cheaper, counters length bias"],
          ["IPO", "Pairs", "Yes", "Squared loss variant of DPO that is less prone to overfitting deterministic preferences"],
        ]},
        { t: "p", x: "TRL ships trainers for these (e.g. `DPOTrainer` with a `loss_type` option, `KTOTrainer`, and SimPO-style losses via configuration). The exam level is: which one fits which data, and which ones drop the reference model." },

        { t: "h2", x: "GRPO and verifiable rewards" },
        { t: "p", x: "Group Relative Policy Optimization (introduced with DeepSeekMath and used for DeepSeek-R1-style reasoning training) keeps the RL loop but removes PPO's value model. For each prompt it samples a **group** of G answers, scores each, and uses the group itself as the baseline: **Aᵢ = (rᵢ − mean(r)) / std(r)**. Answers better than their siblings are reinforced, worse ones discouraged, with a clipped ratio and optional KL to the reference like PPO." },
        { t: "vs", cards: [
          { h: "Why GRPO took off", items: [
            "No critic: one fewer model the size of the policy in memory",
            "Pairs naturally with [[verifiable-rewards|verifiable rewards]]: math answer matches, unit tests pass, JSON validates",
            "Drives long chain-of-thought “reasoning” behaviour to emerge from outcome rewards alone",
          ]},
          { h: "What it costs", items: [
            "Generation-heavy: G samples per prompt every step (fast engines like vLLM are used for rollouts)",
            "If all G answers get the same reward the advantage is 0 — no learning signal; choose prompts of the right difficulty",
            "Reward functions must be robust or the model will exploit them",
          ]},
        ]},
        { t: "code", lang: "python", title: "GRPO sketch on GSM8K with a correctness reward", x: `# pip install -U "trl>=0.20" "datasets>=3"   (optionally: vllm, then use_vllm=True)
import re
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

SYSTEM = "Solve the problem step by step. End with a final line of the form: #### <number>"

def to_prompt(ex):
    return {
        "prompt": [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": ex["question"]}],
        "target": ex["answer"].split("####")[-1].strip().replace(",", ""),
    }

ds = load_dataset("openai/gsm8k", "main", split="train")
ds = ds.map(to_prompt, remove_columns=ds.column_names)

ANSWER = re.compile(r"####\\s*(-?[\\d,]+(?:\\.\\d+)?)")

# Reward functions receive the completions plus every extra dataset column as kwargs.
def correctness_reward(completions, target, **kwargs):
    out = []
    for c, t in zip(completions, target):
        m = ANSWER.search(c[0]["content"])         # conversational: a list with one message
        out.append(1.0 if m and m.group(1).replace(",", "") == t else 0.0)
    return out

def format_reward(completions, **kwargs):          # small shaping bonus for the format
    return [0.1 if ANSWER.search(c[0]["content"]) else 0.0 for c in completions]

cfg = GRPOConfig(
    output_dir="qwen0.5b-grpo-gsm8k",
    num_generations=8,                 # group size G
    per_device_train_batch_size=8,     # must be divisible by num_generations (single GPU)
    max_completion_length=256,
    learning_rate=1e-6,
    bf16=True, logging_steps=5, report_to="none",
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[correctness_reward, format_reward],
    args=cfg,
    train_dataset=ds,
)
trainer.train()    # watch the mean reward climb and completion length change` },

        { t: "h2", x: "Reward hacking" },
        { t: "p", x: "[[reward-hacking|Reward hacking]] is the policy finding a way to score highly without doing what you meant. Learned reward models are especially exploitable because they are just another neural network with blind spots." },
        { t: "table", head: ["Symptom", "Typical cause", "Control"], rows: [
          ["Answers get longer and longer", "Reward model / judges prefer length", "Length-normalised rewards (SimPO), length penalties, length-controlled evaluation"],
          ["Fluent nonsense with a high reward", "Policy drifted off the RM's training distribution", "KL penalty to the reference, early stopping, refresh the RM with on-policy samples"],
          ["“#### 42” with no reasoning, or tests special-cased", "Verifier checks too little", "Stricter verifiers, hidden tests, format and reasoning checks"],
          ["Sycophancy: agrees with the user's wrong claim", "Human raters reward agreement", "Better rater guidelines, targeted preference data, evals for sycophancy"],
        ]},
        { t: "callout", kind: "warn", title: "Never trust the training reward alone", x: "A rising reward curve proves only that the policy is getting better at the reward. Always measure on held-out evaluations the policy was not optimised against — benchmarks, human or LLM-judge review, and safety tests — and read samples." },

        { t: "h2", x: "Key takeaways" },
        { t: "ul", items: [
          "RLHF = SFT → reward model (Bradley–Terry on human comparisons) → PPO maximising reward − β·KL to the SFT reference.",
          "DPO folds the reward into the policy's log-ratio to a reference: no reward model, no sampling, just a logistic loss on preference pairs; β ≈ 0.1, LR ~1e-6–5e-6 for full training.",
          "ORPO and SimPO drop the reference model; KTO works with unpaired good/bad labels.",
          "GRPO samples a group per prompt and uses group-normalised rewards as advantages — no critic; ideal with verifiable rewards for reasoning.",
          "Expect reward hacking; constrain with KL, robust verifiers and independent evaluation.",
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: "nemo-customization",
      title: "NeMo customization, p-tuning, adapters and avoiding catastrophic forgetting",
      minutes: 22,
      outcomes: [
        "Map the NVIDIA customization stack — NeMo Framework recipes, NeMo-RL, NeMo Customizer — to the jobs each does",
        "Launch a NeMo 2.0 fine-tuning recipe with LoRA or full SFT",
        "Distinguish prompt tuning, p-tuning, prefix tuning, adapters, IA3 and LoRA by where they add parameters",
        "Diagnose catastrophic forgetting and overfitting, and apply mitigations",
        "Choose fine-tuning hyperparameters by method and model size",
      ],
      blocks: [
        { t: "lede", x: "Everything you have done with Hugging Face TRL has an NVIDIA-scale counterpart, and the exam is written in NVIDIA's vocabulary. This lesson maps the stack, fills in the older PEFT methods NeMo popularised ([[p-tuning|p-tuning]], [[adapters|adapters]], [[ia3|IA3]]), and covers the failure mode every fine-tuner meets eventually: [[catastrophic-forgetting|catastrophic forgetting]]." },

        { t: "h2", x: "The NVIDIA customization stack" },
        { t: "table", head: ["Component", "What it is", "Use it for"], rows: [
          ["[[nemo|NeMo Framework]] (NeMo 2.0)", "PyTorch Lightning + [[megatron|Megatron-Core]] training framework with Python-based configs and ready-made recipes per model family, launched with NeMo-Run", "Pretraining, continued pretraining, SFT and PEFT at multi-GPU/multi-node scale with tensor/pipeline/sequence parallelism"],
          ["[[nemo-rl|NeMo-RL]] (successor to NeMo-Aligner)", "Post-training library for alignment and RL", "DPO, GRPO, reward-model training and SFT at scale, with fast rollout engines"],
          ["[[nemo-customizer|NeMo Customizer]]", "A NeMo microservice exposing fine-tuning as a REST API on Kubernetes", "Platform teams offering “submit a dataset, get a LoRA/SFT model” to app teams; outputs deploy to NIM"],
          ["NeMo Curator", "GPU-accelerated data curation", "Building the SFT/pretraining datasets (Module 2)"],
          ["[[nim|NIM]] / [[tensorrt-llm|TensorRT-LLM]]", "Optimised inference microservices / engine", "Serving the merged model or base + LoRA adapters"],
        ]},
        { t: "p", x: "NVIDIA keeps evolving where LLM recipes live (newer releases also offer Megatron-Bridge for Hugging Face ↔ Megatron checkpoint conversion and NeMo AutoModel for training Hugging Face models directly), so check the current docs for entry points. The concepts and the exam vocabulary — recipes, PEFT schemes, tensor/pipeline parallel sizes, `.nemo` checkpoints, Customizer jobs — are stable." },
        { t: "code", lang: "python", title: "NeMo 2.0: a LoRA fine-tuning recipe launched with NeMo-Run", x: `# Inside the NeMo Framework container from NGC (nvcr.io/nvidia/nemo:<tag>)
import nemo_run as run
from nemo.collections import llm

recipe = llm.llama3_8b.finetune_recipe(
    name="llama3_8b_lora",
    dir="/workspace/checkpoints",     # where logs and checkpoints go
    num_nodes=1,
    num_gpus_per_node=8,
    peft_scheme="lora",               # "none" -> full SFT with the same recipe
)
# Recipes are plain config trees you can override before launching:
recipe.trainer.max_steps = 1000
recipe.optim.config.lr = 1e-4         # LoRA-range LR; ~1e-5 for full SFT
# The default data module is SQuAD; point recipe.data at your own JSONL instead.

run.run(recipe, executor=run.LocalExecutor(ntasks_per_node=8, launcher="torchrun"))
# On a cluster, swap LocalExecutor for a Slurm or Kubernetes executor — the recipe is unchanged.` },
        { t: "callout", kind: "tip", title: "Customizer in one paragraph", x: "With NeMo Customizer you upload a dataset (to the NeMo data store), POST a customization job naming a base model config, a training type (e.g. SFT) and a fine-tuning type (e.g. LoRA) plus hyperparameters, poll the job, and the resulting adapter or model is registered so a NIM can serve it. It packages the recipes above behind an API so application teams never touch Megatron configs." },

        { t: "h2", x: "PEFT methods you will see in NeMo" },
        { t: "table", head: ["Method", "Where the trainable parameters live", "Size", "Notes"], rows: [
          ["[[prompt-tuning|Prompt tuning]]", "A few dozen soft-prompt embedding vectors prepended at the **input layer only**", "Tiny (e.g. 100 × hidden)", "Weak for small models; competitive only at very large scale"],
          ["[[p-tuning|P-tuning]]", "Virtual-token embeddings produced by a small **prompt encoder** (LSTM or MLP), prepended at the input", "Small (encoder discarded after training; virtual tokens cached in a prompt table)", "The encoder stabilises training vs raw prompt tuning; base weights frozen"],
          ["[[prefix-tuning|Prefix tuning]] / p-tuning v2", "Trainable key/value prefixes at **every layer**", "Small", "Much stronger than input-only soft prompts for smaller models"],
          ["[[adapters|Adapters]]", "Small bottleneck MLPs (down-project → nonlinearity → up-project) inserted after attention and/or FFN sublayers", "~1–5% of params", "Add a little inference latency unless fused; the classic Houlsby design"],
          ["[[ia3|IA3]]", "Learned vectors that **rescale** keys, values and FFN activations element-wise", "≪ 0.1%", "Fewest parameters of all; no new matrices"],
          ["[[lora|LoRA]] / [[dora|DoRA]]", "Low-rank BA update beside existing linear layers", "~0.1–2%", "Mergeable, zero added latency; today's default"],
        ]},
        { t: "callout", kind: "exam", title: "How this is tested", x: [
          "Expect “which method adds trainable virtual tokens through an LSTM/MLP prompt encoder?” (p-tuning), “which inserts bottleneck layers?” (adapters), “which scales activations with learned vectors?” (IA3), “which can be merged with zero inference overhead?” (LoRA).",
          "All of these freeze the base model, so all reduce — but do not eliminate — forgetting.",
        ]},

        { t: "h2", x: "Catastrophic forgetting" },
        { t: "p", x: "Gradient descent on a narrow dataset moves weights toward that dataset with no term that preserves anything else. A model fine-tuned hard on SQL generation can lose general chat ability, multilingual skill, maths, or its safety refusals. Signs: task metric improves while general benchmarks (MMLU-style, HellaSwag, IFEval) and your general-capability probes drop, and outputs become repetitive or format-locked on unrelated prompts." },
        { t: "table", head: ["Mitigation", "Why it works", "Cost"], rows: [
          ["Lower learning rate", "Smaller moves away from the pretrained optimum", "Slower task learning"],
          ["Fewer epochs / early stopping", "Stop before over-specialising", "Needs a held-out eval"],
          ["PEFT (LoRA, adapters) instead of full FT", "Base weights frozen; the update is low-capacity and removable", "Slightly lower task ceiling"],
          ["Replay / data mixing", "Mix 5–30% general instruction or pretraining data into the fine-tuning set", "More tokens to train on"],
          ["Regularise toward the original", "KL to the reference (as in RLHF/DPO), or weight-space penalties like EWC", "Extra compute or complexity"],
          ["Evaluate on general benchmarks every run", "Catch regressions before shipping", "Eval time"],
        ]},
        { t: "callout", kind: "trap", title: "Overfitting is not the same as forgetting", x: [
          "**Overfitting:** the model memorises *your* training set — train loss keeps falling, *task* validation loss rises, outputs echo training examples. Fix with fewer epochs, more/diverse data, lower LR, dropout.",
          "**Forgetting:** task validation can look great while *other* capabilities degrade. You only see it if you evaluate outside the task.",
        ]},

        { t: "h2", x: "Hyperparameter cheat sheet" },
        { t: "table", head: ["Method", "Learning rate", "Epochs", "Other"], rows: [
          ["Full SFT (0.5–70B)", "1e-5 – 2e-5 (lower for larger models)", "1–3", "Warmup 3–10%, cosine, effective batch 32–128, weight decay 0–0.1"],
          ["LoRA / QLoRA", "1e-4 – 2e-4", "1–3", "r 8–64, α = r…2r, dropout 0–0.1, all-linear targets"],
          ["P-tuning / prompt tuning", "~1e-4 – 1e-3 (higher; few params)", "Until val plateaus", "Virtual tokens 10–100"],
          ["DPO", "~5e-7 – 5e-6 full; up to ~5e-5 LoRA", "1–3", "β 0.05–0.5 (0.1 default)"],
          ["GRPO", "~1e-6", "By steps", "G = 4–16 generations, KL β small or 0, max completion length matched to task"],
          ["Continued pretraining", "Re-warm to ~1/10 of the original peak, then decay", "1 pass over the domain data", "Mix in general data (replay) to limit forgetting"],
        ]},

        { t: "h2", x: "Key takeaways" },
        { t: "ul", items: [
          "NeMo Framework recipes cover pretraining/SFT/PEFT at scale; NeMo-RL covers DPO/GRPO-style post-training; NeMo Customizer wraps fine-tuning as a Kubernetes microservice; NIM serves the result.",
          "Prompt tuning = input soft prompts; p-tuning = soft prompts from an LSTM/MLP encoder; prefix tuning = per-layer K/V prefixes; adapters = bottleneck MLPs; IA3 = learned rescaling vectors; LoRA = mergeable low-rank update.",
          "Catastrophic forgetting shows up outside the task metric: always evaluate general benchmarks too.",
          "Mitigate with lower LR, fewer epochs, PEFT, data replay and KL-style regularisation.",
          "Learning rate scales inversely with the number of trainable parameters: full FT ~1e-5, LoRA ~1e-4, soft prompts higher still.",
        ]},
      ],
    },
  ],

  quiz: [
    { id: "f01", domain: "finetuning",
      scenario: "A retailer's support assistant must answer questions about return policies and stock levels that change weekly. Prompting a 8B instruct model with the policy text works but answers are sometimes stale.",
      q: "Which approach best fixes the staleness problem?",
      opts: [
        "Full fine-tuning on the current policy documents every week",
        "Retrieval-augmented generation over a policy and inventory index that is refreshed as data changes",
        "LoRA fine-tuning on last month's chat transcripts",
        "Continued pretraining on the company website",
      ],
      a: 1,
      why: "Staleness is a **knowledge** problem with frequently changing facts. [[rag|RAG]] injects current documents at query time, needs no retraining and can cite sources.",
      distract: "Weekly **full fine-tuning** is expensive, still lags, and is unreliable for facts. **LoRA on transcripts** teaches past behaviour, including stale answers. **Continued pretraining** is for large domain gaps and still freezes knowledge at training time." },

    { id: "f02", domain: "finetuning",
      q: "In LoRA, why is the B matrix initialised to zero while A is random?",
      opts: [
        "So that ΔW = BA is zero at the start and training begins exactly from the pretrained model",
        "So that B can be skipped during the backward pass to save memory",
        "Because a zero matrix has the lowest possible rank, which forces r to stay small",
        "So that the scaling factor α/r cancels out during the first optimizer step",
      ],
      a: 0,
      why: "With B = 0 the product BA is zero, so the adapted model is initially identical to the base; A being random ensures the gradient of B is non-zero, so learning can start.",
      distract: "B still receives gradients — it is the first thing that learns. Rank is set by the shapes (r), not by the initial values. The α/r factor is a constant multiplier that never cancels." },

    { id: "f03", domain: "finetuning",
      scenario: "You must fine-tune a 7B model on a single 16 GB GPU.",
      q: "Which configuration is most likely to fit?",
      opts: [
        "Full fine-tuning in bf16 with AdamW and gradient checkpointing",
        "LoRA on a bf16 base with r = 64 on all linear layers",
        "Full fine-tuning in fp32 with an 8-bit optimizer",
        "QLoRA: NF4 4-bit base with double quantization, LoRA adapters and a paged optimizer",
      ],
      a: 3,
      why: "[[qlora|QLoRA]] stores the frozen 7B base in ~3.5 GB ([[nf4|NF4]]), trains only small adapters, and pages optimizer states, leaving room for activations within 16 GB.",
      distract: "**Full FT** needs ~16 bytes/param ≈ 112 GB (fp32 even more). **LoRA on a bf16 base** needs ~14 GB just for the frozen weights, leaving almost nothing for activations." },

    { id: "f04", domain: "finetuning",
      q: "What does the β parameter control in DPO?",
      opts: [
        "The ratio of chosen to rejected examples sampled in each batch",
        "The learning-rate warmup length",
        "How strongly the policy is kept close to the reference model (the implicit KL constraint)",
        "The temperature used to sample new responses during training",
      ],
      a: 2,
      why: "In [[dpo|DPO]] the implicit reward is β·log(π_θ/π_ref). β is the KL-regularisation strength from the RLHF objective: larger β keeps the policy closer to the reference.",
      distract: "Batches always contain matched pairs, so there is no ratio. Warmup is a separate scheduler setting. DPO does **not** sample during training — it scores fixed pairs." },

    { id: "f05", domain: "finetuning",
      scenario: "After 3 epochs of full SFT on a 20k-example SQL dataset, a model's SQL accuracy rose from 41% to 78%, but its MMLU score fell from 62 to 49 and it answers general questions with SQL fragments.",
      q: "Which change most directly addresses the main problem?",
      opts: [
        "Train for more epochs so the model generalises beyond SQL",
        "Raise the learning rate so training converges in fewer steps",
        "Increase max_length so longer general answers fit",
        "Mix general instruction data into the training set and lower the learning rate or switch to LoRA",
      ],
      a: 3,
      why: "This is [[catastrophic-forgetting|catastrophic forgetting]]. Replay of general data, a lower LR, and freezing the base with PEFT all limit how far weights move away from the general-purpose optimum.",
      distract: "**More epochs** and **a higher LR** push further toward SQL and worsen forgetting. **max_length** has nothing to do with lost capabilities." },

    { id: "f06", domain: "finetuning",
      q: "Which PEFT method trains virtual-token embeddings generated by a small LSTM or MLP prompt encoder, with the base model frozen?",
      opts: [
        "P-tuning",
        "IA3",
        "Adapters",
        "LoRA",
      ],
      a: 0,
      why: "[[p-tuning|P-tuning]] uses a prompt encoder (LSTM/MLP) to produce continuous virtual-token embeddings prepended to the input; after training those embeddings are cached and the encoder can be dropped.",
      distract: "**IA3** learns rescaling vectors on activations. **Adapters** insert bottleneck MLP layers inside each block. **LoRA** adds low-rank matrices beside linear layers." },

    { id: "f07", domain: "finetuning",
      q: "Why is GRPO cheaper in memory than PPO for RL fine-tuning of an LLM?",
      opts: [
        "It does not need a reference model, so the KL term is dropped",
        "It replaces the learned value (critic) model with a baseline computed from a group of sampled answers to the same prompt",
        "It trains on fixed preference pairs instead of generating samples",
        "It quantises the policy to 4-bit during rollouts",
      ],
      a: 1,
      why: "[[grpo|GRPO]] computes each sample's advantage as its reward normalised by the mean and standard deviation of its group, removing PPO's critic — a model as large as the policy.",
      distract: "GRPO can still use a KL term to a reference. Fixed preference pairs describe **DPO**. Quantisation is not part of the algorithm." },

    { id: "f08", domain: "finetuning",
      scenario: "A 0.5B base model is being instruction-tuned with full SFT. Training loss falls steadily to 0.35 by epoch 4. Eval loss was lowest at the end of epoch 2 and has risen since.",
      q: "What is the best action?",
      opts: [
        "Raise the learning rate for epoch 5 to escape the plateau",
        "Remove the eval set — eval loss is noisy on small models",
        "Keep the epoch-2 checkpoint and use fewer epochs or more data in future runs",
        "Turn off completion-only loss so the model sees more tokens",
      ],
      a: 2,
      why: "Diverging train and eval loss is classic [[overfitting|overfitting]]. Keep the best checkpoint; reduce epochs, add diverse data or lower the LR.",
      distract: "A **higher LR** accelerates memorisation. Removing the **eval set** hides the problem. Training on **prompt tokens** dilutes the objective and does not fix overfitting." },

    { id: "f09", domain: "finetuning",
      q: "Which statement about merging a LoRA adapter (merge_and_unload) is correct?",
      opts: [
        "Merging adds a small amount of latency because the adapter still runs as a side branch",
        "Merging is only possible for adapters trained with QLoRA",
        "After merging, different adapters can still be swapped per request on the same weights",
        "Merging folds (α/r)·BA into the base weights, producing a standard dense model with no extra inference cost",
      ],
      a: 3,
      why: "A [[merge-adapter|merged]] adapter disappears into W = W₀ + (α/r)BA, so inference is identical in cost to the base model and works with any serving engine.",
      distract: "After merging there is **no side branch** left. Any LoRA can be merged, not only QLoRA (and QLoRA adapters should be merged into a higher-precision base). Swapping adapters per request requires **keeping them separate** (multi-LoRA serving)." },

    { id: "f10", domain: "finetuning",
      q: "A team has only thumbs-up/thumbs-down feedback on individual responses, with no paired comparisons. Which alignment method fits this data directly?",
      opts: [
        "DPO",
        "KTO",
        "Reward-model training with the Bradley–Terry loss",
        "SimPO",
      ],
      a: 1,
      why: "[[kto|KTO]] learns from unpaired examples labeled desirable or undesirable, which is exactly binary feedback.",
      distract: "**DPO**, **SimPO** and **Bradley–Terry reward models** all require paired comparisons (chosen vs rejected for the same prompt)." },

    { id: "f11", domain: "finetuning",
      q: "What are the three memory-saving ingredients introduced by QLoRA?",
      opts: [
        "INT8 weights, gradient accumulation and FlashAttention",
        "Pruning, distillation and FP8 activations",
        "NF4 4-bit storage of the frozen base, double quantization of the scaling constants, and paged optimizers",
        "ZeRO-3 sharding, CPU offload and activation checkpointing",
      ],
      a: 2,
      why: "QLoRA = [[nf4|NF4]] base weights + [[double-quantization|double quantization]] + [[paged-optimizer|paged optimizers]], with LoRA adapters trained in higher precision.",
      distract: "The other options list real techniques from other contexts (inference quantization, compression, distributed training), not QLoRA's contributions." },

    { id: "f12", domain: "finetuning",
      scenario: "A platform team wants application developers to submit a dataset through a REST API and receive a LoRA-customised model that can be deployed as a NIM, without writing training code.",
      q: "Which NVIDIA component is designed for this?",
      opts: [
        "NeMo Customizer",
        "NeMo Curator",
        "TensorRT-LLM",
        "NeMo Guardrails",
      ],
      a: 0,
      why: "[[nemo-customizer|NeMo Customizer]] is the fine-tuning microservice: create customization jobs (e.g. LoRA or SFT) via API on Kubernetes; outputs are served with NIM.",
      distract: "**NeMo Curator** curates datasets. **TensorRT-LLM** builds optimised inference engines. **NeMo Guardrails** adds programmable safety rails at inference time." },
  ],
}
