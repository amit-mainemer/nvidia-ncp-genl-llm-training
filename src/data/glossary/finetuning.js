// Glossary terms owned by Module 4: Fine-Tuning.
export default [
  { id: "full-fine-tuning", term: "Full fine-tuning", cat: "Fine-tuning & alignment",
    def: "Fine-tuning that updates every weight of the model. With mixed-precision AdamW it needs ~16 bytes per parameter (bf16 weights and grads, fp32 master weights, two fp32 Adam moments) before activations — ~112 GB for a 7B model.",
    exam: "Highest quality ceiling and highest cost; the most prone to catastrophic forgetting. Pick it only when PEFT plateaus and you have the GPUs." },

  { id: "instruction-tuning", term: "Instruction tuning", cat: "Fine-tuning & alignment",
    def: "Supervised fine-tuning of a base model on (instruction, response) pairs, usually rendered through a chat template, so that it follows instructions and takes turns instead of merely continuing text.",
    exam: "It is SFT with instruction data and completion-only loss; it teaches behaviour and format, not new knowledge." },

  { id: "continued-pretraining", term: "Continued pretraining (domain-adaptive pretraining)", cat: "Fine-tuning & alignment",
    def: "Resuming the next-token pretraining objective on a large unlabeled in-domain or new-language corpus (typically 1B–100B+ tokens), usually with a re-warmed, lower learning rate and some general data mixed in.",
    exam: "The answer when there is a genuine domain or language gap and lots of unlabeled text. It must be followed by SFT/alignment, and replay of general data limits forgetting." },

  { id: "base-model", term: "Base model", cat: "Fine-tuning & alignment",
    def: "A checkpoint that has only been pretrained (e.g. `Qwen/Qwen2.5-0.5B`). It continues text and has no built-in turn-taking, chat format or safety tuning.",
    exam: "Start from a base model for heavy SFT, non-chat formats or continued pretraining." },

  { id: "instruct-model", term: "Instruct (chat) model", cat: "Fine-tuning & alignment",
    def: "A checkpoint that has already been instruction-tuned and preference-aligned by its publisher (e.g. `Qwen/Qwen2.5-0.5B-Instruct`), with a specific chat template.",
    exam: "Start from it for light tuning with small data; always reuse its exact chat template, and re-check safety after fine-tuning." },

  { id: "lora-rank", term: "LoRA rank (r)", cat: "Fine-tuning & alignment",
    def: "The inner dimension of the LoRA factors B (d×r) and A (r×k). It sets the capacity of the update and the parameter count r·(d + k) per adapted matrix. Typical values 8–64.",
    exam: "Parameter-count questions: r = 16 on a 4096×4096 matrix adds 131,072 parameters." },

  { id: "lora-alpha", term: "LoRA alpha (α)", cat: "Fine-tuning & alignment",
    def: "A scaling hyperparameter: the LoRA update is multiplied by α/r (or α/√r with rsLoRA). Keeping α proportional to r (α = r or 2r) keeps the effective update size stable when the rank changes.",
    exam: "α scales the update; r sets its capacity; target_modules sets where it is applied." },

  { id: "qlora", term: "QLoRA", cat: "Fine-tuning & alignment",
    def: "LoRA fine-tuning on top of a frozen base model stored in 4-bit NF4, with double quantization of the scaling constants and paged optimizers. Weights are dequantized to bf16/fp16 for each matmul; only the LoRA adapters are trained.",
    exam: "The go-to answer for “fine-tune a 7B–70B model on one GPU”: ~0.5 bytes/param for the base instead of 2 (LoRA) or 16 (full FT)." },

  { id: "nf4", term: "NF4 (4-bit NormalFloat)", cat: "Fine-tuning & alignment",
    def: "A 4-bit data type whose 16 levels are placed at quantiles of a standard normal distribution, applied block-wise (64 weights per absmax scale). Well matched to the roughly normal distribution of pretrained weights.",
    exam: "Set with `bnb_4bit_quant_type=\"nf4\"`; it is a storage format — compute happens in the compute dtype." },

  { id: "double-quantization", term: "Double quantization", cat: "Fine-tuning & alignment",
    def: "Quantizing the per-block quantization constants themselves (fp32 → 8-bit, in blocks of 256), reducing their overhead from ~0.5 to ~0.127 bits per parameter.",
    exam: "Saves ~0.37 bits/param (≈ 3 GB on a 65B model). `bnb_4bit_use_double_quant=True`." },

  { id: "paged-optimizer", term: "Paged optimizer", cat: "Fine-tuning & alignment",
    def: "An optimizer whose state tensors live in CUDA unified memory and are automatically paged to CPU RAM when GPU memory runs short, avoiding OOM on memory spikes (e.g. long sequences).",
    exam: "One of QLoRA's three ingredients; `optim=\"paged_adamw_8bit\"` in Transformers/TRL." },

  { id: "bitsandbytes", term: "bitsandbytes", cat: "Tools & frameworks",
    def: "A CUDA library providing 8-bit and 4-bit (NF4/FP4) quantized linear layers and 8-bit/paged optimizers, integrated into Transformers via `BitsAndBytesConfig`.",
    exam: "The library behind QLoRA in the Hugging Face stack." },

  { id: "dora", term: "DoRA (Weight-Decomposed Low-Rank Adaptation)", cat: "Fine-tuning & alignment",
    def: "A LoRA variant that decomposes each weight into a magnitude vector and a direction matrix; LoRA updates the direction while the magnitude is learned separately. Often narrows the gap to full fine-tuning at the same rank, at some extra training cost.",
    exam: "Enable in peft with `use_dora=True`; still mergeable." },

  { id: "merge-adapter", term: "Merging an adapter", cat: "Fine-tuning & alignment",
    def: "Folding a trained LoRA update into the base weights, W = W₀ + (α/r)·BA (peft `merge_and_unload()`), producing an ordinary dense checkpoint with no adapter overhead at inference.",
    exam: "Merge for single-model deployment; keep adapters separate for multi-LoRA serving. Merge QLoRA adapters into a bf16/fp16 base, not the 4-bit one." },

  { id: "adapters", term: "Adapters (bottleneck adapters)", cat: "Fine-tuning & alignment",
    def: "Small trainable bottleneck modules (down-projection → nonlinearity → up-projection, with a residual) inserted after the attention and/or FFN sublayers of a frozen model. Typically 1–5% extra parameters.",
    exam: "Unlike LoRA, adapters sit in series in the forward path and add some inference latency unless fused." },

  { id: "prompt-tuning", term: "Prompt tuning (soft prompts)", cat: "Fine-tuning & alignment",
    def: "Learning a small set of continuous “virtual token” embeddings prepended to the input embeddings of a frozen model; only those vectors are trained.",
    exam: "Input layer only. Very few parameters; competitive mainly for very large models." },

  { id: "p-tuning", term: "P-tuning", cat: "Fine-tuning & alignment",
    def: "A soft-prompt method in which a small prompt encoder (LSTM or MLP) generates the virtual-token embeddings inserted into the input of a frozen model. After training the generated embeddings can be cached in a prompt table and the encoder discarded.",
    exam: "NeMo's classic PEFT option alongside adapters, IA3 and LoRA; recognise it by the “prompt encoder”." },

  { id: "prefix-tuning", term: "Prefix tuning", cat: "Fine-tuning & alignment",
    def: "Prepending trainable key/value vectors (a prefix) to the attention of every layer of a frozen model. P-tuning v2 applies the same deep-prompt idea.",
    exam: "Every layer, not just the input — that is the difference from prompt tuning." },

  { id: "ia3", term: "IA3", cat: "Fine-tuning & alignment",
    def: "Infused Adapter by Inhibiting and Amplifying Inner Activations: learns vectors that element-wise rescale attention keys, values and FFN intermediate activations. Adds far fewer parameters than LoRA and no new matrices.",
    exam: "The “learned rescaling vectors” PEFT method; smallest trainable footprint." },

  { id: "catastrophic-forgetting", term: "Catastrophic forgetting", cat: "Fine-tuning & alignment",
    def: "Loss of previously learned capabilities (general knowledge, other languages, reasoning, safety refusals) when a model is trained on a narrow new distribution.",
    exam: "Detected only by evaluating outside the task. Mitigate with lower LR, fewer epochs, PEFT, replay of general data and KL-style regularisation." },

  { id: "reward-model", term: "Reward model", cat: "Fine-tuning & alignment",
    def: "A model (usually an LLM with a scalar head) trained on human or AI preference comparisons, typically with the Bradley–Terry loss −log σ(r(x, y_w) − r(x, y_l)), to score how good a response is.",
    exam: "Stage 2 of RLHF. DPO removes the need for one; GRPO can use one or a programmatic verifier." },

  { id: "ppo", term: "PPO (Proximal Policy Optimization)", cat: "Fine-tuning & alignment",
    def: "A policy-gradient RL algorithm with a clipped probability-ratio objective that limits each update. In RLHF it maximises reward − β·KL(π‖π_ref) using a learned value (critic) model for advantages.",
    exam: "Needs policy, reference, reward and value models in memory and on-policy generation — powerful but complex and expensive." },

  { id: "kl-penalty", term: "KL penalty", cat: "Fine-tuning & alignment",
    def: "A term β·KL(π_θ ‖ π_ref) that penalises the trained policy for drifting from a frozen reference model during RL or preference optimisation.",
    exam: "Prevents fluency collapse and limits reward hacking; DPO's β is the same constraint in implicit form." },

  { id: "reference-model", term: "Reference model", cat: "Fine-tuning & alignment",
    def: "A frozen copy of the starting (usually SFT) policy used in RLHF, DPO and KTO to compute the KL constraint or the log-probability ratios. With LoRA it can be the base with the adapter disabled.",
    exam: "DPO, KTO and PPO use one; ORPO and SimPO are reference-free." },

  { id: "dpo", term: "DPO (Direct Preference Optimization)", cat: "Fine-tuning & alignment",
    def: "Preference alignment with a logistic loss on preference pairs: −log σ(β[log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)]). It optimises the RLHF objective without a reward model or sampling.",
    exam: "Needs (prompt, chosen, rejected) data and a reference model; β ≈ 0.1; small LRs (~1e-6–5e-6 full)." },

  { id: "orpo", term: "ORPO (Odds Ratio Preference Optimization)", cat: "Fine-tuning & alignment",
    def: "A single-stage method that adds an odds-ratio penalty favouring chosen over rejected responses to the ordinary SFT loss, with no reference model.",
    exam: "SFT + preference alignment in one pass, reference-free." },

  { id: "kto", term: "KTO (Kahneman–Tversky Optimization)", cat: "Fine-tuning & alignment",
    def: "A preference-alignment loss inspired by prospect theory that learns from unpaired responses labeled desirable or undesirable, relative to a reference model.",
    exam: "The fit for thumbs-up/thumbs-down feedback without paired comparisons." },

  { id: "simpo", term: "SimPO (Simple Preference Optimization)", cat: "Fine-tuning & alignment",
    def: "A reference-free DPO variant whose implicit reward is the length-normalised average log-probability of a response, with a target reward margin γ between chosen and rejected.",
    exam: "Reference-free and counters length bias." },

  { id: "grpo", term: "GRPO (Group Relative Policy Optimization)", cat: "Fine-tuning & alignment",
    def: "An RL algorithm that samples a group of G responses per prompt and uses each response's reward normalised within the group, (r − mean)/std, as its advantage — no value model. Introduced with DeepSeekMath and used for reasoning models.",
    exam: "PPO without the critic; pairs with verifiable rewards (correct answer, passing tests). No signal when all G answers get the same reward." },

  { id: "verifiable-rewards", term: "Verifiable rewards (RLVR)", cat: "Fine-tuning & alignment",
    def: "Rewards computed by a program rather than a learned model — e.g. the final maths answer matches, code passes unit tests, output validates against a schema.",
    exam: "Harder to hack than learned reward models, but weak verifiers are still exploited." },

  { id: "reward-hacking", term: "Reward hacking", cat: "Fine-tuning & alignment",
    def: "A policy achieving high reward by exploiting flaws in the reward model or verifier (length, flattery, format tricks) rather than genuinely improving.",
    exam: "Symptom: training reward rises while held-out or human evaluation does not. Controls: KL penalty, length normalisation, robust verifiers, independent evals." },

  { id: "nemo-rl", term: "NeMo-RL", cat: "NVIDIA stack",
    def: "NVIDIA's open-source post-training library for alignment and reinforcement learning (e.g. SFT, DPO, GRPO, reward models) at scale; the successor to NeMo-Aligner.",
    exam: "The NVIDIA answer for RLHF/DPO/GRPO-style post-training." },

  { id: "nemo-customizer", term: "NeMo Customizer", cat: "NVIDIA stack",
    def: "A NeMo microservice that exposes LLM fine-tuning (e.g. LoRA and full SFT) as a REST API running on Kubernetes; jobs take a dataset and base model config and produce customised models deployable with NIM.",
    exam: "Fine-tuning-as-a-service for platform teams — not data curation (Curator) or inference optimisation (TensorRT-LLM)." },
]
