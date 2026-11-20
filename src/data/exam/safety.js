// Exam-bank questions for Safety, Ethics, and Compliance (official domain 10, 5%).
export default [
  { id: "xs01", domain: "safety",
    scenario: "An enterprise assistant built with NeMo Guardrails must block harmful requests with low latency. Today a self-check input rail calls the 70B main model for every message, adding ~600 ms.",
    q: "Which change best reduces latency while keeping content-safety coverage?",
    opts: [
      "Remove input rails and rely on the main model's alignment",
      "Move all checks into the system prompt of the 70B model",
      "Run the self-check rail only on every tenth message",
      "Replace the self-check with a small dedicated content-safety classifier such as a NemoGuard content-safety NIM",
    ],
    a: 3,
    why: "Small specialised safety classifiers ([[nemoguard|NemoGuard]], [[llama-guard|Llama Guard]]) are faster and more consistent than asking a large model to judge itself, and plug into `config.yml` as a `content_safety` model.",
    distract: "Removing rails or relying on the **system prompt** drops a defence layer. **Sampling** one message in ten leaves 90% unchecked." },

  { id: "xs02", domain: "safety",
    scenario: "Before launching a customer-facing chatbot, a team must demonstrate it resists jailbreaks and prompt injection and must catch regressions in future releases.",
    q: "Which plan best meets both goals?",
    opts: [
      "Manual and automated red teaming (e.g. garak probes), then turning confirmed failures into a regression suite run on every release",
      "Running MMLU and HellaSwag before each release",
      "Asking the model whether it is vulnerable",
      "Relying on user reports after launch",
    ],
    a: 0,
    why: "[[red-teaming|Red teaming]] finds vulnerabilities; capturing them as automated tests keeps them fixed across model, prompt and guardrail changes.",
    distract: "**Knowledge benchmarks** do not measure attack resistance. **Self-assessment** by the model is unreliable. **Post-launch reports** find problems after users are harmed." },

  { id: "xs03", domain: "safety",
    q: "A team wants to know whether its fine-tuned summarisation model produces more toxic summaries for articles about some demographic groups than others. Which approach is most appropriate?",
    opts: [
      "Measure overall ROUGE on a general news benchmark",
      "Compute perplexity on the training set",
      "Score summaries with a toxicity classifier and compare rates across groups, using counterfactual article variants where possible",
      "Check the license of the base model",
    ],
    a: 2,
    why: "A bias audit disaggregates a harm metric ([[toxicity|toxicity]]) by group, with [[counterfactual-evaluation|counterfactual]] variants isolating the attribute's effect.",
    distract: "**ROUGE** and **perplexity** measure quality and fit, not group disparities. **Licensing** is a compliance question, not a bias measurement." },
]
