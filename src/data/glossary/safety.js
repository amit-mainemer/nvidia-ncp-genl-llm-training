// Glossary terms owned by Module 10: Safety, Ethics, and Compliance.
export default [
  { id: "nemo-guardrails", term: "NeMo Guardrails", cat: "NVIDIA stack",
    def: "NVIDIA's open-source toolkit (Python package `nemoguardrails`, also offered as a microservice) for adding programmable input, dialog, retrieval, execution and output rails around any LLM. Configured with `config.yml`, `prompts.yml` and Colang flow files.",
    exam: "The NVIDIA answer to “how do you enforce topic, safety and security policies at inference time?”" },

  { id: "colang", term: "Colang", cat: "Safety & governance",
    def: "The modelling language of NeMo Guardrails for dialog flows. In Colang 1.0 you write `define user …` (example utterances for an intent), `define bot …` (responses) and `define flow …` (what happens after an intent). Colang 2.0 is a newer, event-driven version.",
    exam: "Recognise the `define user / define bot / define flow` pattern as a dialog/topical rail." },

  { id: "input-rail", term: "Input rail", cat: "Safety & governance",
    def: "A guardrail applied to the user's message before the main LLM sees it — e.g. jailbreak detection, content-safety classification, PII masking or topic checks. It can block, rewrite or pass the message.",
    exam: "First line of defence against harmful requests and direct attacks." },

  { id: "output-rail", term: "Output rail", cat: "Safety & governance",
    def: "A guardrail applied to the LLM's draft response — content safety, fact-checking against retrieved context, PII leakage or format validation — that can block, replace or trigger regeneration.",
    exam: "The backstop for anything that slipped past input rails, and the place for hallucination checks." },

  { id: "dialog-rail", term: "Dialog rail", cat: "Safety & governance",
    def: "A guardrail that operates on the conversation itself — matching canonical user intents and routing them to predefined bot responses or multi-step flows.",
    exam: "Implemented with Colang flows; used for scripted behaviour and topical control." },

  { id: "retrieval-rail", term: "Retrieval rail", cat: "Safety & governance",
    def: "A guardrail applied to retrieved RAG chunks before they are inserted into the prompt — removing injected instructions, masking PII or enforcing access control.",
    exam: "The control for indirect prompt injection through documents." },

  { id: "topical-rail", term: "Topical rail", cat: "Safety & governance",
    def: "A rail that keeps the assistant within allowed subjects, refusing or redirecting off-topic requests (politics, competitors, medical advice) — via Colang flows or a topic-control classifier.",
    exam: "Off-topic is not an attack; the fit is a topical/dialog rail, not a jailbreak detector." },

  { id: "jailbreak", term: "Jailbreak", cat: "Safety & governance",
    def: "A user prompt crafted to make a model violate its own safety policy — through role-play, hypotheticals, many-shot examples, encodings, low-resource languages or adversarial suffixes.",
    exam: "Jailbreak = bypassing the model's safety training; contrast with prompt injection, which overrides the application's instructions." },

  { id: "prompt-injection", term: "Prompt injection", cat: "Safety & governance",
    def: "An attack in which untrusted text supplies instructions that override the application's intended instructions — directly (typed by the user) or indirectly (hidden in content the model reads). Goals include leaking the system prompt or data and misusing tools.",
    exam: "An application-security problem: defend with privilege separation, rails, constrained tools and output filtering, not with prompt wording alone." },

  { id: "indirect-prompt-injection", term: "Indirect prompt injection", cat: "Safety & governance",
    def: "Prompt injection delivered through third-party content — web pages, emails, PDFs, RAG chunks, tool outputs — rather than the user's own message. The user may be entirely innocent.",
    exam: "The main injection risk for RAG systems and agents; mitigated with retrieval and output rails and least-privilege tools." },

  { id: "llama-guard", term: "Llama Guard", cat: "Safety & governance",
    def: "Meta's family of LLM-based safety classifiers (e.g. Llama Guard 3 1B/8B, Llama Guard 4) that label a user prompt or model response as `safe` or `unsafe` plus hazard category codes, based on a customisable taxonomy aligned with MLCommons hazards.",
    exam: "A plug-in content-safety classifier for input and output rails." },

  { id: "nemoguard", term: "NemoGuard", cat: "NVIDIA stack",
    def: "NVIDIA's family of safety models packaged as NIMs for use with NeMo Guardrails — including content-safety (trained on NVIDIA's Aegis safety data), topic-control and jailbreak-detection models.",
    exam: "Low-latency classifiers that implement content-safety, topical and jailbreak rails." },

  { id: "defence-in-depth", term: "Defence in depth", cat: "Safety & governance",
    def: "Layering independent controls — safety-aligned model, system prompt, input rails, retrieval and tool controls, output rails, monitoring and human review — so that no single failure exposes users to harm.",
    exam: "When asked for “the most robust” design, pick the layered answer over any single control." },

  { id: "data-privacy", term: "Data privacy", cat: "Safety & governance",
    def: "Protecting personal data across the LLM lifecycle: minimising and redacting PII in training data, prompts, outputs and logs; controlling retention and access; respecting consent and regulations such as GDPR.",
    exam: "Redact before logging and before sending data to third parties; never bake private data into shared weights." },

  { id: "toxicity", term: "Toxicity", cat: "Safety & governance",
    def: "Rude, hateful, harassing, threatening or demeaning content. Measured with classifiers and benchmarks such as RealToxicityPrompts; toxicity rates should also be compared across demographic groups.",
    exam: "Toxicity classifiers can themselves be biased (e.g. against dialects) — audit the auditor." },

  { id: "bias", term: "Bias (in AI systems)", cat: "Safety & governance",
    def: "Systematic differences in model behaviour or outcomes across groups that are unjustified for the task, arising from representation, historical, measurement, aggregation, evaluation or deployment issues. Harms are allocational or representational.",
    exam: "Detect with counterfactual evaluation and disaggregated metrics; mitigate across data, training, inference and monitoring." },

  { id: "fairness-metrics", term: "Fairness metrics", cat: "Safety & governance",
    def: "Quantities that compare outcomes across groups: demographic parity (equal positive rates), disparate impact ratio (min/max rate; the four-fifths rule flags < 0.8), equal opportunity (equal TPR), equalized odds (equal TPR and FPR), counterfactual fairness.",
    exam: "Know the definitions and that they can conflict when base rates differ." },

  { id: "counterfactual-evaluation", term: "Counterfactual evaluation", cat: "Safety & governance",
    def: "Testing for bias by changing only a protected attribute (name, pronoun, nationality, dialect) in otherwise identical inputs and measuring how the model's output changes.",
    exam: "The standard way to isolate the effect of a demographic attribute on an LLM's output." },

  { id: "bbq", term: "BBQ (Bias Benchmark for QA)", cat: "Evaluation",
    def: "A question-answering benchmark with ambiguous contexts (correct answer: unknown) and disambiguated contexts, designed to measure how much a model relies on social stereotypes across categories such as age, gender, race and religion.",
    exam: "A model that picks the stereotyped person in ambiguous contexts shows bias." },

  { id: "red-teaming", term: "Red teaming", cat: "Safety & governance",
    def: "Structured adversarial testing in which humans and automated tools try to make a model or system fail — harmful output, data leakage, bias, tool misuse — followed by triage, mitigation and regression testing.",
    exam: "Combine manual (creative, domain-specific) and automated (scale, regression) red teaming." },

  { id: "garak", term: "garak", cat: "Tools & frameworks",
    def: "NVIDIA's open-source LLM vulnerability scanner. It runs probes (attack generators such as jailbreaks, prompt injection, encoding tricks, data leakage) against a model or endpoint and uses detectors to score which attacks succeeded.",
    exam: "The NVIDIA tool for automated red teaming." },

  { id: "model-card", term: "Model card", cat: "Safety & governance",
    def: "A short document published with a model describing its details, intended and out-of-scope uses, training data, evaluation results (ideally disaggregated by group), limitations, ethical considerations and license. NVIDIA's Model Card++ adds bias, explainability, privacy and safety subcards.",
    exam: "Transparency artefact expected for responsible release." },

  { id: "datasheet", term: "Datasheet for datasets", cat: "Safety & governance",
    def: "Documentation for a dataset covering motivation, composition, collection, preprocessing and labeling, intended uses, distribution, maintenance, and consent and privacy handling.",
    exam: "The dataset counterpart of a model card." },

  { id: "eu-ai-act", term: "EU AI Act", cat: "Safety & governance",
    def: "The European Union's risk-based AI regulation: unacceptable-risk uses are prohibited, high-risk uses (e.g. hiring, credit, education) carry risk-management, documentation, logging and human-oversight duties, limited-risk systems have transparency duties, and general-purpose AI models have their own obligations. Applies in phases from 2025.",
    exam: "The tier depends on the use case, not the model: the same LLM can be limited risk in one product and high risk in another." },
]
