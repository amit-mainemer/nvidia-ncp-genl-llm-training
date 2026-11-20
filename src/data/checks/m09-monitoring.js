// End-of-lesson comprehension checks for Module 9: Production Monitoring and Reliability. Keyed by lesson id.
export default {
  observability: [
    {
      q: 'Which two latency metrics best describe a streamed LLM response?',
      opts: [
        'CPU time and GPU kernel time spent on each request',
        'Mean and maximum end-to-end latency over all requests',
        'Time to first token and time per output token (inter-token latency)',
        'Network round-trip time and DNS lookup time at the client',
      ],
      a: 2,
      why: 'Streaming splits user-perceived latency into waiting for the first token (queueing + prefill) and the pace of subsequent tokens (decode). They respond to different causes and knobs.',
      distract: 'CPU/GPU time and network timings are diagnostic details, not user-facing LLM latency. Mean and max hide the TTFT/TPOT split and the percentile behaviour that matters.',
    },
    {
      q: 'On which port does Triton Inference Server expose Prometheus metrics by default?',
      opts: [
        '8002',
        '8000',
        '8001',
        '9400',
      ],
      a: 0,
      why: 'Triton uses 8000 for HTTP inference, 8001 for gRPC and **8002** for `/metrics`. Port 9400 is the DCGM exporter\'s default.',
      distract: '**8000** is HTTP inference and **8001** is gRPC. **9400** belongs to dcgm-exporter.',
    },
    {
      q: 'Why should you alert on p95 TTFT or SLO burn rather than on GPU utilisation above 90%?',
      opts: [
        'GPU utilisation cannot be collected or stored in Prometheus at all',
        'p95 TTFT is much cheaper to compute and store than GPU utilisation',
        'GPU utilisation alerts are only supported by DCGM, not Alertmanager',
        'Page on user-facing symptoms; a busy GPU is often a healthy LLM server',
      ],
      a: 3,
      why: 'An efficient inference engine keeps GPUs busy by design. [[slo|SLO]]-based symptom alerts page only when users are affected; resource metrics stay on dashboards for diagnosis.',
      distract: 'GPU utilisation is easily scraped via the DCGM exporter and can be alerted on — it is simply a poor paging signal. Cost of computation is irrelevant.',
    },
    {
      q: 'What is the most responsible default for logging prompts and responses in production?',
      opts: [
        'Log every prompt and response in plain text indefinitely so it can be reused for training',
        'Log metadata for every request; keep text only sampled, PII-redacted, restricted and time-limited',
        'Never log anything at all about requests, so there is no privacy risk to manage',
        'Log only the responses in full, because the prompts legally belong to the user',
      ],
      a: 1,
      why: 'Metadata (ids, version, tokens, latency, finish reason) is needed for every request. Text is sensitive, so it is redacted, sampled, restricted and expired according to policy.',
      distract: 'Unlimited plain-text logs create privacy and compliance risk. Logging nothing makes debugging impossible. Responses can contain user data too, and prompts are needed to debug.',
    },
    {
      q: 'The Population Stability Index between last month\'s and this week\'s prompt-length distributions is 0.35. How should you interpret it?',
      opts: [
        'The two distributions are essentially identical, so nothing to do',
        'The model\'s accuracy on this traffic has dropped by about 35%',
        'A significant shift: investigate and check quality on the affected traffic',
        'The metric is invalid because PSI only applies to embeddings',
      ],
      a: 2,
      why: 'PSI above ~0.25 is conventionally a significant shift. [[data-drift|Drift]] is a trigger to investigate and to check quality on the drifted slice, not proof of a regression.',
      distract: 'PSI near 0 means identical. PSI is not an accuracy measure. It works on any binned scalar feature, such as prompt length.',
    },
  ],

  lifecycle: [
    {
      q: 'Which change can alter an LLM\'s outputs without touching its weights, and therefore must be versioned with the release?',
      opts: [
        'The Grafana dashboard layout',
        'The Prometheus scrape interval',
        'The Kubernetes namespace name',
        'The chat template and system prompt',
      ],
      a: 3,
      why: 'Prompts, templates, sampling settings and quantization all change behaviour. A [[model-registry|registry]] entry should pin them together with the weights.',
      distract: 'Dashboards, scrape intervals and namespaces affect observation and deployment plumbing, not the tokens the model produces.',
    },
    {
      q: 'What distinguishes a canary deployment from a shadow deployment?',
      opts: [
        'A canary runs on CPU nodes, while a shadow always runs on GPUs',
        'Canary users receive the new model\'s answers; shadow answers are only logged',
        'A canary requires a model registry entry, while a shadow does not',
        'A shadow deployment switches all traffic to the new version at once',
      ],
      a: 1,
      why: 'A [[canary-deployment|canary]] serves real users at a small percentage; a [[shadow-deployment|shadow]] mirrors traffic and discards the candidate\'s responses, so there is no user impact.',
      distract: 'Hardware and registries are unrelated to the distinction. Switching all traffic at once describes blue-green.',
    },
    {
      q: 'Your company changed its refund policy yesterday, and the assistant keeps quoting the old policy. What is the best fix?',
      opts: [
        'Update the documents in the RAG knowledge base',
        'Launch a full re-fine-tuning run with the new policy text',
        'Lower the sampling temperature',
        'Roll back to the previous model version',
      ],
      a: 0,
      why: 'Fresh facts belong in retrieval: updating the [[rag|RAG]] corpus takes effect immediately and is cheap. Retraining is for behaviour and distribution changes.',
      distract: 'Fine-tuning for a single fact change is slow and unreliable. Temperature does not add knowledge. The previous version knows the old policy too.',
    },
    {
      q: 'Why is "keep the previous version warm" part of a good rollback plan for LLMs?',
      opts: [
        'Older model versions are always faster to serve than new ones',
        'Kubernetes cannot delete pods that hold GPUs while they are serving',
        'Big engines take minutes to load, so a cold rollback would breach the SLO',
        'The model registry forbids deleting any version that was once in production',
      ],
      a: 2,
      why: 'Loading tens of GB of weights and building/initialising an engine is slow. A warm previous version makes [[rollback|rollback]] a traffic switch measured in seconds.',
      distract: 'Older versions are not necessarily faster. Kubernetes deletes GPU pods normally. Registries keep versions, but a registry entry is not a running server.',
    },
    {
      q: 'In a CI promotion gate, which comparison is correct?',
      opts: [
        'Compare the candidate\'s scores with the published scores of the base model',
        'Compare the candidate\'s training loss with production\'s training loss',
        'Compare the candidate on a new test set and production on the old one',
        'Run the same pinned eval suite on both and block on regressions beyond tolerance',
      ],
      a: 3,
      why: 'Regression benchmarking needs the same tasks, seeds, prompts and judge for both versions, with per-metric tolerances and slice-level checks.',
      distract: 'Published base-model scores and training losses do not measure your task. Using different test sets makes the comparison meaningless.',
    },
  ],
}
