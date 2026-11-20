/**
 * Module 8 glossary — Model Deployment. Serving engines and APIs, containers and
 * Kubernetes, autoscaling, and the latency/throughput metrics used for capacity planning.
 * Shared terms (inference, kv-cache, nim, triton, tensorrt-llm, lora …) live in core.js.
 */
export default [
  /* ---------------- Serving engines & APIs ---------------- */
  { id: 'vllm', term: 'vLLM', cat: 'Tools & frameworks',
    def: 'An open-source, high-throughput LLM inference engine built around PagedAttention (paged KV-cache memory) and continuous batching, with prefix caching, quantised weights, multi-LoRA, structured outputs and an OpenAI-compatible server (`vllm serve <model>`). Also usable in-process for offline batch inference (`vllm.LLM`).',
    exam: 'The default open-source answer for "serve a Hugging Face model behind an OpenAI-compatible API". Know `--enable-lora`, `--max-model-len`, `--gpu-memory-utilization` and the `/health`, `/v1/models` and `/metrics` endpoints.' },

  { id: 'openai-api', term: 'OpenAI-compatible API', cat: 'Serving & MLOps',
    def: 'The de facto HTTP interface for LLM serving: `POST /v1/chat/completions` (messages with roles), `/v1/completions` (raw prompt), `/v1/embeddings` and `GET /v1/models`, with parameters such as temperature, top_p, max_tokens, stop, n and stream (server-sent events). Implemented by vLLM, NIM, SGLang, TensorRT-LLM\'s trtllm-serve and Triton\'s OpenAI frontend.',
    exam: 'Lets one client codebase target a local server, NIM or a hosted service by changing only `base_url` and `model`.' },

  { id: 'multi-lora', term: 'Multi-LoRA serving', cat: 'Serving & MLOps',
    def: 'Serving many LoRA adapters on one shared base model: the base weights are loaded once, small adapters are loaded on demand, and requests select an adapter by model name — requests for different adapters can be batched together. Supported by vLLM (`--enable-lora --lora-modules`), NIM (`NIM_PEFT_SOURCE`) and TensorRT-LLM.',
    exam: 'Many fine-tunes of the same base → multi-LoRA. One fine-tune that must be as fast as possible → merge the adapter and optimise the merged model.' },

  { id: 'triton-model-repository', term: 'Triton model repository', cat: 'NVIDIA stack',
    def: 'The directory Triton serves from: one sub-directory per model containing a `config.pbtxt` (backend, inputs/outputs, max_batch_size, dynamic_batching, instance_group) and numbered version folders (`1/`, `2/` …) holding the model files.',
    exam: 'Know the layout and that versions are numeric folders. Triton ports: 8000 HTTP, 8001 gRPC, 8002 metrics.' },

  { id: 'dynamic-batching', term: 'Dynamic batching (Triton)', cat: 'Serving & MLOps',
    def: 'A Triton scheduler feature that groups independent incoming requests into one batch on the server, waiting at most `max_queue_delay_microseconds` and preferring `preferred_batch_size` values. Raises GPU utilisation for fixed-shape models such as embedders, rerankers and classifiers.',
    exam: 'Request-level batching for fixed-shape models. For autoregressive LLMs the equivalent is in-flight (continuous) batching inside the TensorRT-LLM or vLLM backend, which adds and removes sequences every decode step.' },

  { id: 'triton-ensemble', term: 'Triton ensemble', cat: 'NVIDIA stack',
    def: 'A Triton model of `platform: "ensemble"` whose config wires the outputs of some models to the inputs of others (e.g. tokenizer → model → detokenizer) so a whole pipeline runs server-side in one request, without client round-trips. Business Logic Scripting (BLS) in the Python backend covers conditional or looping flows.',
    exam: 'Pre/post-processing chained with the model on the server → ensemble; branching logic → BLS.' },

  /* ---------------- Containers & orchestration ---------------- */
  { id: 'docker', term: 'Docker / containers', cat: 'Serving & MLOps',
    def: 'Packaging an application with its user-space dependencies (CUDA libraries, Python, engine, optionally weights) into an immutable image that runs identically on any compatible host. GPU access requires the NVIDIA Container Toolkit and `docker run --gpus …`.',
    exam: 'The host provides the driver; the image provides CUDA user-space. "CUDA driver version is insufficient" means the host driver is too old for the image.' },

  { id: 'nvidia-container-toolkit', term: 'NVIDIA Container Toolkit', cat: 'NVIDIA stack',
    def: 'The runtime hook that exposes NVIDIA GPUs to containers by mounting the host driver libraries and device nodes at container start. Configured for Docker with `nvidia-ctk runtime configure --runtime=docker`; enables `--gpus all` or `--gpus \'"device=0,1"\'`.',
    exam: 'Required for any GPU container. In Kubernetes the GPU Operator installs and configures it on every GPU node.' },

  { id: 'ngc', term: 'NGC catalog', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s hub of GPU-optimised software: containers on the `nvcr.io` registry (PyTorch, NeMo, Triton, NIM), Helm charts, pretrained models and resources. Authenticate with username `$oauthtoken` and an NGC API key.',
    exam: 'Where NIM and Triton images come from; NIM needs `NGC_API_KEY` at runtime to download model profiles.' },

  { id: 'kubernetes', term: 'Kubernetes', cat: 'Serving & MLOps',
    def: 'The container orchestrator that schedules pods onto nodes, restarts failed containers, rolls out new versions and scales replicas. GPUs are exposed as the extended resource `nvidia.com/gpu` by the NVIDIA device plugin.',
    exam: 'GPUs are requested in whole units in `limits`, cannot be overcommitted, and are shared only through MIG or time-slicing.' },

  { id: 'gpu-operator', term: 'NVIDIA GPU Operator', cat: 'NVIDIA stack',
    def: 'A Kubernetes operator that deploys and manages the full GPU node stack as containers: driver, container toolkit, device plugin, GPU Feature Discovery, DCGM exporter and MIG manager. Installed with Helm from the NGC chart repository.',
    exam: 'Pods stuck with "Insufficient nvidia.com/gpu" on nodes that have GPUs → device plugin / GPU Operator not running there.' },

  { id: 'helm', term: 'Helm', cat: 'Tools & frameworks',
    def: 'The Kubernetes package manager: a chart bundles templated manifests (Deployment, Service, PVC, HPA, config) with a values file, installed and upgraded as one versioned release that can be rolled back.',
    exam: 'NVIDIA ships the GPU Operator, NIM Operator, NIM and Triton deployments as Helm charts on NGC.' },

  { id: 'nim-operator', term: 'NVIDIA NIM Operator', cat: 'NVIDIA stack',
    def: 'A Kubernetes operator for NIM microservices. Its custom resources include NIMCache (pre-download model profiles to a PVC) and NIMService (deploy a NIM using that cache, with service exposure, resources and scaling), plus pipeline resources for multi-NIM RAG deployments.',
    exam: 'The Kubernetes-native way to run NIMs at scale with cached models — avoids re-downloading weights on every pod start.' },

  { id: 'gpu-sharing', term: 'GPU sharing: MIG and time-slicing', cat: 'Serving & MLOps',
    def: 'Two ways to run several workloads on one GPU. Multi-Instance GPU (MIG) splits A100/H100-class GPUs into up to 7 hardware-isolated instances with dedicated SMs and memory (e.g. `1g.10gb`). Time-slicing advertises one GPU as N schedulable replicas that share it in time, with no memory or fault isolation.',
    exam: 'Small models with latency SLOs or multiple tenants → MIG. Dev/test or bursty low-priority workloads → time-slicing.' },

  { id: 'health-probes', term: 'Health probes (startup, readiness, liveness)', cat: 'Serving & MLOps',
    def: 'Kubernetes checks on a container: a startup probe allows a slow initialisation (loading weights) before other probes start; a readiness probe gates whether the pod receives traffic; a liveness probe restarts a hung container. LLM servers expose endpoints such as `/health` (vLLM) or `/v1/health/ready` (NIM).',
    exam: 'Without a readiness probe, traffic reaches pods still loading the model and fails. Without a startup probe, a liveness probe can kill a pod that is merely loading.' },

  { id: 'autoscaling', term: 'Autoscaling LLM inference', cat: 'Serving & MLOps',
    def: 'Adding or removing replicas with load, typically with a Kubernetes HorizontalPodAutoscaler (via Prometheus Adapter) or KEDA, driven by inference metrics such as queued requests, running requests per pod, KV-cache utilisation or TTFT percentiles rather than CPU.',
    exam: 'CPU is the wrong signal for GPU-bound servers. Keep warm minimum replicas because scale-up takes minutes (scheduling, image pull, weight load).' },

  /* ---------------- Latency, throughput & benchmarking ---------------- */
  { id: 'ttft', term: 'TTFT (time to first token)', cat: 'Inference optimization',
    def: 'Time from sending a request to receiving the first output token: queueing + prefill of the prompt + the first sampling step (+ network). Grows with prompt length and with load.',
    exam: 'TTFT rising sharply under load while TPOT is stable → queueing → add replicas. Prefix caching and chunked prefill reduce it.' },

  { id: 'tpot', term: 'TPOT / ITL (time per output token / inter-token latency)', cat: 'Inference optimization',
    def: 'The average time between consecutive output tokens after the first: (E2E − TTFT) / (N_out − 1). Determined by the memory-bandwidth-bound decode step and the batch size; 1/TPOT is the per-user token rate.',
    exam: 'Improved by quantisation, speculative decoding, faster HBM and tensor parallelism; worsens as concurrency grows.' },

  { id: 'e2e-latency', term: 'End-to-end latency', cat: 'Inference optimization',
    def: 'Total time from request to last token: E2E = TTFT + (N_out − 1) × TPOT. For non-streaming clients it is the only latency the user experiences.',
    exam: 'Expect the formula in calculations; remember the first token is already counted in TTFT.' },

  { id: 'throughput', term: 'Throughput (tokens/s, requests/s)', cat: 'Inference optimization',
    def: 'Work completed per unit time across all users: output tokens/s (or total tokens/s including prompts) and requests/s, per GPU or per deployment. Rises with batch size and concurrency until the KV cache or compute saturates.',
    exam: 'Always ask "at what latency?" — peak throughput is measured at concurrency where per-user latency may violate the SLO.' },

  { id: 'goodput', term: 'Goodput', cat: 'Inference optimization',
    def: 'Throughput counting only requests that met the latency SLO (e.g. TTFT and TPOT targets). The right quantity for capacity planning and for comparing serving configurations.',
    exam: 'Plan capacity at the highest load that still meets the SLO, not at peak raw throughput.' },

  { id: 'genai-perf', term: 'GenAI-Perf / AIPerf', cat: 'NVIDIA stack',
    def: 'NVIDIA\'s client-side benchmarking tools for generative AI endpoints (OpenAI-compatible, Triton, NIM). They generate synthetic or recorded prompts at a set concurrency or request rate and report TTFT, inter-token latency, E2E latency, token and request throughput, with percentiles. AIPerf is the newer successor to GenAI-Perf.',
    exam: 'The NVIDIA answer to "how do I measure TTFT/ITL and throughput of my LLM endpoint". Use `--streaming` and sweep concurrency.' },

  { id: 'streaming', term: 'Streaming (server-sent events)', cat: 'Serving & MLOps',
    def: 'Returning generated tokens incrementally over an open HTTP response (`stream=True`, SSE chunks) instead of one response at the end. Reduces perceived latency to TTFT; total latency is unchanged.',
    exam: 'Required for client-side TTFT measurement and for chat UX. Load balancers and probes must tolerate long-lived connections; drain them on rollout.' },

  { id: 'offline-inference', term: 'Offline (batch) inference', cat: 'Serving & MLOps',
    def: 'Processing a fixed dataset with no user waiting, optimised for throughput and cost rather than latency — typically an in-process engine (vLLM `LLM`, TensorRT-LLM) running at the largest batch the KV cache allows.',
    exam: 'Evaluation runs, synthetic-data generation and bulk labelling → offline batch, not an autoscaled online endpoint.' },
]
