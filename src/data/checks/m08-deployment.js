// End-of-lesson comprehension checks for Module 8: Model Deployment. Keyed by lesson id.
export default {
  serving: [
    {
      q: 'You trained a LoRA adapter with QLoRA (4-bit base). How should you produce a standalone merged checkpoint for serving?',
      opts: [
        'Merge the adapter into the 4-bit quantised base model and save the result',
        'Copy the adapter files into the base model folder; engines merge them automatically',
        'Load the base in bf16, apply the adapter, merge_and_unload(), save with the tokenizer',
        'Convert the adapter to ONNX and serve it on its own without the base',
      ],
      a: 2,
      why: 'Merging into full-precision weights avoids quantisation error in the merged result; `merge_and_unload()` folds W + (α/r)·BA into the base and removes the adapter modules. Save the tokenizer for its chat template.',
      distract: 'Merging into 4-bit weights loses accuracy. Simply copying files does not merge anything. An adapter is only a weight delta — it cannot run without its base.',
    },
    {
      q: 'In `vllm serve ./merged --served-model-name my-sft`, what must clients put in the request\'s `model` field?',
      opts: [
        'my-sft',
        './merged',
        'The Hugging Face id of the original base model',
        'Anything — vLLM serves one model and ignores the field',
      ],
      a: 0,
      why: '`--served-model-name` sets the public name listed by `/v1/models` and expected in requests.',
      distract: 'The path is used only when no served name is given. The base id is not registered by this server. vLLM validates the model name and returns an error for unknown names.',
    },
    {
      q: 'What does a NIM container need at runtime to download its model profile from NGC?',
      opts: [
        'A Hugging Face token in HF_TOKEN',
        'A Triton model repository mounted at /models',
        'A pre-built TensorRT-LLM engine passed with --engine-dir',
        'An NGC API key in the NGC_API_KEY environment variable',
      ],
      a: 3,
      why: '[[nim|NIM]] authenticates to [[ngc|NGC]] with `NGC_API_KEY` and caches what it downloads in `/opt/nim/.cache` (mount a volume there).',
      distract: 'HF tokens are for the Hugging Face Hub. NIM does not need a Triton model repository from you. NIM selects or builds the engine profile itself.',
    },
    {
      q: 'In a Triton `config.pbtxt`, what does `dynamic_batching { max_queue_delay_microseconds: 2000 }` do?',
      opts: [
        'Rejects any request that waits in the queue longer than 2 ms',
        'Lets the server wait up to 2 ms to group requests into a batch',
        'Runs two model instances, each with a 2 ms time slice',
        'Streams generated tokens to clients every 2 ms for LLM backends',
      ],
      a: 1,
      why: '[[dynamic-batching|Dynamic batching]] trades a small, bounded queue delay for larger batches and higher GPU utilisation.',
      distract: 'The delay is a batching window, not a timeout. Instance counts are set by `instance_group`. Streaming cadence is not configured here.',
    },
    {
      q: 'You must serve an embedding model, a reranker and an LLM, with tokenisation and post-processing on the server, behind one endpoint. Which option fits best?',
      opts: [
        'Three separate vLLM servers called in sequence by the client',
        'A single NIM, which can load any additional models you mount',
        'Triton with several backends and an ensemble (or BLS) wiring the steps',
        'Multi-LoRA serving with one adapter per pipeline stage',
      ],
      a: 2,
      why: 'Triton hosts mixed backends in one server and [[triton-ensemble|ensembles]]/BLS chain them server-side.',
      distract: 'Client-side chaining adds round-trips and pushes orchestration into the app. A NIM serves its packaged model. LoRA adapters adapt one LLM; they cannot implement an embedder or reranker stage.',
    },
  ],

  'containers-k8s': [
    {
      q: 'Which component makes `docker run --gpus all …` expose the host\'s GPUs inside the container?',
      opts: [
        'The CUDA toolkit libraries installed inside the image',
        'The NVIDIA Kubernetes device plugin on the host',
        'The NVIDIA DCGM daemon running on the host',
        'The NVIDIA Container Toolkit registered with Docker',
      ],
      a: 3,
      why: 'The [[nvidia-container-toolkit|NVIDIA Container Toolkit]] mounts the host driver libraries and GPU device nodes into the container at start.',
      distract: 'CUDA in the image is user-space only and cannot reach the GPU without the toolkit. The device plugin is a Kubernetes component. DCGM is monitoring.',
    },
    {
      q: 'A vLLM pod takes 6 minutes to load weights and keeps being restarted before it finishes. Which probe change fixes this?',
      opts: [
        'Remove the readiness probe so traffic is accepted immediately',
        'Add a startup probe that allows for the load time before liveness checks',
        'Lower the liveness probe period to 1 second for faster detection',
        'Replace the HTTP probes with a TCP socket probe on port 8002',
      ],
      a: 1,
      why: 'A startup probe holds off liveness checks until the app first reports healthy, so long model loads are not mistaken for hangs ([[health-probes|health probes]]).',
      distract: 'Removing readiness sends traffic to a pod that cannot serve. A shorter liveness period kills it faster. Port 8002 is Triton\'s metrics port and the problem is timing, not probe type.',
    },
    {
      q: 'Which resource specification is valid for one GPU per pod on Kubernetes?',
      opts: [
        'resources.limits: nvidia.com/gpu: 1',
        'resources.requests: nvidia.com/gpu: 0.5',
        'resources.limits: gpu: "1000m"',
        'resources.requests: cpu: "gpu"',
      ],
      a: 0,
      why: 'GPUs are the extended resource `nvidia.com/gpu`, requested in whole integers in `limits` (requests, if set, must equal limits).',
      distract: 'Fractions are not allowed — use MIG or time-slicing. `gpu` with millicores is not a valid resource. CPU values are quantities, not resource names.',
    },
    {
      q: 'Several teams each need a small model with a predictable latency SLO on shared H100s. Which sharing method fits?',
      opts: [
        'Time-slicing with 7 replicas per GPU',
        'Running all models in one process with a shared CUDA context',
        'MIG partitions such as 1g.10gb, each assigned to one team',
        'Requesting nvidia.com/gpu: 0 so the scheduler packs pods freely',
      ],
      a: 2,
      why: 'MIG gives hardware-isolated compute and memory per instance, so one tenant cannot OOM or slow another ([[gpu-sharing|GPU sharing]]).',
      distract: 'Time-slicing has no memory or fault isolation, so latency is unpredictable. A shared process mixes tenants with no isolation. A zero GPU request gives the pod no GPU at all.',
    },
    {
      q: 'Which metric is the best primary signal for autoscaling an LLM server?',
      opts: [
        'Requests waiting in the engine queue per replica',
        'Average CPU utilisation of the inference containers',
        'Host memory utilisation on the GPU worker nodes',
        'GPU temperature reported by the DCGM exporter',
      ],
      a: 0,
      why: 'Queued requests indicate that TTFT is about to rise because the replica is saturated — a leading indicator for [[autoscaling|autoscaling]]. KV-cache utilisation is a good companion signal.',
      distract: 'CPU is mostly idle on a GPU-bound server. Host memory says little about inference load. Temperature is a hardware-health metric.',
    },
  ],

  'latency-throughput': [
    {
      q: 'Which phase mostly determines TTFT for a request with a 6,000-token prompt on a lightly loaded server?',
      opts: [
        'Decode of the output tokens',
        'Prefill of the prompt',
        'Detokenisation of the response',
        'TCP connection setup',
      ],
      a: 1,
      why: '[[ttft|TTFT]] = queueing + prefill + first token. At light load queueing is near zero, and prefill cost grows with prompt length.',
      distract: 'Decode happens after the first token. Detokenisation and connection setup are tiny compared with prefilling 6,000 tokens.',
    },
    {
      q: 'Why does output-token throughput rise sharply when concurrency increases from 1 to 32?',
      opts: [
        'Larger batches make each sequence\'s tokens cheaper to sample',
        'Higher concurrency enables tensor parallelism automatically',
        'TTFT decreases as more requests arrive, freeing decode time',
        'One weight read per decode step serves every sequence in the batch',
      ],
      a: 3,
      why: 'Decode is memory-bandwidth-bound: the weight read dominates the step time, so batching amortises it across many sequences and [[throughput|throughput]] multiplies.',
      distract: 'Sampling cost is negligible either way. Parallelism is a deployment setting, not triggered by load. TTFT rises, not falls, with load.',
    },
    {
      q: 'Your SLO is p95 TTFT < 1 s. What should capacity planning be based on?',
      opts: [
        'The maximum throughput measured at any concurrency level',
        'The average TTFT measured at the expected peak load',
        'The highest per-replica concurrency keeping p95 TTFT under 1 s',
        'The single-user latency measured at concurrency 1',
      ],
      a: 2,
      why: 'Plan on [[goodput|goodput]]: the highest load meeting the SLO percentile, then divide expected in-flight requests by it.',
      distract: 'Peak throughput is usually past the SLO. Averages ignore the tail the SLO targets. Concurrency 1 wastes most of the GPU.',
    },
    {
      q: 'Which GenAI-Perf option is required to report TTFT and inter-token latency?',
      opts: [
        '--streaming',
        '--concurrency 1',
        '--request-count 1000',
        '--synthetic-input-tokens-stddev 0',
      ],
      a: 0,
      why: 'Without streaming the client only sees the complete response, so it can measure E2E latency but not when the first token arrived or the gaps between tokens.',
      distract: 'Concurrency, request count and input-length variance shape the load, but do not enable per-token timing.',
    },
    {
      q: 'Peak load is 10 requests/s and each request spends on average 4 s in the system. One replica meets the SLO up to 16 concurrent requests. Before headroom, how many replicas do you need?',
      opts: [
        '1',
        '3',
        '4',
        '40',
      ],
      a: 1,
      why: 'Little\'s law: in-flight L = λ × W = 10 × 4 = 40; 40 / 16 = 2.5 → round up to 3 replicas (then add headroom and N+1).',
      distract: '1 ignores concurrency. 4 adds headroom the question excludes. 40 is the number of in-flight requests, not replicas.',
    },
  ],
}
