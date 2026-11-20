export default {
  id: 'deployment',
  n: 8,
  domain: 'deployment',
  weight: 9,
  title: 'Model Deployment',
  blurb: 'Put your fine-tuned model behind a production API: serve it with vLLM, NIM or Triton, choose merged weights or multi-LoRA, containerise it with the NVIDIA Container Toolkit, run it on Kubernetes with the GPU Operator and the right autoscaling signals, and size the fleet from measured TTFT, TPOT and throughput.',
  hours: '4–5 h',
  lessons: [
    /* ------------------------------------------------------------------ */
    {
      id: 'serving',
      title: 'Lab: serving your model — vLLM, NIM, Triton, merged vs multi-LoRA',
      minutes: 30,
      outcomes: [
        'Serve a fine-tuned model with vLLM behind an OpenAI-compatible API and call it with curl and Python',
        'Decide between merging a LoRA adapter and serving many adapters on one base model',
        'Run an NVIDIA NIM container and send it a chat completion request',
        'Lay out a Triton model repository with config.pbtxt, dynamic batching and ensembles',
        'Choose between vLLM, NIM and Triton + TensorRT-LLM for a given requirement',
      ],
      blocks: [
        { t: 'lede', x: 'Training produced a checkpoint; users need an endpoint. In this lab you serve the model you fine-tuned in module 4 with [[vllm|vLLM]], call it through the [[openai-api|OpenAI-compatible API]], and then compare the NVIDIA options you will meet on the exam: [[nim|NIM]] containers and the [[triton|Triton Inference Server]] with its [[triton-model-repository|model repository]] and [[dynamic-batching|dynamic batching]]. Along the way you decide whether to merge your [[lora|LoRA]] adapter or serve it with [[multi-lora|multi-LoRA]].' },

        { t: 'h2', x: 'The serving landscape' },
        { t: 'table', head: ['Option', 'What it is', 'API', 'Reach for it when'], rows: [
          ['[[vllm|vLLM]]', 'Open-source LLM engine: PagedAttention, continuous batching, prefix caching, LoRA, quantised weights', 'OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`', 'Fast start with any Hugging Face model; open-source stack; dev to production'],
          ['[[nim|NVIDIA NIM]]', 'NVIDIA-built container per model: an optimised engine (TensorRT-LLM, vLLM or SGLang) picked for the detected GPU, plus API server and health/metrics endpoints', 'OpenAI-compatible, port 8000', 'Supported, enterprise-grade, optimised serving of catalog models (NVIDIA AI Enterprise)'],
          ['[[triton|Triton Inference Server]]', 'General multi-framework server: TensorRT, TensorRT-LLM, vLLM, ONNX Runtime, PyTorch, Python backends', 'KServe v2 HTTP/gRPC (+ generate endpoint; OpenAI-compatible frontend available)', 'Many models / frameworks in one server, ensembles (pre/post-processing), maximum control'],
          ['[[tensorrt-llm|TensorRT-LLM]]', 'NVIDIA\'s LLM optimisation library and runtime (covered in module 7)', 'Via Triton\'s backend or its own `trtllm-serve` OpenAI server', 'Lowest latency / highest throughput on NVIDIA GPUs'],
        ]},
        { t: 'p', x: 'NVIDIA has since grouped Triton under the **Dynamo** umbrella (as "Dynamo-Triton"), and NVIDIA Dynamo adds datacenter-scale features such as disaggregated prefill/decode across GPUs. The exam vocabulary — model repository, backends, dynamic batching, ensembles, ports 8000/8001/8002 — is unchanged.' },

        { t: 'lab', title: 'Serve your fine-tuned model with vLLM', x: 'Take the LoRA adapter you trained on Qwen2.5-0.5B-Instruct in module 4, serve it two ways — merged into the base weights, and as a hot-pluggable adapter — and call it with curl and the OpenAI Python client, including streaming.', gpu: 'Free Colab T4 or any 16 GB+ NVIDIA GPU', time: '45 min', steps: [
          'Install `vllm`, `peft` and `openai` (vLLM needs a Linux + CUDA GPU; on Colab use a T4 runtime and `--dtype half`, because the T4 has no bf16).',
          'Merge the adapter into the base model and save a standalone checkpoint with its tokenizer.',
          'Start `vllm serve` on the merged checkpoint and check `/health` and `/v1/models`.',
          'Send a chat request with curl, then with the Python client, then with `stream=True`.',
          'Restart vLLM on the base model with `--enable-lora` and your adapter; call the adapter and the base by model name.',
          'Compare GPU memory in `nvidia-smi` and the answers of both setups.',
        ]},
        { t: 'code', lang: 'python', title: 'merge_lora.py — fold the adapter into the base weights', x: `import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_dir = "./qwen-sft-lora"          # output of the module 4 LoRA lab
out_dir = "./qwen-sft-merged"

# Load the base in bf16/fp16 - NOT 4-bit. If you trained with QLoRA, merge into a
# full-precision base; merging into quantised weights loses accuracy.
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter_dir)
merged = model.merge_and_unload()        # W <- W + (alpha/r) * B @ A, adapter modules removed
merged.save_pretrained(out_dir, safe_serialization=True)

# Save the tokenizer too: it carries the chat template the model was fine-tuned with
# (TRL saves the tokenizer next to the adapter; fall back to base_id if yours did not)
AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out_dir)` },
        { t: 'code', lang: 'bash', title: 'Serve and call the merged model', x: `# --served-model-name is what clients put in "model"; --max-model-len caps context (and KV-cache need)
vllm serve ./qwen-sft-merged \\
  --served-model-name my-sft \\
  --max-model-len 4096 \\
  --gpu-memory-utilization 0.85 \\
  --port 8000
# add --dtype half on a T4 (no bf16), --api-key <key> to require auth

# In another shell
curl -s localhost:8000/health -o /dev/null -w "%{http_code}\\n"   # 200 when ready
curl -s localhost:8000/v1/models | python -m json.tool

curl -s localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
        "model": "my-sft",
        "messages": [{"role": "user", "content": "Explain gradient accumulation in two sentences."}],
        "max_tokens": 128,
        "temperature": 0.2
      }' | python -m json.tool` },
        { t: 'code', lang: 'python', title: 'client.py — normal and streaming calls', x: `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

r = client.chat.completions.create(
    model="my-sft",
    messages=[{"role": "user", "content": "What does a KV cache store?"}],
    max_tokens=128, temperature=0.2,
)
print(r.choices[0].message.content, r.usage)

# Streaming: tokens arrive as server-sent events; the user sees text after TTFT, not after the full answer
stream = client.chat.completions.create(
    model="my-sft",
    messages=[{"role": "user", "content": "List three causes of CUDA OOM."}],
    max_tokens=200, stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)` },

        { t: 'h2', x: 'Merged weights or multi-LoRA?' },
        { t: 'vs', cards: [
          { h: 'Merge the adapter', items: [
            '`merge_and_unload()` produces a normal checkpoint — any engine can serve it (vLLM, NIM, TensorRT-LLM)',
            'Zero adapter overhead at inference: same latency as the base model',
            'Easy to quantise or compile into an optimised engine afterwards',
            'One full model copy per fine-tune — 10 customers = 10 deployments',
          ]},
          { h: 'Serve adapters on a shared base ([[multi-lora|multi-LoRA]])', items: [
            'One base model in GPU memory, many small adapters loaded on demand',
            'Requests choose an adapter by model name; different adapters batch together',
            'Small per-token overhead for the adapter matmuls',
            'Ideal for per-customer / per-task fine-tunes of the same base',
          ]},
        ]},
        { t: 'p', x: '**Do the arithmetic.** Qwen2.5-0.5B has hidden size 896, intermediate size 4,864 and 24 layers. A rank-16 LoRA on all seven linear projections adds r × (d_in + d_out) per matrix — about 0.37M parameters per layer, **~8.8M in total ≈ 18 MB in bf16**. Ten adapters ≈ 180 MB on one shared base (~1 GB of weights), versus ten merged copies ≈ 10 GB. For an 8B base the gap is 16 GB per copy.' },
        { t: 'code', lang: 'bash', title: 'Multi-LoRA with vLLM', x: `vllm serve Qwen/Qwen2.5-0.5B-Instruct \\
  --enable-lora \\
  --lora-modules my-sft=./qwen-sft-lora support-bot=./support-lora \\
  --max-lora-rank 16 \\
  --max-loras 4 \\
  --port 8000
# --max-lora-rank must be >= the largest adapter rank; --max-loras = adapters active in one batch

# The adapter name IS the model name; the base model stays callable by its own id
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \\
  -d '{"model": "support-bot", "messages": [{"role":"user","content":"Reset my VPN token"}], "max_tokens": 64}'` },
        { t: 'callout', kind: 'trap', title: 'Multi-LoRA gotchas', x: [
          'An adapter only works on the **exact base model** it was trained on (same architecture, same weights revision). A Qwen adapter cannot be loaded onto a Llama base.',
          'Adapters with a rank above `--max-lora-rank` fail to load.',
          'If you fine-tuned with an extended vocabulary or changed embeddings, a plain adapter on the original base is not enough — merge, or serve the modified base.',
        ]},

        { t: 'h2', x: 'NVIDIA NIM: a model in one container' },
        { t: 'p', x: 'A [[nim|NIM]] packages a model with an inference engine and an API server. On start-up it inspects the GPU, picks a matching optimised profile (for example a [[tensorrt-llm|TensorRT-LLM]] engine for H100 at FP8, or a vLLM profile where no engine exists), downloads it from [[ngc|NGC]] into a cache directory and serves an OpenAI-compatible API on port 8000.' },
        { t: 'code', lang: 'bash', title: 'Run an LLM NIM', x: `export NGC_API_KEY=<your NGC key>                 # from ngc.nvidia.com -> Setup -> API key
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

export LOCAL_NIM_CACHE=~/.cache/nim               # persist downloaded weights/engines between runs
mkdir -p "$LOCAL_NIM_CACHE"

docker run -it --rm --gpus all --shm-size=16GB \\
  -e NGC_API_KEY \\
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \\
  -u $(id -u) \\
  -p 8000:8000 \\
  nvcr.io/nim/meta/llama-3.1-8b-instruct:latest    # pin a specific tag in production

# readiness, then a request
curl -s localhost:8000/v1/health/ready
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \\
  -d '{"model": "meta/llama-3.1-8b-instruct",
       "messages": [{"role": "user", "content": "One sentence on NVLink."}], "max_tokens": 64}'` },
        { t: 'ul', items: [
          '**LoRA with NIM**: mount a directory of adapters and point `NIM_PEFT_SOURCE` at it; each adapter becomes a model name, served on the shared base (NeMo- and Hugging Face-format adapters are supported for compatible models).',
          '**Profiles**: `list-model-profiles` inside the container shows the available engines; `NIM_MODEL_PROFILE` pins one.',
          '**Your own fine-tune**: serve it as a LoRA on the matching NIM base, or — for supported architectures — point a NIM at your merged checkpoint; check the NIM docs for the current model-loading options.',
          'The cache volume matters: without it every container start re-downloads gigabytes.',
        ]},

        { t: 'h2', x: 'Triton Inference Server' },
        { t: 'p', x: 'Triton serves models from a [[triton-model-repository|model repository]]: one directory per model, numbered version sub-directories, and a `config.pbtxt` that names the backend, inputs/outputs, batching and instance placement. It listens on **8000 (HTTP)**, **8001 (gRPC)** and **8002 (Prometheus metrics)**.' },
        { t: 'code', lang: 'text', title: 'Model repository layout', x: `model_repository/
├── tokenizer/               # Python backend: text -> token ids
│   ├── config.pbtxt
│   └── 1/model.py
├── reranker/                # ONNX cross-encoder with dynamic batching
│   ├── config.pbtxt
│   └── 1/model.onnx
├── llm_vllm/                # vLLM backend: engine args live in model.json
│   ├── config.pbtxt
│   └── 1/model.json
└── rag_pipeline/            # ensemble: wires tokenizer -> reranker, no code
    ├── config.pbtxt
    └── 1/                   # empty version directory is still required` },
        { t: 'code', lang: 'protobuf', title: 'reranker/config.pbtxt — dynamic batching and two GPU instances', x: `name: "reranker"
backend: "onnxruntime"
max_batch_size: 64                       # >0 enables batching; first dim of inputs is the batch
input  [ { name: "input_ids",      data_type: TYPE_INT64, dims: [ -1 ] },
         { name: "attention_mask", data_type: TYPE_INT64, dims: [ -1 ] } ]
output [ { name: "logits",         data_type: TYPE_FP32,  dims: [ 1 ] } ]

dynamic_batching {
  preferred_batch_size: [ 16, 32 ]
  max_queue_delay_microseconds: 2000     # wait up to 2 ms to form a bigger batch
}
instance_group [ { count: 2, kind: KIND_GPU, gpus: [ 0 ] } ]   # two copies on GPU 0 overlap work` },
        { t: 'code', lang: 'bash', title: 'Start Triton and query it', x: `docker run --rm --gpus all --shm-size=8g \\
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
  -v $PWD/model_repository:/models \\
  nvcr.io/nvidia/tritonserver:<yy.mm>-vllm-python-py3 \\
  tritonserver --model-repository=/models

curl -s localhost:8000/v2/health/ready -o /dev/null -w "%{http_code}\\n"
curl -s localhost:8000/v2/models/llm_vllm/generate \\
  -d '{"text_input": "What is Triton?", "parameters": {"max_tokens": 64, "stream": false}}'
curl -s localhost:8002/metrics | grep nv_inference_request_success` },
        { t: 'table', head: ['Triton concept', 'What it gives you'], rows: [
          ['Backends', 'TensorRT, TensorRT-LLM, vLLM, ONNX Runtime, PyTorch (LibTorch), Python — mixed in one server'],
          ['[[dynamic-batching|Dynamic batching]]', 'Server-side grouping of independent requests into one batch, bounded by a queue delay — for fixed-shape models such as embedders, rerankers, classifiers'],
          ['In-flight (continuous) batching', 'Inside the TensorRT-LLM and vLLM backends: sequences join and leave the running batch every decode step — the LLM equivalent'],
          ['Instance groups', 'Several copies of a model per GPU or across GPUs to overlap execution'],
          ['[[triton-ensemble|Ensembles]] / BLS', 'Chain models (tokenise → model → detokenise) in config; BLS = Python business-logic scripting for conditional flows'],
          ['Model control', 'Load/unload/poll models from the repository without restarting the server'],
          ['Metrics (:8002)', 'Request counts, queue and compute latency, GPU utilisation and memory in Prometheus format'],
        ]},

        { t: 'h2', x: 'Choosing a serving stack' },
        { t: 'table', head: ['Requirement', 'Good default'], rows: [
          ['Quickest path from an HF checkpoint to an OpenAI API', 'vLLM'],
          ['Enterprise support, NVIDIA-optimised engine selected automatically, catalog model', 'NIM'],
          ['Many per-customer fine-tunes of one base', 'Multi-LoRA on vLLM or NIM'],
          ['Mixed model types (LLM + embedder + reranker + classic CV) with pre/post-processing', 'Triton with ensembles / BLS'],
          ['Absolute lowest latency at fixed shapes on NVIDIA GPUs', 'TensorRT-LLM engine (via Triton, trtllm-serve or a NIM profile)'],
          ['Huge offline batch jobs', 'In-process engine (`vllm.LLM`) — no HTTP server needed (lesson 3)'],
        ]},
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Know NIM\'s run recipe: `docker run --gpus all -e NGC_API_KEY -v <cache>:/opt/nim/.cache -p 8000:8000 nvcr.io/nim/<org>/<model>:<tag>`, then OpenAI-style `/v1/chat/completions`.',
          'Know Triton\'s ports (8000 HTTP, 8001 gRPC, 8002 metrics), the repository layout (`<model>/config.pbtxt`, `<model>/<version>/…`) and what `dynamic_batching` and `instance_group` do.',
          'Many fine-tunes of one base → multi-LoRA; one fine-tune that must be as fast as possible → merge, then optimise.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          '`vllm serve <path>` gives any HF checkpoint an OpenAI-compatible API; check `/health` and `/v1/models` before sending traffic.',
          'Merge adapters (into a full-precision base) for a single, maximally fast model; serve multi-LoRA when many fine-tunes share a base.',
          'NIM = optimised engine + API in one container, configured with NGC_API_KEY and a cache volume, on port 8000.',
          'Triton = model repository + config.pbtxt + backends; dynamic batching for fixed-shape models, in-flight batching inside LLM backends, ensembles for pipelines.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'containers-k8s',
      title: 'Containers, NGC and Kubernetes: scaling LLM inference',
      minutes: 24,
      outcomes: [
        'Run GPU containers with the NVIDIA Container Toolkit and pull NGC images',
        'Write a Dockerfile for an LLM inference service and decide how to ship the weights',
        'Deploy an inference server on Kubernetes with the GPU Operator, GPU resource limits, probes and a model-cache PVC',
        'Share GPUs among small models with MIG or time-slicing',
        'Autoscale LLM replicas on queue depth or KV-cache utilisation instead of CPU',
      ],
      blocks: [
        { t: 'lede', x: 'A model server on one machine is a demo; production needs reproducible [[docker|containers]], an orchestrator that knows what a GPU is, and scaling rules that understand LLM load. This lesson goes from `docker run --gpus` to a [[kubernetes|Kubernetes]] Deployment with the [[gpu-operator|GPU Operator]], [[health-probes|health probes]], a model cache and [[autoscaling|autoscaling]] on the metrics that actually predict latency.' },

        { t: 'h2', x: 'GPUs inside containers' },
        { t: 'p', x: 'The **NVIDIA driver** lives on the host; the **CUDA user-space libraries** (cuBLAS, cuDNN, [[nccl|NCCL]], PyTorch) live in the image. The [[nvidia-container-toolkit|NVIDIA Container Toolkit]] connects them: at container start it mounts the driver libraries and GPU device nodes into the container. That is why one image runs on any host whose driver is new enough for the image\'s CUDA version.' },
        { t: 'code', lang: 'bash', title: 'Enable and test GPU containers', x: `# After installing the nvidia-container-toolkit package on the host:
sudo nvidia-ctk runtime configure --runtime=docker   # registers the NVIDIA runtime with Docker
sudo systemctl restart docker

docker run --rm --gpus all ubuntu nvidia-smi          # driver + GPUs visible inside the container
docker run --rm --gpus '"device=0,1"' ubuntu nvidia-smi -L   # only GPUs 0 and 1

# Multi-process inference/training needs shared memory: raise /dev/shm or share the host IPC namespace
docker run --rm --gpus all --ipc=host nvcr.io/nvidia/pytorch:<yy.mm>-py3 \\
  python -c "import torch; print(torch.cuda.get_device_name(0))"` },
        { t: 'callout', kind: 'trap', title: '"CUDA driver version is insufficient"', x: 'The image\'s CUDA version must be supported by the **host driver** — the toolkit does not install drivers into containers. Upgrade the host driver (or pick an older image tag); rebuilding the image with more CUDA libraries does not help.' },

        { t: 'h2', x: 'NGC: NVIDIA\'s container registry' },
        { t: 'specs', rows: [
          ['Registry', '`nvcr.io` — log in with username `$oauthtoken` and your NGC API key as the password'],
          ['Frameworks', '`nvcr.io/nvidia/pytorch:<yy.mm>-py3`, `nvcr.io/nvidia/nemo:<tag>` — monthly releases, tested CUDA/cuDNN/NCCL stacks'],
          ['Serving', '`nvcr.io/nvidia/tritonserver:<yy.mm>-py3` and variants (`-trtllm-python-py3`, `-vllm-python-py3`, `-py3-sdk` with GenAI-Perf)'],
          ['NIM', '`nvcr.io/nim/<publisher>/<model>:<tag>` — one image per model family'],
          ['Also on NGC', 'Helm charts (`helm.ngc.nvidia.com`), pretrained models, Jupyter-ready containers; security-scanned with published release notes'],
        ]},

        { t: 'h2', x: 'A Dockerfile for an inference service' },
        { t: 'code', lang: 'dockerfile', title: 'Dockerfile — vLLM serving your merged model', x: `# Start from an engine image instead of installing CUDA + vLLM yourself; pin an exact tag.
FROM vllm/vllm-openai:v0.11.0

# Option A (shown): bake small weights into the image -> immutable, versioned artifact.
# For multi-GB models prefer Option B: mount a volume / PVC and keep the image small.
COPY qwen-sft-merged /models/my-sft

ENV HF_HUB_OFFLINE=1
EXPOSE 8000

# The base image's ENTRYPOINT is vLLM's OpenAI server; CMD supplies its arguments
CMD ["--model", "/models/my-sft", "--served-model-name", "my-sft", \\
     "--max-model-len", "4096", "--port", "8000"]` },
        { t: 'table', head: ['Ship weights…', 'Pros', 'Cons'], rows: [
          ['Baked into the image', 'Immutable, one artifact to version and roll back; no runtime download', 'Huge images (8B ≈ 16 GB), slow pulls on every new node'],
          ['On a PVC / shared filesystem', 'Fast restarts, one copy shared by replicas (ReadWriteMany), small images', 'Must manage cache population and versions separately'],
          ['Downloaded at start (HF Hub / NGC)', 'Simplest to write', 'Slow cold starts, external dependency, rate limits — avoid in production'],
        ]},

        { t: 'h2', x: 'Kubernetes and the GPU Operator' },
        { t: 'p', x: 'Kubernetes schedules pods onto nodes, but out of the box it knows nothing about GPUs. The [[gpu-operator|NVIDIA GPU Operator]] installs, as containers managed by Kubernetes, everything a GPU node needs: the driver, the container toolkit, the **device plugin** (advertises `nvidia.com/gpu` as a schedulable resource), GPU Feature Discovery (node labels such as GPU product and memory), the [[dcgm|DCGM]] exporter (GPU metrics for [[prometheus|Prometheus]]) and the MIG manager.' },
        { t: 'code', lang: 'bash', title: 'Install the GPU Operator with Helm', x: `helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
helm install --wait --generate-name -n gpu-operator --create-namespace nvidia/gpu-operator

kubectl get pods -n gpu-operator                       # driver, toolkit, device-plugin, dcgm-exporter ...
kubectl describe node <gpu-node> | grep nvidia.com/gpu # Capacity / Allocatable: nvidia.com/gpu: 8` },
        { t: 'code', lang: 'yaml', title: 'deployment.yaml — vLLM with GPU limit, model cache and probes', x: `apiVersion: apps/v1
kind: Deployment
metadata: { name: my-sft, labels: { app: my-sft } }
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }   # never drop below current capacity
  selector: { matchLabels: { app: my-sft } }
  template:
    metadata: { labels: { app: my-sft } }
    spec:
      terminationGracePeriodSeconds: 120                 # let in-flight streams finish
      containers:
      - name: vllm
        image: registry.example.com/my-sft:1.4.0
        args: ["--model", "/models/my-sft", "--served-model-name", "my-sft", "--port", "8000"]
        ports: [{ containerPort: 8000, name: http }]
        resources:
          limits: { nvidia.com/gpu: 1, memory: 24Gi }      # GPUs: whole units, set in limits
          requests: { cpu: "4", memory: 16Gi }
        volumeMounts:
        - { name: models, mountPath: /models }
        - { name: shm, mountPath: /dev/shm }
        startupProbe:                                      # model load can take minutes
          httpGet: { path: /health, port: http }
          periodSeconds: 10
          failureThreshold: 60                             # up to 10 min before liveness kicks in
        readinessProbe:                                    # only route traffic when the model is loaded
          httpGet: { path: /health, port: http }
          periodSeconds: 5
        livenessProbe:                                     # restart a hung server
          httpGet: { path: /health, port: http }
          periodSeconds: 15
          failureThreshold: 4
      volumes:
      - name: models
        persistentVolumeClaim: { claimName: model-cache }  # pre-populated weights, shared by replicas
      - name: shm
        emptyDir: { medium: Memory, sizeLimit: 8Gi }
---
apiVersion: v1
kind: Service
metadata: { name: my-sft }
spec:
  selector: { app: my-sft }
  ports: [{ port: 80, targetPort: http }]` },
        { t: 'callout', kind: 'warn', title: 'GPU resources are not like CPU', x: [
          '`nvidia.com/gpu` is an extended resource: whole integers only, no overcommit, and it is specified in `limits` (a request, if given, must equal the limit). `nvidia.com/gpu: 0.5` is invalid — use MIG or time-slicing to share.',
          'A rolling update with `maxSurge: 1` needs one **spare GPU** to start the new pod. On a full cluster, the surge pod stays Pending — plan headroom or use `maxUnavailable: 1` and accept reduced capacity.',
        ]},

        { t: 'h2', x: 'Sharing a GPU between small models' },
        { t: 'p', x: 'A 0.5B model needs ~1 GB of weights; giving it an 80 GB H100 wastes most of the card. Two [[gpu-sharing|GPU-sharing]] mechanisms are configured through the GPU Operator:' },
        { t: 'table', head: ['', 'MIG (Multi-Instance GPU)', 'Time-slicing'], rows: [
          ['How', 'Hardware partitions with dedicated SMs, memory and cache (A100, H100, H200, B200…)', 'Device plugin advertises N "replicas" of one GPU; processes take turns'],
          ['Isolation', 'Strong: memory and fault isolation, predictable latency', 'None: shared memory, one process can OOM the others, noisy neighbours'],
          ['Granularity', 'Fixed profiles, e.g. up to 7 × `1g.10gb` on an H100 80 GB', 'Any replica count'],
          ['K8s resource', '`nvidia.com/mig-1g.10gb: 1` (mixed strategy) or `nvidia.com/gpu` (single strategy)', '`nvidia.com/gpu: 1` (a slice)'],
          ['Use for', 'Multi-tenant production with latency SLOs', 'Dev/test, bursty low-traffic models'],
        ]},
        { t: 'code', lang: 'yaml', title: 'Time-slicing config for the GPU Operator device plugin', x: `apiVersion: v1
kind: ConfigMap
metadata: { name: time-slicing-config, namespace: gpu-operator }
data:
  any: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 4            # each physical GPU is advertised as 4 schedulable GPUs
# then point the ClusterPolicy at it:
# kubectl patch clusterpolicies.nvidia.com/cluster-policy --type merge \\
#   -p '{"spec":{"devicePlugin":{"config":{"name":"time-slicing-config","default":"any"}}}}'` },

        { t: 'h2', x: 'Helm and the NIM Operator' },
        { t: 'p', x: '[[helm|Helm]] packages the Deployment, Service, PVC, autoscaler and config as one versioned, parameterised chart — NVIDIA publishes charts for the GPU Operator, NIMs and [[triton|Triton]] on NGC. For NIMs specifically, the [[nim-operator|NIM Operator]] adds Kubernetes custom resources: a **NIMCache** downloads model profiles once to a PVC, and a **NIMService** deploys the [[nim|NIM]] pointing at that cache, with service, scaling and probes configured for you.' },
        { t: 'code', lang: 'yaml', title: 'NIM Operator resources (abridged — see the NIM Operator docs for all fields)', x: `apiVersion: apps.nvidia.com/v1alpha1
kind: NIMCache
metadata: { name: llama-8b-cache }
spec:
  source:
    ngc:
      modelPuller: nvcr.io/nim/meta/llama-3.1-8b-instruct:<tag>
      pullSecret: ngc-secret
      authSecret: ngc-api-secret
  storage:
    pvc: { create: true, size: 50Gi, volumeAccessMode: ReadWriteMany }
---
apiVersion: apps.nvidia.com/v1alpha1
kind: NIMService
metadata: { name: llama-8b }
spec:
  image: { repository: nvcr.io/nim/meta/llama-3.1-8b-instruct, tag: <tag>, pullSecrets: [ngc-secret] }
  authSecret: ngc-api-secret
  storage: { nimCache: { name: llama-8b-cache } }
  resources: { limits: { nvidia.com/gpu: 1 } }
  expose: { service: { type: ClusterIP, port: 8000 } }` },

        { t: 'h2', x: 'Autoscaling on the right signal' },
        { t: 'p', x: 'The default Horizontal Pod Autoscaler scales on CPU — useless here: the server\'s CPU is nearly idle while the GPU is saturated. **GPU utilisation** is not much better: a decode-heavy server shows high utilisation at both light and heavy load. Scale on signals that track user-visible latency:' },
        { t: 'ul', items: [
          '**Queue depth** — requests waiting to be scheduled (vLLM exposes `vllm:num_requests_waiting`; NIM and Triton expose equivalents). Waiting requests mean TTFT is about to rise.',
          '**KV-cache utilisation** — near 100% means the engine starts preempting or queuing sequences (vLLM\'s KV-cache usage gauge; the exact metric name varies by version).',
          '**Running requests per replica / concurrency** — scale to keep each replica near the concurrency you benchmarked as SLO-safe.',
          '**Latency percentiles** (p95 TTFT) — the most direct, but they react late, after users already feel it.',
        ]},
        { t: 'code', lang: 'yaml', title: 'HPA on queue depth (metric exposed via Prometheus Adapter)', x: `apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: my-sft }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: my-sft }
  minReplicas: 2                       # keep warm capacity: a new replica needs minutes to load
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric: { name: vllm_num_requests_waiting }   # Prometheus Adapter maps vllm:num_requests_waiting
      target: { type: AverageValue, averageValue: "5" }
  behavior:
    scaleDown: { stabilizationWindowSeconds: 600 }  # do not flap: scale down slowly` },
        { t: 'callout', kind: 'tip', title: 'Make scale-up fast', x: 'A new replica must be scheduled on a node with a free GPU, pull its image and load weights before it is Ready — often several minutes, longer if a new node has to be provisioned. Keep `minReplicas` above zero for latency-sensitive services, pre-pull images, load weights from a local PVC cache rather than the internet, and scale before saturation (queue depth), not after (p99 latency). KEDA is a common alternative to the Prometheus Adapter and also supports scale-to-zero for batch or dev endpoints.' },
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          '`--gpus all` + NVIDIA Container Toolkit for Docker; GPU Operator + device plugin + `nvidia.com/gpu` limits for Kubernetes.',
          'Autoscaling LLM servers → custom metrics (queue length, KV-cache usage, concurrency, TTFT), not CPU.',
          'Small models on big GPUs → MIG (isolation, SLOs) or time-slicing (no isolation). Readiness probe = do not route traffic until the model is loaded; startup probe = tolerate slow loads.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'Driver on the host, CUDA in the image, NVIDIA Container Toolkit in between; pull tested stacks from NGC (`nvcr.io`).',
          'Ship large weights on a PVC cache rather than in the image or via runtime downloads.',
          'GPU Operator makes GPUs schedulable; request whole `nvidia.com/gpu` units, or MIG profiles / time-slices to share.',
          'Use startup, readiness and liveness probes, graceful termination and surge capacity for zero-downtime rollouts.',
          'Autoscale on queue depth or KV-cache utilisation with warm minimum replicas; Helm and the NIM Operator package it all.',
        ]},
      ],
    },

    /* ------------------------------------------------------------------ */
    {
      id: 'latency-throughput',
      title: 'Latency, throughput and capacity planning (TTFT, TPOT, genai-perf)',
      minutes: 22,
      outcomes: [
        'Define TTFT, TPOT/ITL, end-to-end latency, throughput and goodput, and relate them with a formula',
        'Explain why throughput rises and per-user latency degrades as concurrency increases',
        'Benchmark an OpenAI-compatible endpoint with GenAI-Perf or AIPerf at several concurrency levels',
        'Size a GPU fleet from an SLO and a traffic forecast using Little\'s law',
        'Choose streaming online serving or offline batch inference for a workload',
      ],
      blocks: [
        { t: 'lede', x: '"How fast is it?" has at least four different answers for an LLM, and they pull against each other. This lesson defines the metrics — [[ttft|TTFT]], [[tpot|TPOT]], [[e2e-latency|end-to-end latency]], [[throughput|throughput]] and [[goodput|goodput]] — shows the trade-off curve you get as concurrency grows, measures it with [[genai-perf|GenAI-Perf]], and turns the measurement into a GPU count for a real SLO.' },

        { t: 'h2', x: 'Anatomy of one request' },
        { t: 'flow', steps: [
          { b: 'Queue', s: 'Waits for a scheduler slot and KV-cache space' },
          { b: 'Prefill', s: 'Whole prompt processed in parallel — compute-bound; fills the KV cache' },
          { b: 'First token', s: 'TTFT = queue + prefill + first sampling step (+ network)' },
          { b: 'Decode', s: 'One token per step for this sequence — memory-bandwidth-bound' },
          { b: 'Done', s: 'EOS, stop sequence or max_tokens; E2E latency measured here' },
        ]},
        { t: 'specs', rows: [
          ['TTFT (time to first token)', 'Request sent → first token received. What a chat user perceives as "responsiveness". Grows with prompt length and with queueing.'],
          ['TPOT / ITL (time per output token / inter-token latency)', 'Average gap between output tokens after the first: (E2E − TTFT) / (output tokens − 1). 1 / TPOT = tokens/s seen by one user.'],
          ['E2E latency', 'Request → last token. E2E = TTFT + (N_out − 1) × TPOT.'],
          ['Throughput', 'System-wide output tokens/s (sometimes total tokens/s incl. prompt) and requests/s, across all concurrent users.'],
          ['Goodput', 'Throughput counting only requests that met the SLO (e.g. TTFT < 500 ms and TPOT < 40 ms). The number that matters for capacity.'],
        ]},
        { t: 'p', x: '**Worked numbers.** TTFT = 250 ms, TPOT = 20 ms, 300 output tokens → E2E = 0.25 + 299 × 0.020 ≈ 6.2 s. The user sees text after a quarter of a second and reads at 50 tokens/s — comfortably faster than reading speed (~5–8 tokens/s). Without streaming, the same request "feels" like 6.2 s.' },

        { t: 'h2', x: 'What drives each metric' },
        { t: 'table', head: ['Metric', 'Dominated by', 'Levers'], rows: [
          ['TTFT', 'Queueing + prefill compute (∝ prompt length)', 'More replicas, prefix caching for shared system prompts, chunked prefill, shorter prompts, FP8, tensor parallelism'],
          ['TPOT', 'Reading all weights + KV cache from HBM each step; batch size', 'Quantisation (fewer bytes), speculative decoding, smaller batches, faster HBM, tensor parallelism'],
          ['Throughput', 'How many sequences share each weight read (batch size), KV-cache capacity', 'Larger batches / more concurrency, PagedAttention, KV-cache quantisation, more GPU memory'],
        ]},
        { t: 'p', x: '**The bandwidth floor.** Decoding one token reads every weight once. An 8B model in bf16 is 16 GB; an H100 SXM delivers ~3.35 TB/s of [[hbm|HBM]] bandwidth, so one decode step takes at least 16 / 3,350 ≈ 4.8 ms — about 200 tokens/s for a single stream, no matter how many FLOPs the GPU has. Batch 32 sequences and that same weight read serves 32 tokens: throughput jumps almost 32×, while each step gets only slightly slower until attention over the growing [[kv-cache|KV cache]] and compute start to dominate. That is the whole economics of LLM serving.' },

        { t: 'h2', x: 'The latency–throughput curve' },
        { t: 'p', x: 'Sweep concurrency (simultaneous in-flight requests) and plot throughput against per-user latency. A typical shape, for an 8B model on one H100 with 512-token prompts and 256-token outputs (illustrative numbers — measure your own):' },
        { t: 'table', head: ['Concurrency', 'Output tok/s (system)', 'TPOT (ms)', 'Per-user tok/s', 'TTFT p95 (ms)'], rows: [
          ['1', '190', '5.3', '190', '35'],
          ['8', '1,300', '6.2', '160', '60'],
          ['32', '3,900', '8.2', '122', '140'],
          ['64', '5,600', '11.4', '88', '310'],
          ['128', '6,400', '20', '50', '1,900 (queueing)'],
        ]},
        { t: 'ul', items: [
          '**Left side**: the GPU is under-used; adding users is almost free — throughput scales nearly linearly.',
          '**Knee**: throughput gains flatten while TPOT and TTFT climb. The best operating point is usually just before the knee, as far right as the SLO allows.',
          '**Right side**: the KV cache is full; new requests wait in the queue. Throughput barely grows, TTFT explodes. Running here is a capacity problem, not a tuning problem.',
        ]},
        { t: 'callout', kind: 'trap', title: 'Reading benchmark claims', x: [
          'Peak throughput is always reported at high concurrency — where per-user latency may be unacceptable. Ask "at what TPOT and TTFT?".',
          'Report p95/p99, not averages: averages hide the queueing tail users complain about.',
          'Input/output lengths change everything. Long prompts stress prefill (TTFT); long outputs stress decode (TPOT). Benchmark with your real length distribution.',
          'Per-GPU and per-user throughput are different numbers; do not divide one by the other without stating concurrency.',
        ]},

        { t: 'h2', x: 'Measuring it: GenAI-Perf and AIPerf' },
        { t: 'p', x: '[[genai-perf|GenAI-Perf]] is NVIDIA\'s client-side benchmark for generative endpoints (shipped in the [[triton|Triton]] SDK container and on PyPI); **AIPerf** is its newer successor with closely matching options. Both drive OpenAI-compatible or Triton endpoints with synthetic or recorded prompts at a fixed concurrency or request rate and report TTFT, ITL, E2E latency, throughput and percentiles.' },
        { t: 'code', lang: 'bash', title: 'Benchmark your vLLM endpoint at several concurrency levels', x: `pip install genai-perf      # or use the nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk container

for c in 1 8 32 64 128; do
  genai-perf profile \\
    -m my-sft \\
    --url localhost:8000 \\
    --endpoint-type chat \\
    --streaming \\
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \\
    --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 \\
    --output-tokens-mean 256 \\
    --extra-inputs max_tokens:256 --extra-inputs ignore_eos:true \\
    --concurrency $c \\
    --warmup-request-count 10 \\
    --request-count 300
done
# Older releases also need: --service-kind openai
# AIPerf: aiperf profile --model my-sft --url localhost:8000 --endpoint-type chat --streaming \\
#         --concurrency 32 --synthetic-input-tokens-mean 512 --output-tokens-mean 256 ...
# Flag names evolve between releases - check --help for your version.` },
        { t: 'ul', items: [
          '`--streaming` is required to measure TTFT and ITL; without it you only get E2E latency.',
          '`ignore_eos` + `max_tokens` fixes output length so runs are comparable.',
          'Run the client close to the server (same cluster) and warm up first so you measure the model, not the network or cold caches.',
          'Use `--request-rate` instead of `--concurrency` to model open-loop arrival (users who do not wait for each other).',
        ]},

        { t: 'h2', x: 'SLO-driven capacity planning' },
        { t: 'p', x: 'Capacity planning turns an SLO and a traffic forecast into a GPU count. The tool is **Little\'s law**: average in-flight requests L = arrival rate λ × average time in system W.' },
        { t: 'ol', items: [
          '**SLO**: p95 TTFT < 500 ms and TPOT < 15 ms (≥ 66 tokens/s per user) for an 8B chat model.',
          '**Traffic**: peak 20 requests/s; 512 prompt tokens and 256 output tokens on average.',
          '**Benchmark** (table above): concurrency 64 per GPU meets both SLOs (TTFT p95 310 ms, TPOT 11.4 ms); 128 does not.',
          '**Time per request**: W ≈ 0.31 + 255 × 0.0114 ≈ 3.2 s.',
          '**In-flight requests at peak**: L = 20 × 3.2 ≈ 64.',
          '**GPUs**: 64 in flight / 64 per GPU = 1.0 GPU. Add 30% headroom for bursts and length variance → 1.3 → round up to 2. Add N+1 so a failure or rolling update does not breach the SLO → **3 GPUs** (replicas).',
          '**Cross-check with tokens**: demand = 20 × 256 = 5,120 output tok/s; one GPU at concurrency 64 gives 5,600 tok/s — consistent, and confirms there is little slack at one GPU.',
        ]},
        { t: 'callout', kind: 'tip', title: 'Then re-measure', x: 'After a change that moves the curve — FP8 quantisation, a TensorRT-LLM engine, speculative decoding, a new GPU — re-run the sweep and redo the arithmetic. A model that is 1.6× faster at the same latency needs proportionally fewer GPUs; that is how optimisation work (module 7) turns into money.' },

        { t: 'h2', x: 'Streaming, online and offline inference' },
        { t: 'p', x: '[[streaming|Streaming]] (server-sent events, `stream=True`) sends tokens as they are generated. It does not make the request finish sooner — it makes the wait **perceived** as TTFT instead of E2E, and it is required for measuring TTFT client-side.' },
        { t: 'code', lang: 'python', title: 'Measure TTFT and TPOT from the client', x: `import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
t0 = time.perf_counter()
first, n_chunks, usage = None, 0, None
stream = client.chat.completions.create(
    model="my-sft", max_tokens=256, stream=True,
    stream_options={"include_usage": True},        # final chunk carries token counts
    messages=[{"role": "user", "content": "Explain tensor parallelism."}],
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        first = first or time.perf_counter()
        n_chunks += 1
    if chunk.usage:
        usage = chunk.usage
end = time.perf_counter()
n_out = usage.completion_tokens if usage else n_chunks
print(f"TTFT {1000*(first - t0):.0f} ms | TPOT {1000*(end - first)/max(n_out - 1, 1):.1f} ms "
      f"| E2E {end - t0:.2f} s | {n_out} tokens")` },
        { t: 'vs', cards: [
          { h: 'Online (real-time) serving', items: [
            'A server answers requests as they arrive, usually streaming',
            'Optimise for latency SLOs (TTFT, TPOT) at acceptable cost',
            'Capacity sized for peak traffic plus headroom',
            'Chatbots, copilots, RAG assistants, agents',
          ]},
          { h: '[[offline-inference|Offline (batch) inference]]', items: [
            'A job processes a fixed dataset; no user is waiting',
            'Optimise for throughput and cost per million tokens — run at max batch size',
            'No HTTP server needed: in-process engine (vLLM `LLM`, TensorRT-LLM) or a batch API',
            'Evaluation runs, synthetic-data generation, document labelling, embeddings backfills',
          ]},
        ]},
        { t: 'code', lang: 'python', title: 'Offline batch inference with vLLM (no server)', x: `from vllm import LLM, SamplingParams

llm = LLM(model="./qwen-sft-merged", max_model_len=4096, gpu_memory_utilization=0.9)
params = SamplingParams(temperature=0.0, max_tokens=128)

conversations = [[{"role": "user", "content": f"Summarise ticket #{i} in one line: ..."}] for i in range(10_000)]
outputs = llm.chat(conversations, params)      # the engine batches all 10k internally, max throughput
for o in outputs[:3]:
    print(o.outputs[0].text)` },
        { t: 'callout', kind: 'exam', title: 'How this is tested', x: [
          'Definitions and formula: E2E = TTFT + (N_out − 1) × TPOT; TTFT is prefill + queueing, TPOT is decode; throughput rises with batch/concurrency while per-user latency worsens.',
          'Diagnosis: "TTFT grows sharply at peak while TPOT is fine" → queueing / not enough replicas; "TPOT too slow at low load" → decode-bound — quantise, use a faster engine or speculative decoding.',
          'Tooling: GenAI-Perf (or AIPerf) for LLM endpoint benchmarks with concurrency sweeps; offline batch for throughput-only workloads.',
        ]},

        { t: 'h2', x: 'Key takeaways' },
        { t: 'ul', items: [
          'TTFT = responsiveness (prefill + queue); TPOT/ITL = streaming speed (decode); E2E = TTFT + (N−1) × TPOT; goodput = throughput within SLO.',
          'Decode is memory-bandwidth-bound, so batching many sequences multiplies throughput until the KV cache fills and queueing begins.',
          'Benchmark with GenAI-Perf/AIPerf using realistic lengths, streaming and a concurrency sweep; report p95/p99.',
          'Capacity = Little\'s law on the SLO-safe concurrency per GPU, plus headroom and N+1.',
          'Stream for interactive use; run offline batch jobs in-process at maximum batch size.',
        ]},
      ],
    },
  ],

  quiz: [
    { id: 'y01', domain: 'deployment',
      scenario: 'A SaaS company has fine-tuned 40 LoRA adapters (rank 16) on the same Llama 3.1 8B base, one per customer. Most customers send a few requests per minute.',
      q: 'Which serving design uses GPU memory most efficiently?',
      opts: [
        'Merge each adapter and run 40 separate deployments, one per customer',
        'Serve the base once with multi-LoRA and select each adapter by model name',
        'Merge all 40 adapters into a single checkpoint by summing their deltas',
        'Fine-tune one new model on all customers\' data and retire the adapters',
      ],
      a: 1,
      why: '[[multi-lora|Multi-LoRA]] keeps one copy of the 16 GB base in memory and loads tens-of-MB adapters on demand; requests for different adapters can share a batch. vLLM (`--enable-lora`) and NIM (`NIM_PEFT_SOURCE`) both support it.',
      distract: '40 merged deployments need ~40 × 16 GB of weights for low-traffic tenants. Summing deltas mixes customers\' behaviour into one model. Retraining a combined model is costly and loses per-customer isolation.' },

    { id: 'y02', domain: 'deployment',
      q: 'On which ports does Triton Inference Server listen by default for HTTP, gRPC and Prometheus metrics?',
      opts: [
        '8080 HTTP, 9090 gRPC, 9100 metrics',
        '80 HTTP, 443 gRPC, 8000 metrics',
        '8000 HTTP, 8001 gRPC, 8002 metrics',
        '8000 for all three, multiplexed by path',
      ],
      a: 2,
      why: 'Triton defaults: **8000** HTTP/REST (KServe v2), **8001** gRPC, **8002** `/metrics` in Prometheus format.',
      distract: '9090 and 9100 are Prometheus server and node-exporter defaults. 80/443 are web defaults. Triton uses separate listeners, not one multiplexed port.' },

    { id: 'y03', domain: 'deployment',
      scenario: 'An LLM Deployment on Kubernetes autoscales on CPU utilisation with a 70% target. During a traffic spike, p95 TTFT rises from 300 ms to 9 s, but the HPA never adds replicas.',
      q: 'What is the best fix?',
      opts: [
        'Scale on a custom metric such as queued requests or KV-cache usage',
        'Lower the CPU target to 10% so the HPA reacts much sooner to spikes',
        'Decrease the pod CPU request so measured utilisation appears higher',
        'Switch to vertical pod autoscaling of the pods\' CPU and memory',
      ],
      a: 0,
      why: 'LLM servers are GPU-bound; CPU stays low while requests queue. [[autoscaling|Autoscaling]] on queue depth or KV-cache usage tracks the condition that causes TTFT to rise.',
      distract: 'A very low CPU target would scale unpredictably on noise, not load. Changing the CPU request only alters the ratio, not the signal\'s relevance. Vertical scaling of CPU/memory adds no GPU capacity.' },

    { id: 'y04', domain: 'deployment',
      scenario: 'A pod spec requests `nvidia.com/gpu: 1`. The pod stays Pending with "Insufficient nvidia.com/gpu", although `nvidia-smi` on the node works and the node has 4 idle GPUs.',
      q: 'What is the most likely cause?',
      opts: [
        'The container image does not include the NVIDIA kernel driver',
        'GPUs must be requested as `nvidia.com/gpu: 0.25` per container',
        'The node has no allocatable CPU left for the pod\'s requests',
        'The NVIDIA device plugin from the GPU Operator is not running on that node',
      ],
      a: 3,
      why: 'The scheduler only knows about `nvidia.com/gpu` if the device plugin advertises it; without it, node Allocatable shows no GPUs. The [[gpu-operator|GPU Operator]] deploys the device plugin with the driver and toolkit.',
      distract: 'The driver belongs on the host, and its absence in the image would not affect scheduling. Fractional GPU requests are invalid. A CPU shortage would report "Insufficient cpu".' },

    { id: 'y05', domain: 'deployment',
      q: 'A request has TTFT = 400 ms, TPOT = 25 ms and produces 201 output tokens. What is its end-to-end latency?',
      opts: [
        '5.0 s',
        '5.4 s',
        '5.425 s',
        '10.05 s',
      ],
      a: 1,
      why: 'E2E = TTFT + (N_out − 1) × TPOT = 0.4 + 200 × 0.025 = 5.4 s. The first token\'s time is already inside TTFT.',
      distract: '5.0 s ignores TTFT. 5.425 s counts the first token twice (201 × 25 ms). 10.05 s is 201 × 50 ms, doubling the per-token time.' },

    { id: 'y06', domain: 'deployment',
      scenario: 'Benchmarks of a model server show: concurrency 16 → 2,100 tok/s, p95 TTFT 120 ms; concurrency 64 → 5,200 tok/s, p95 TTFT 450 ms; concurrency 128 → 5,500 tok/s, p95 TTFT 3.8 s. The SLO is p95 TTFT < 500 ms.',
      q: 'What concurrency per replica should capacity planning assume?',
      opts: [
        '128, because it gives the highest throughput',
        '16, because it gives the lowest latency',
        '64, the highest tested load that still meets the SLO',
        'Any value — capacity should be planned on average TTFT, not p95',
      ],
      a: 2,
      why: 'Plan on [[goodput|goodput]]: the highest load that still meets the SLO. At 128 the server is past the knee — throughput barely improves while TTFT breaks the SLO through queueing.',
      distract: '128 maximises raw throughput but violates the SLO. 16 meets it with large waste (2.5× fewer tokens per GPU). Averages hide the tail that the SLO is written for.' },

    { id: 'y07', domain: 'deployment',
      scenario: 'A NIM container restarts on every deploy and spends 12 minutes downloading model files from NGC before becoming ready.',
      q: 'What is the most effective change?',
      opts: [
        'Mount a persistent cache volume (or NIM Operator NIMCache PVC) at the NIM cache path',
        'Increase the liveness probe failure threshold so the download is not interrupted',
        'Set NGC_API_KEY as a build argument so the image includes the model',
        'Expose port 8002 so Triton can stream the model faster',
      ],
      a: 0,
      why: 'NIM downloads profiles into its cache directory (`/opt/nim/.cache`). Persisting it — a host volume with Docker, or a PVC populated by a [[nim-operator|NIMCache]] in Kubernetes — turns restarts into fast local loads.',
      distract: 'A longer probe threshold tolerates the download but does not remove it. Build arguments end up in image layers (a secret leak) and NIM images are not rebuilt this way. Port 8002 is Triton\'s metrics port and has nothing to do with downloads.' },

    { id: 'y08', domain: 'deployment',
      scenario: 'A team must classify 50 million archived support tickets with an LLM by the end of the week. No user waits for individual results.',
      q: 'Which approach minimises cost?',
      opts: [
        'An autoscaled online endpoint with streaming, called once per ticket',
        'Low-concurrency online requests to keep TTFT under 200 ms each',
        'Multi-LoRA serving so each ticket can use a different adapter',
        'Offline batch inference with an in-process engine at maximum batch size',
      ],
      a: 3,
      why: '[[offline-inference|Offline batch inference]] optimises purely for throughput: no HTTP overhead, no latency SLO, maximum batching — the lowest cost per token.',
      distract: 'Streaming and per-request HTTP add overhead and target latency nobody needs. Keeping TTFT low means small batches — the opposite of cheap. Multi-LoRA solves an adapter-count problem, not this one.' },
  ],
}
