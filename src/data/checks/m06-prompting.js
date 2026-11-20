// End-of-lesson comprehension checks for Module 6: Prompt Engineering. Keyed by lesson id.
export default {
  'prompting-basics': [
    {
      q: 'What changes inside the model when you add five demonstrations to a prompt (few-shot prompting)?',
      opts: [
        'Nothing in the weights; the examples only condition the predictions',
        'The final layer is briefly fine-tuned on the five demonstrations',
        'The KV cache stores the examples permanently for later requests',
        'The tokenizer adds the demonstration labels to its vocabulary',
      ],
      a: 0,
      why: '[[in-context-learning|In-context learning]] uses the examples purely as context. The weights are frozen; remove the examples and the behaviour is gone.',
      distract: 'No gradient step happens at inference. The KV cache lives only for the request (or a prefix cache that can be evicted) and is not learning. The vocabulary is fixed at training time.',
    },
    {
      q: 'Which statement best matches research findings on why few-shot demonstrations help?',
      opts: [
        'They work only if every demonstration label is correct',
        'They help mainly because they add more tokens of compute per answer',
        'They teach the model new facts that persist across requests',
        'They mostly convey the format, the label space and the input distribution',
      ],
      a: 3,
      why: 'Studies of ICL (e.g. Min et al., 2022) found that format and label space carry much of the benefit; even randomised labels hurt less than expected.',
      distract: 'Label correctness matters less than expected, not exclusively. Extra compute per token is the argument for chain-of-thought, not for few-shot. Nothing persists between requests.',
    },
    {
      q: 'Why should you use `tokenizer.apply_chat_template` instead of building the prompt string by hand for an instruct model?',
      opts: [
        'It compresses the conversation so it uses fewer prompt tokens',
        'It renders the exact role markers and special tokens used in training',
        'It removes the system prompt, which instruct models do not support',
        'It converts the conversation into a base-model completion prompt',
      ],
      a: 1,
      why: 'Each model family has its own chat format (ChatML, Llama 3 headers, …). The template in the tokenizer config reproduces it exactly, and `add_generation_prompt=True` adds the assistant header.',
      distract: 'Templates add tokens rather than compress. Instruct models generally support system prompts. The template produces the chat format, not a base-model completion prompt.',
    },
    {
      q: 'You put a pasted customer email inside `<email>…</email>` tags in the prompt. What is the main benefit?',
      opts: [
        'XML tags are special tokens that the model processes faster than plain text',
        'Tags make the model skip the email content and answer from its own knowledge',
        'It separates data from instructions, improving reliability and limiting injection',
        'Tags are required by the chat template to render user messages correctly',
      ],
      a: 2,
      why: 'Clear delimiters tell the model where the data starts and ends, so instructions are not confused with content and embedded "instructions" in the data are less likely to be followed.',
      distract: 'Tags are ordinary text tokens, not faster. The model still reads the content — it is the data to work on. Chat templates do not require XML tags.',
    },
    {
      q: 'A vLLM server is running at localhost:8000. Which code change lets an application written for the OpenAI SDK talk to it?',
      opts: [
        'Install the vLLM Python package in the application and import vllm.LLM',
        'Convert all requests to gRPC and point them at port 8001',
        'Rewrite the requests to use the Hugging Face pipeline API',
        'Set base_url to http://localhost:8000/v1 and use the served model name',
      ],
      a: 3,
      why: 'vLLM (and NIM) implement the [[openai-api|OpenAI-compatible API]], so only `base_url` and the `model` value change.',
      distract: 'In-process `vllm.LLM` is offline batch inference, not a client. vLLM\'s OpenAI server is HTTP, not gRPC on 8001. The pipeline API runs a model locally rather than calling the server.',
    },
  ],

  'reasoning-decoding': [
    {
      q: 'What distinguishes few-shot chain-of-thought from zero-shot chain-of-thought?',
      opts: [
        'Few-shot CoT requires sampling, while zero-shot CoT requires greedy decoding',
        'Zero-shot CoT needs a dedicated reasoning model; few-shot CoT works on any model',
        'Few-shot CoT shows worked-reasoning examples; zero-shot CoT just adds "think step by step"',
        'Zero-shot CoT produces no intermediate steps, only the final answer line',
      ],
      a: 2,
      why: '[[chain-of-thought|Few-shot CoT]] (Wei et al.) shows examples of step-by-step reasoning; zero-shot CoT (Kojima et al.) elicits reasoning with a phrase and no examples.',
      distract: 'Either can use any decoding method. Zero-shot CoT works on ordinary instruct models. Zero-shot CoT does produce intermediate steps — that is its point.',
    },
    {
      q: 'In transformers you call `model.generate(..., do_sample=False, temperature=0.8, top_p=0.9)`. What happens?',
      opts: [
        'Greedy decoding; the temperature and top-p values are ignored',
        'Sampling at temperature 0.8 with nucleus 0.9',
        'Beam search with 0.8 × 0.9 as the length penalty',
        'An error, because temperature requires num_beams > 1',
      ],
      a: 0,
      why: 'With `do_sample=False` (and one beam) generation is [[greedy-decoding|greedy]]; sampling parameters have no effect (recent versions warn about it).',
      distract: 'Sampling needs `do_sample=True`. Beam search needs `num_beams > 1` and does not reinterpret these values. Temperature does not require beams; it is simply unused here.',
    },
    {
      q: 'The top token has probability 0.6. With min_p = 0.1, which tokens remain candidates?',
      opts: [
        'Only the top 10% of the vocabulary by rank',
        'Tokens until the cumulative probability reaches 0.1',
        'Exactly the 10 most likely tokens',
        'Tokens whose probability is at least 0.06',
      ],
      a: 3,
      why: '[[min-p|Min-p]] keeps tokens with p ≥ min_p × p_max = 0.1 × 0.6 = 0.06, so the cut-off adapts to how confident the model is.',
      distract: 'A rank fraction is not a standard sampler. Cumulative mass is top-p. A fixed count is top-k.',
    },
    {
      q: 'Why does a repetition penalty of 2.0 often damage JSON or code output?',
      opts: [
        'It raises the effective temperature, making the whole output random',
        'It penalises tokens that must legitimately repeat, like quotes and keys',
        'It disables stop sequences, so the output never terminates cleanly',
        'It silently switches to beam search, which cannot emit brackets',
      ],
      a: 1,
      why: '[[repetition-penalty|Repetition penalties]] down-weight every token already in context; structured formats need many repeated tokens, so a strong penalty pushes the model to wrong alternatives.',
      distract: 'Penalties change specific logits, not temperature. They do not affect stop sequences or switch decoding strategy.',
    },
    {
      q: 'You use vLLM guided decoding with a JSON Schema, but some responses still fail to parse. What is the most likely cause?',
      opts: [
        'The schema was ignored because temperature was 0',
        'Guided decoding only validates JSON after generation',
        'Generation hit max_tokens before the JSON object closed',
        'Guided decoding cannot handle enum fields',
      ],
      a: 2,
      why: '[[constrained-decoding|Constrained decoding]] prevents invalid tokens but cannot finish an object if the token budget runs out; check `finish_reason == "length"` and raise `max_tokens`.',
      distract: 'Temperature does not disable guidance. Masking happens during generation, not after it. Enums (`Literal`) are well supported.',
    },
  ],

  'rag-domain': [
    {
      q: 'A requirement says: "every answer must be traceable to the source document, and documents change daily". Which approach is designed for this?',
      opts: [
        'Full fine-tuning on the documents every night',
        'Retrieval-augmented generation with numbered, cited sources',
        'Few-shot prompting with three example answers',
        'Raising temperature so the model explores more documents',
      ],
      a: 1,
      why: '[[rag|RAG]] injects the retrieved passages at query time, so you know which sources were used and updates only need re-indexing.',
      distract: 'Fine-tuning bakes knowledge into weights with no traceability and costly updates. Few-shot examples do not provide changing knowledge. Temperature does not retrieve anything.',
    },
    {
      q: 'Why do RAG pipelines usually add overlap between consecutive chunks?',
      opts: [
        'To double the number of vectors, which always improves ANN recall',
        'Because embedding models require overlapping inputs to work at all',
        'To keep the vector index evenly balanced across database shards',
        'So facts that cross a chunk boundary appear whole in at least one chunk',
      ],
      a: 3,
      why: 'Overlap (often 10–20%) prevents boundary cuts from splitting a fact or losing its context.',
      distract: 'More vectors do not inherently improve recall and cost storage. Embedding models take independent inputs. Shard balance is unrelated to chunk overlap.',
    },
    {
      q: 'In reciprocal rank fusion with k = 60, a document ranked 1st by BM25 and 3rd by dense retrieval scores about:',
      opts: [
        '1/61 + 1/63 ≈ 0.0323',
        '1 + 3 = 4, lower is better',
        '(1 + 3) / 2 = 2, the mean rank',
        'max(1/61, 1/63) ≈ 0.0164',
      ],
      a: 0,
      why: 'RRF sums 1/(k + rank) over the ranked lists: 1/61 + 1/63 ≈ 0.0164 + 0.0159 = 0.0323. Higher is better.',
      distract: 'Summing or averaging ranks is not RRF and over-weights tail positions. RRF sums contributions; taking the max discards the second list.',
    },
    {
      q: 'A retrieval-tuned embedding NIM accepts `input_type`. What happens if you embed user questions with `input_type: "passage"`?',
      opts: [
        'The request is rejected by the NIM with a schema validation error',
        'Nothing changes; the parameter only affects usage billing and logs',
        'Recall silently drops because queries are encoded as documents',
        'Vectors become twice as long and the index must be rebuilt',
      ],
      a: 2,
      why: 'Asymmetric retrieval models encode queries and passages differently; mixing them up lowers recall without any error.',
      distract: '"passage" is a valid value, so nothing is rejected. It changes the embedding, not billing. Vector dimension is fixed by the model.',
    },
    {
      q: 'Which practice most directly reduces hallucinated answers when retrieval finds nothing relevant?',
      opts: [
        'Instruct the model to answer only from sources, else say "I don\'t know"',
        'Increase top_k of the sampler so more candidate tokens are considered',
        'Remove citations, because they encourage the model to invent sources',
        'Put the retrieved passages after the question in the middle of the prompt',
      ],
      a: 0,
      why: 'Giving the model an explicit, permitted way out — plus grounding instructions and low temperature — sharply reduces invented answers when context is missing.',
      distract: 'A larger sampler top-k increases randomness. Citations make answers verifiable; missing citations do not prevent invention. Middle placement is where models use context worst.',
    },
  ],
}
