# Runtime compatibility and measurement boundaries

All model/package/data downloads use `hf-mirror.com`, `ghfast.top`, or the Tsinghua PyPI mirror. Model revisions and dataset hashes are retained remotely. No inference request is sent to an external model service.

The existing CUDA 13 PyTorch environment could import but could not execute CUDA on the server driver. The isolated project environment uses PyTorch 2.8.0+cu128 and vLLM 0.10.2. vLLM uses `/usr/local/cuda-12.8`, FlashAttention, and the native sampler; the inherited FlashInfer package otherwise tried to compile with an obsolete `/usr/bin/nvcc`.

Upstream compatibility changes:

- Vendored `mem0/embeddings/langchain.py`: `from langchain.embeddings.base import Embeddings` becomes `from langchain_core.embeddings import Embeddings`. No algorithm is changed.
- The local router removes redundant `response_format=json_object` when an explicit `guided_json` schema is supplied. vLLM 0.10.2 rejects the two simultaneous guidance settings; the explicit object schema remains enforced.
- NLTK resources, spaCy model and tiktoken encodings are resolved to mirrored local assets. Tiktoken files are checked against the SHA-256 expected by the installed tiktoken implementation. Missing resources fail rather than trigger a direct download.

Deliberate experimental choices, **not** an exact paper reproduction: BF16 A800 measurements, 8B construction and 32B answering, temperature 0, thinking disabled, schema guidance, 4,096-token construction chunks, 256-token answer cap, and prefix caching initially disabled for call characterization. A generation hitting its token cap invalidates the run. Prefix caching and optimized launch paths are required controls before any hardware conclusion.

`embedding_server.py` uses the pinned HF model, last-token pooling, left padding and L2 normalization, with maximum kernel microbatch 32; actual batch and lengths are logged. This serving implementation must not be treated as a mature performance baseline merely because the model is genuine. `measure_embedding_graph.py` tests whether host launch overhead explains the initial latency floor; graph replay is a calibration control, not evidence that the native workload can batch independent facts.

The runner fails on errors that upstream Mem0 logs and swallows, as well as truncated model responses. Tiny fixtures are connectivity diagnostics only. Raw histories, prompts, responses, index operations, state and GPU samples remain on moe-server.
