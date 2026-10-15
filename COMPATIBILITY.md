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

GraphRAG run `graph_full_01` was stopped after CPU-side feeding stalls with uncontrolled intra-op threading. Run `graph_full_02` uses one OpenMP/MKL/OpenBLAS/PyTorch intra-op thread; the native 32-worker concept-extraction pool remains. A `py-spy` stack snapshot places workers inside spaCy pipeline components; this is a CPU path, not GPU kernel evidence. A diagnostic CPU probe compares serial and batched spaCy on 16 actual fragments with identical entity outputs; its timings are not an isolated end-to-end speedup claim. The pinned spaCy assets are model 3.7.1, loaded by runtime 3.8.16, which emits a compatibility warning; record this limitation rather than asserting a paper-identical NLP environment.

Run `graph_full_03` is an explicit software control: the full spaCy pipeline uses `nlp.pipe(batch_size=32)` before the native concurrent concept LLM calls. Sixteen fragments receive an exact entity-span/label parity check against individual calls; no NLP component is removed. All CPU costs remain charged. The entire host is shared: a contemporaneous load average was about 106 on 104 CPUs, with other users accounting for roughly 104 cores in lifetime-averaged process CPU statistics. These runs cannot establish isolated CPU throughput or a causal end-to-end speedup from thread settings.

GraphRAG's native `ScopedTracker` reports zero completion tokens in this installed SDK's structured-parse path despite positive usage fields in actual server responses. Final token statistics therefore use the independently recorded router responses, not the native aggregate counters. Summed concurrent request latency is also kept distinct from the union of request intervals and phase wall time.

Retrieval settings: artifact Mem0 uses top-10; the GraphRAG pilot also uses top-10 (upstream example uses 5); EmbedRAG and BM25 use the upstream top-5. Their chunk representations differ, so equal `k` would not imply equal prompt tokens. These settings do not constitute a controlled architecture or equal-quality algorithm ranking.
