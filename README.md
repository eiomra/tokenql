# TokenQL

TokenQL is a SQL-like interface and a custom, bounded-memory Qwen inference runtime. It can inspect next-token probabilities, explicitly commit tokens, generate complete responses, persist token histories, and rewind a session.

The source code is publicly available in this repository at
[github.com/eiomra/tokenql](https://github.com/eiomra/tokenql). TokenQL is an
experimental software project, not a peer-reviewed publication, and it has no
DOI or publication identifier. No software license has been selected yet, so
public access should not be interpreted as permission to redistribute or
incorporate the code into another project.

The longer implementation history, benchmark discussion, negative results, and
limitations are documented in the [unpublished technical draft](ARTICLE.md).

## Two interfaces

### 1. Normal user chat

Use `chat.py` when you only want to talk to the model:

```powershell
python .\chat.py --offline --threads 4 --weight-buffer-mb 32
```

```text
TokenQL Chat | backend=streamed-q8 | model=Qwen/Qwen2.5-0.5B-Instruct
Commands: /new clears the conversation, /thinking on|off changes reasoning, /exit quits
Thinking: off
you> Explain gravity simply.
ai> Gravity is the force that pulls objects toward one another.
you> Give me an example.
ai> An apple falls because Earth pulls it downward.
```

The chat keeps user and assistant messages as multi-turn context for the lifetime of the process. Enter `/new` to clear that context. Ask once without opening an interactive prompt:

```powershell
python .\chat.py --offline --prompt "Explain gravity in one sentence."
```

Replies are printed token by token while they are generated. Qwen3 thinking is off by default to avoid spending time and tokens on hidden reasoning. Start with `--thinking on`, or enter `/thinking on` and `/thinking off` during a chat to switch modes. Normal chat has no system prompt by default because every extra prompt token can route additional MoE experts during the expensive first pass; add one explicitly with `--system` when needed. The default maximum is 64 tokens. Useful controls include `--system`, `--max-tokens`, `--strategy`, `--temperature`, `--top-p`, `--top-k`, and `--seed`. Add `--stats` to see elapsed time, tokens per second, and logical weight reads after each answer.

### 2. Developer TokenQL console

Use `tokenql.py` to inspect distributions, commit individual tokens, study cache behavior, and manage persistent sessions:

```powershell
python .\tokenql.py --offline --threads 4 --weight-buffer-mb 32
```

Session IDs are persisted in `tokenql.db`, so ordinary `CREATE SESSION` intentionally rejects an existing ID. Reset the same ID with `CREATE OR REPLACE`, or remove it explicitly:

```sql
CREATE OR REPLACE SESSION 'chat-123'
WITH PROMPT 'Explain gravity in one short sentence.';

DROP SESSION 'chat-123';
```

Use `SHOW SESSIONS;` to list saved IDs.

Two execution backends are included:

- `streamed`: TokenQL's own Qwen2/Qwen2.5 and Qwen3-MoE decoder. Dense matrices are read from SSD in bounded chunks. MoE experts use a fixed-budget compressed LRU, and KV data is stored in an SSD-backed FP16 map. This backend does not instantiate `AutoModelForCausalLM`, Ollama, or llama.cpp.
- `transformers`: the original Hugging Face reference backend, retained for numerical comparisons.

PyTorch remains the portable kernel provider in the custom backend. TokenQL owns model paging, Q8/Q4 packing, activation quantization, integer matrix multiplication, RMSNorm, RoPE, grouped-query attention, causal masking, MLP execution, KV storage, vocabulary projection, sampling, and token iteration.

## Convert a model

The converter reads safetensors a bounded number of rows at a time. Every matrix is stored as signed INT8 with one FP32 scale per row; norms and biases remain FP32.

```powershell
python .\convert_tokenql.py `
  --source Qwen/Qwen2.5-0.5B-Instruct `
  --output .\models\qwen2.5-0.5b-tokenql-q8 `
  --offline `
  --chunk-mb 64
```

The already converted model on this computer is in `models/qwen2.5-0.5b-tokenql-q8`. Its original safetensors checkpoint is about 942 MiB; the TokenQL Q8 weights are about 473 MiB.

Create block-Q4 directly by adding `--quantization q4 --block-size 128`, or convert an existing TokenQL Q8 model with the lower-memory NumPy converter:

```powershell
python .\requantize_tokenql.py `
  --source .\models\qwen2.5-0.5b-tokenql-q8 `
  --output .\models\qwen2.5-0.5b-tokenql-q4 `
  --block-size 128 `
  --chunk-mb 16
```

The converted block-Q4 model is about 250.5 MiB. Q8 is the default for this small model because its AVX-512 integer path is faster and more accurate when the file already fits the operating-system cache. Q4 is the capacity option for models larger than available memory because it roughly halves SSD traffic.

Converter version 1 supports dense Qwen2/Qwen2.5 and text-only Qwen3-MoE safetensors checkpoints, including sharded checkpoints. Multimodal Qwen variants are not supported.

## Run a model larger than RAM with MoE paging

Qwen3-30B-A3B contains 30 billion total parameters but routes each token to a much smaller active subset. Convert it directly to block Q4. The source checkpoint is about 61 GB, so this is intentionally not downloaded by setup or tests:

```powershell
$env:HF_HUB_DISABLE_XET='1'
python .\convert_tokenql.py `
  --source Qwen/Qwen3-30B-A3B `
  --output .\models\qwen3-30b-a3b-tokenql-q4 `
  --quantization q4 `
  --block-size 128 `
  --chunk-mb 16
```

The converter keeps expert boundaries and coalesces all experts for one layer into one offset-addressable pack. A 48-layer checkpoint therefore produces 48 expert packs rather than tens of thousands of tiny expert fragments.

Conversion is resumable. If the process stops before `tokenql_manifest.json` is written, run the same command again. TokenQL validates the byte length of every completed tensor, reuses valid files, and overwrites only missing or incomplete tensors. The model is runnable only after the final manifest has been written.

Build the exact FP32-activation AVX-512 Q4 kernel when Visual Studio C++ Build
Tools are installed:

```powershell
py -3.12 .\build_native.py
```

It reads the converter's row-major Q4 nibbles directly and does not expand the
weight matrix to FP32 or INT8 in RAM. Runtime feature detection requires
AVX-512F, AVX-512BW, and AVX-512VBMI. Numba remains the portable compiled
fallback for aligned GEMV, batched prompt projection, and fused MoE gate/up.

The following prepack step is optional. It provides a PyTorch INT4 fallback for
machines where Numba is unavailable and can still be useful for unusual padded
matrix shapes. It is resumable, preserves the portable Q4 files, and creates
roughly one additional model-size copy of optimized sidecars:

```powershell
py -3.12 .\optimize_tokenql.py --model-dir .\models\qwen3-30b-a3b-tokenql-q4 --chunk-mb 16 --threads 4
```

With the native DLL and a supported CPU, chat starts with
`backend=paged-moe-q4-avx512`. Without the DLL it uses
`backend=paged-moe-q4-gemv` through Numba. Without Numba it uses
`backend=paged-moe-q4-native` when optimized sidecars and the required PyTorch
kernel are available, then falls back to the portable Q4 implementation.

Run with a hard budget for TokenQL-managed weight memory:

```powershell
py -3.12 .\chat.py `
  --backend streamed `
  --model-dir .\models\qwen3-30b-a3b-tokenql-q4 `
  --ram-budget-mb 4096 `
  --weight-buffer-mb 128 `
  --max-context 2048 `
  --matmul int8 `
  --threads 4 `
  --stats
```

`--ram-budget-mb` covers TokenQL's managed weight workspace, resident FP32
vectors, guaranteed-reuse compressed shared matrices, and the remaining expert
cache. Output activations, Python/PyTorch libraries, tokenizer state, and the
currently read KV layer are reported separately by the operating system and are
not included in this managed-weight limit. Reducing `--max-context` bounds the
disk KV file and per-layer KV reads.

The MoE cache allocates whole expert slots across layers, uses frequency-gated
admission to learn recurring routes without sequential-scan thrashing, and
asynchronously loads the exact selected experts once the router result is
known. `--expert-prediction` additionally experiments with using the previous
token's routes during attention, but is off by default because it increased I/O
and latency on the Qwen3-30B-A3B benchmark.

The Numba fallback's first launch can include a few seconds of LLVM compilation;
compiled kernels are stored under `__pycache__`. Four threads benchmark best on
the tested 4-core/8-thread Ice Lake laptop. More logical threads can reduce
performance because packed-Q4 inference is memory-bandwidth limited.

On that laptop, a warmed synthetic single-token loop can exceed 1 token/second
with a 4 GiB managed budget. This is not the end-to-end chat rate. A reproducible
13-token `hi` prefill plus a 12-token reply measured 59.50 seconds overall
(0.20 output token/second), including 33.70 seconds of prefill; decode alone was
0.44 token/second. Prompt expert streaming and decode-time expert misses remain
the main full-pipeline bottlenecks.

Interactive `chat.py` retains the disk KV cache across append-only turns. Qwen's
thinking-off template is asymmetric for current versus historical assistant
messages, so TokenQL preserves the exact evaluated token prefix and appends only
the new turn suffix rather than re-rendering old tokens. In a two-turn test this
reduced the second prefill from 39 tokens/36.31 seconds to 19 tokens/21.55
seconds. The request timer now includes both continuation prefill and decode;
older measurements that reported only decode for later turns are not directly
comparable. `/new` drops the conversation KV state while leaving the
process-wide shared/expert caches warm.

`--io-workers` controls concurrent expert readers and defaults to one. Four
workers greatly increased the count of futures ready before use. In a seeded
four-turn chat test they reduced aggregate request time by about 4.7%. Expert
misses now use one contiguous read for all three Q4 projections, and layer-major
prefill keeps a bounded window of exact routed experts in flight. Together these
changes reduced the same four-turn aggregate from 338.06 to 281.37 seconds
(16.8%), continuation prefill to 16.85--20.42 seconds, and decode I/O wait to
21--24 seconds. Decode remained below one token/second, so keep the worker count
explicit while benchmarking; readiness counts alone should not be treated as
I/O latency. Stats report measured decode I/O wait seconds. Three compute
threads were also slower than four.

`--expert-prediction` remains experimental and off by default. The original
previous-token/all-route predictor reached only 43.4% accuracy and increased
warm-turn weight traffic by roughly 45--52%. The current conservative predictor
selects at most two routes present in both prior tokens, launches them one layer
early, and uses a separate single-worker speculative queue so exact reads cannot
sit behind guesses. It reached 81.5% precision, but only 5 of 834 candidates
produced useful I/O: stable routes were already resident in the LFU cache. Decode
was 0.79 versus 0.85 token/second without prediction, so it is not enabled by
default.

A decode profile also exposed repeated `Path.resolve()` and filesystem metadata
queries on every matrix access. Resolving each distinct pack once at startup
improved the same 11-step decode from 0.67 to 0.85 token/second and reduced a
13-token prefill from 20.04 to 15.57 seconds. A paired AVX-512 gate/up kernel is
bit-identical and 1.40x faster in isolation, but full decode remained 0.86
token/second because expert I/O was then the dominant wall. Reusing one
unbuffered pack handle per I/O thread instead of reopening a file on every miss
subsequently crossed the target at the original 4 GiB budget: a two-turn
verification decoded at 0.99 and 1.01 token/second, with 11.66- and 7.93-second
prefills. The second 32-token request was 0.79 output token/second end to end
because that overall rate includes prefill.

With persistent handles active, eight I/O workers are the measured cold-prefill
sweet spot on the test laptop: at the 4 GiB budget they reduced a seeded
13-token prefill from 11.66 to 7.26 seconds and the first response from 23.75 to
17.68 seconds while preserving 1.03--1.12 decode token/second. Sixteen workers
regressed prefill to 11.23 seconds due to contention. An 8 GiB budget with four
I/O workers instead favors warm throughput: the seeded second turn improved
from 40.25 to 33.51 seconds, decode from 1.01 to 1.22 token/second, and logical
weight reads from 14.52 to 4.98 GiB. Worker count remains explicit because the
best storage queue depth is hardware-specific.

At the 8 GiB budget, neither worker count dominated every metric. Eight workers
reduced first-turn latency from 21.90 to 20.47 seconds, while four workers
produced marginally higher warm decode throughput (1.22--1.25 versus
1.18--1.19 token/second) and a slightly faster second turn (33.51 versus 33.88
seconds). Eight workers had the lower two-turn aggregate at 54.35 seconds, but
the difference was only about 2%. At the 4 GiB budget, eight workers retained
the fastest and lowest-memory cold greeting. These results should be treated as
hardware-specific tradeoffs rather than a universally best worker setting.

`--prefill-layout-profile` measures hypothetical exact expert-read grouping
without changing the generated result. On the test prompts, a single
min-to-max layer span would amplify bytes by 3.14x cold and 5.70x warm, so that
policy is unsuitable. Grouping only byte-adjacent selected experts had 1.00x
amplification and could reduce read calls by 30.1% cold/13.7% warm, but the
implemented bounded version increased measured prefill from 7.26 to 12.61
seconds cold and 7.36 to 8.75 seconds warm. Persistent handles had already made
individual reads cheap, while independent copies were required to preserve
accurate per-expert RAM accounting. The implementation is therefore available
only behind `--prefill-coalescing` and remains off by default. Cold prefill also
varied from 7.26 to 11.21 seconds across otherwise identical eight-worker runs,
so isolated minima should not be treated as sustained throughput.

The bounded g1 policy was subsequently tested with
`--prefill-coalescing-gap 1` and no layout-profiler overhead. It lost more
decisively than g0: cold/continuation prefill rose from 12.61/8.75 seconds to
15.80/12.80 seconds, first-turn reads rose from 8.01 to 8.73 GiB, and the two
requests rose from 63.85 to 84.65 seconds total. Decode also fell to
0.77--0.80 token/second as the additional reads and copies contended with the
following decode. Because g2 has still greater measured byte amplification,
testing stopped at g1. `--prefill-coalescing-gap` accepts 0, 1, 2, 4, or 8 for
experiments, but no tier is recommended for normal use.

`SHOW RUNTIME;` reports shared/expert-cache capacity and residency, hits, misses,
evictions, bypasses, hit rate, slot slack, prefetch usefulness/readiness, and
prediction accuracy. The chat `--stats` line separates prefill time, decode
throughput, shared-cache residency, prefetch, and expert-cache results.

## Run the streamed backend

Because the converted model directory exists in this workspace, automatic backend selection uses it. Choose `chat.py` for normal conversation or `tokenql.py` for technical queries:

```powershell
python .\chat.py --offline
python .\tokenql.py --offline
```

The startup banner will say `backend=streamed-q8`. The fully explicit equivalent is:

```powershell
python .\tokenql.py `
  --backend streamed `
  --model-dir .\models\qwen2.5-0.5b-tokenql-q8 `
  --weight-buffer-mb 32 `
  --max-context 4096 `
  --matmul auto `
  --threads 4
```

Ask for a complete response with one command:

```sql
SELECT RESPONSE FROM MODEL
WHERE PROMPT = 'Explain gravity in one short sentence.';
```

The output is plain text. Sampling can be controlled in the same command:

```sql
SELECT RESPONSE FROM MODEL
WHERE PROMPT = 'Write a greeting in French.'
WITH (max_tokens=80, temperature=0.7, top_p=0.9);
```

Inspect runtime I/O and configuration:

```sql
SHOW RUNTIME;
```

Use Q4 explicitly when capacity or SSD traffic matters more than small-model speed:

```powershell
python .\chat.py `
  --backend streamed `
  --model-dir .\models\qwen2.5-0.5b-tokenql-q4 `
  --weight-buffer-mb 16 `
  --matmul int8 `
  --threads 4 `
  --stats
```

## What “SSD streamed” means

TokenQL maps one matrix file, reads only a configured row chunk, performs that matrix operation, and closes the mapping before moving on. A bounded background buffer prefetches the next chunk while the current chunk is computed. The model's total weight size therefore does not determine the executor's largest weight allocation. `--weight-buffer-mb` controls the combined buffers; `--no-prefetch` disables overlap for comparison.

The operating system can retain recently accessed file pages in its reclaimable filesystem cache. This improves repeat speed but is not the same as TokenQL allocating the whole model on its heap. If weights exceed physical RAM, the operating system evicts old pages and later reads them from SSD.

A dense model still requires every layer for every generated token. A 100B Q8 model needs roughly 100 GB of SSD storage and approximately one full 100 GB logical scan per token. Low RAM makes such execution possible, not fast. Batching many independent queries through each loaded layer is the next major throughput optimization.

A database can avoid scanning millions of rows because indexes identify a small subset of relevant pages. A dense transformer has no corresponding index: every weight matrix participates in every token. MoE expert routing is the closest model architecture to an index because the router selects only a subset of expert weights. The disk-first runtime does not cache the full model in RAM, but dense-model latency cannot be eliminated by the query language alone.

The KV cache is a preallocated FP16 file. Only the currently executing layer's used prefix is copied into RAM. Its disk size depends on layers, KV heads, head dimension, and `--max-context`.

## Measured verification

Measurements on this computer using the converted Qwen2.5-0.5B model and a 32 MiB weight buffer:

- Model weights on disk: 473.1 MiB.
- Process RSS before streamed inference: 471.1 MiB, mostly Python, PyTorch, tokenizer, and native libraries.
- RSS after prefill: 556.8 MiB.
- Peak working set: 699.0 MiB.
- Disk-backed KV allocation at 512-token context: 6.0 MiB.
- One prefill plus two decode passes: 1,418.8 MiB of logical weight reads.
- Hugging Face and TokenQL selected the same next token, `OK`.
- Full-logit cosine similarity: 0.9937868.
- Top-10 token overlap: 10/10.
- Probability divergence, KL(HF || TokenQL): 0.0000806.

Q8 quantization intentionally changes logits slightly. The reference backend remains useful for checking new models and converter versions.

After adding integer activation/weight multiplication on this CPU, the same Q8 verification prefill decreased from approximately 3.0 seconds with float matrix multiplication to approximately 1.6 seconds. Its next-token choice remained `OK`; compared with the float Q8 path, full-logit cosine similarity was 0.9546 and KL divergence was 0.0010. Block-Q4 reduced each logical scan to 250.5 MiB, although its generic block kernel is slower on this small model and introduces more quantization error.

## Inspect and control individual tokens

Create or reset a persistent session:

```sql
CREATE OR REPLACE SESSION 'chat-123'
WITH PROMPT 'Explain gravity in one short sentence.';
```

Read the next-token distribution without modifying the session:

```sql
SELECT TOP 5 TOKEN, PROBABILITY
FROM PREDICT_NEXT_TOKEN('chat-123');
```

Commit a chosen token or let the configured strategy choose it:

```sql
UPDATE SESSION 'chat-123' APPEND TOKEN 9707;

UPDATE SESSION 'chat-123'
APPEND NEXT_TOKEN
WITH (strategy='greedy');
```

Generate, read, and rewind:

```sql
UPDATE SESSION 'chat-123'
GENERATE
WITH (strategy='greedy', max_tokens=80);

SELECT TEXT FROM SESSION('chat-123');

DELETE TOKENS FROM SESSION 'chat-123' WHERE POSITION >= 10;
```

Prompts and committed token IDs are persisted in SQLite. Runtime-specific KV files are rebuilt from that history after restarting.

## Reference backend

Run the original Hugging Face implementation when comparing logits or debugging a new converter:

```powershell
python .\tokenql.py --backend transformers --offline
```

## Architecture

```text
TokenQL statement
      |
query parser / session scheduler
      |
streamed Qwen executor
      |-- Q8/Q4 matrix pager ----- SSD shared-weight files
      |-- MoE router/cache ------- bounded frequency-aware expert cache
      |-- per-layer expert packs - offset-addressable SSD ranges
      |-- async double buffer ---- bounded I/O thread
      |-- integer matrix math ---- native AVX-512 / Numba fallback
      |-- manual Qwen operations - TokenQL executor
      |-- disk KV cache ---------- SSD cache file
      |-- tokenizer -------------- local tokenizer files
      |
SQLite token history
```

This is a research-grade reference implementation, not yet a high-performance replacement for llama.cpp. The Qwen3-MoE path is numerically checked against Hugging Face using a generated tiny checkpoint, including incremental KV-cache decoding and forced expert-cache eviction. Native fused block-Q4 kernels, asynchronous exact expert reads, layer-major prefill scheduling, and experimental expert prediction are implemented. The remaining production milestones include model-format integrity hashes, broader architecture support, repeated cross-hardware benchmarks, and matched comparisons with mature runtimes.
