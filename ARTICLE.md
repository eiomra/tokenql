# TokenQL: Engineering SSD-Backed Inference for a 30B MoE Model on a 16 GB Laptop

## A reproducible account of bounded-memory Q4 inference, expert paging, and the optimizations that actually mattered

**Author:** Oboyi Thompson

**Status:** Unpublished technical draft

**Draft updated:** August 2, 2026

---

## Abstract

Large language models are normally loaded into RAM or GPU memory before inference. That works well when enough memory is available, but it prevents many consumer computers from running models whose weights exceed their physical memory. I built TokenQL, an experimental SQL-like language model interface and custom Qwen inference runtime that keeps a quantized model on SSD, loads only the weights needed at each stage, and operates within a configurable memory budget.

On an HP Pavilion laptop with an Intel Core i5-1035G1 CPU, 16 GB of physical RAM, and a 500 GB WDC SSD, TokenQL runs the 30.5-billion-parameter Qwen3-30B-A3B mixture-of-experts model from a 30.05 GiB Q4 representation. With an 8 GiB **TokenQL-managed weight budget**, four CPU compute threads, and eight I/O workers, a representative three-turn chat produced decode rates of 1.20, 1.15, and 1.54 tokens per second. The first request still took 21.73 seconds overall, including a 12.00-second prompt-prefill phase. Later requests benefited from the warmed expert cache and reached up to 1.17 output tokens per second end to end.

The result is not a claim that a complete 30B model fits in 8 GB, nor that SSD paging is a new inference technique. It is a systems-engineering result: a custom converter, model format, decoder, compressed expert cache, disk-backed KV cache, asynchronous expert loading, and exact packed-Q4 CPU kernels were combined into a functioning bounded-memory runtime. This article describes the architecture, performance journey, failed experiments, limitations, and work still needed for a formal evaluation.

**Code availability:** The implementation is publicly available at
[github.com/eiomra/tokenql](https://github.com/eiomra/tokenql). This manuscript
has not been formally published and has no DOI or publication identifier. The
source code is released under the
[MIT License](https://github.com/eiomra/tokenql/blob/main/LICENSE).

## 1. Motivation: from SQL-style control to bounded-memory inference

TokenQL began as an attempt to expose autoregressive language-model generation through database-like operations. A normal user can request a complete answer:

```sql
SELECT RESPONSE FROM MODEL
WHERE PROMPT = 'Explain gravity in one short sentence.';
```

A developer can instead inspect and mutate generation one token at a time:

```sql
CREATE OR REPLACE SESSION 'demo'
WITH PROMPT 'Explain gravity in one short sentence.';

SELECT TOP 5 TOKEN, PROBABILITY
FROM PREDICT_NEXT_TOKEN('demo');

UPDATE SESSION 'demo'
APPEND NEXT_TOKEN
WITH (strategy='greedy');

SELECT TEXT FROM SESSION('demo');
```

This interface separates prediction, which is read-only, from committing a token, which mutates session state. SQLite stores prompts and committed token histories, while the runtime can rebuild its transient KV state after a restart.

The SQL syntax alone cannot make transformer inference cheaper. A database can use an index to avoid scanning irrelevant rows, but every weight in a dense transformer layer participates in every token. The architectural opportunity came from mixture-of-experts models: their routers select only a subset of experts for each token. Qwen3-30B-A3B has 30.5 billion total parameters but activates approximately 3.3 billion per token; it has 128 routed experts per MoE layer and selects eight of them for each token.[^1] That sparsity makes expert weights behave more like selectively accessed records, although the analogy is imperfect.

## 2. What TokenQL implements

The streamed TokenQL backend does not instantiate Hugging Face `AutoModelForCausalLM`, Ollama, or llama.cpp for inference. Hugging Face tokenizers and safetensors are used at conversion and tokenization boundaries, but TokenQL controls model paging and executes the Qwen decoder itself.

```text
chat.py or TokenQL statement
             |
     parser/session manager -------- SQLite token history
             |
       custom Qwen runtime
             |
   +---------+--------------------------+
   | shared Q4 weights                  | routed MoE experts
   | resident compressed cache          | per-layer SSD pack files
   +----------------+-------------------+
                    |
       async readers + bounded cache
                    |
       native packed-Q4 CPU kernels
                    |
        disk-backed FP16 KV cache
```

The main components are:

1. **A resumable model converter.** It reads sharded Hugging Face safetensors in bounded chunks and writes a TokenQL manifest plus block-Q4 weight files. Each layer's experts are stored contiguously in one offset-addressable pack, avoiding tens of thousands of separate files. Safetensors was a useful source format because it exposes tensor metadata and supports safe, efficient tensor storage.[^2]

2. **A manual Qwen/Qwen3-MoE decoder.** TokenQL implements RMS normalization, rotary position embeddings, grouped-query attention, causal masking, MoE routing, expert execution, sampling, and incremental decoding.

3. **Bounded shared and expert caches.** Frequently reused non-expert matrices remain in a compressed shared-weight cache. Routed experts occupy a separate frequency-aware cache with a fixed capacity. Entire expert entries—gate, up, and down projections—can be fetched with one contiguous read.

4. **A disk-backed KV cache.** The KV cache is stored as a preallocated FP16 file. Only the currently processed layer's required prefix is brought into memory. Append-only chat turns reuse their exact evaluated prefix instead of rerendering and recomputing the complete conversation.

5. **Native Q4 kernels.** A custom AVX-512 kernel consumes the converter's packed Q4 nibbles directly with FP32 activations. It does not expand an entire weight matrix to FP32 or INT8 in RAM. A paired gate/up kernel reduces duplicated traversal in the MoE feed-forward path. Numba provides a portable compiled fallback.

6. **Asynchronous expert I/O.** A bounded pool of I/O workers reads exact routed experts while available CPU work continues. File paths are resolved once, and every reader thread retains persistent unbuffered pack-file handles.

7. **Two user surfaces.** `chat.py` provides an ordinary user/assistant conversation, including Qwen's thinking on/off control. `tokenql.py` provides the technical SQL-like console for inspecting distributions, committing tokens, rewinding histories, and viewing runtime statistics.

## 3. Test platform and model

| Component | Configuration |
|---|---|
| Laptop | HP Pavilion Laptop 14-ce3xxx |
| Operating system | Windows 11 Home Single Language |
| CPU | Intel Core i5-1035G1, 4 cores / 8 logical processors |
| Physical RAM | 16,951,066,624 bytes, approximately 15.79 GiB |
| Storage | WDC WDS500G2B0A-00SM50, 500 GB SSD |
| Graphics present | Intel UHD Graphics and NVIDIA GeForce MX130 |
| Inference device | CPU; neither GPU was used for the reported run |
| Model | Qwen3-30B-A3B |
| Model architecture | 30.5B total parameters, approximately 3.3B activated per token, 48 layers, 128 experts, eight routed experts per token[^1] |
| TokenQL representation | Block Q4, block size 128 |
| Converted model size | 32,263,534,205 bytes, 30.05 GiB, 1,021 files |
| Context limit used | 2,048 tokens |
| Runtime configuration | 8,192 MiB managed weight budget, 128 MiB weight buffer, four compute threads, eight I/O workers |

The phrase **8 GiB managed weight budget** needs careful interpretation. It covers TokenQL's compressed shared matrices, expert-cache capacity, resident vectors, and managed weight buffers. It does **not** include the Python interpreter, PyTorch and tokenizer libraries, transient activations, the operating system's filesystem cache, or all process memory. The complete 30.05 GiB converted model remains on SSD. Therefore, the result should not be described as “a 30B model using only 8 GB of total RAM.”

## 4. The performance journey

The useful result was not produced by one idea. It came from repeatedly measuring the full pipeline and removing its current bottleneck.

| Stage | Representative result | Main finding |
|---|---:|---|
| Early Qwen3 streamed Q4 path | 0.02–0.03 tok/s | Per-token weight traffic and generic execution made the runtime unusable. |
| Initial native path | 0.07–0.10 tok/s | Native operations helped, but 39–43 GiB of logical reads over a 32-token reply still dominated. |
| Packed GEMV path | 0.16 tok/s | Direct packed-weight matrix-vector execution provided a real improvement. |
| AVX-512 and shared-cache work | 0.44–0.60 decode tok/s | Compute became fast enough to expose expert I/O as the primary wall. |
| Cached paths and filesystem metadata | 0.67 → 0.85 decode tok/s | Profiling found roughly 7,700 repeated `Path.resolve()` calls in 11 decode steps. Removing Python filesystem overhead mattered more than another speculative algorithm. |
| Persistent per-thread pack handles | 0.99–1.01 decode tok/s at 4 GiB | Reusing file handles and reading into existing buffers crossed the 1 tok/s target. |
| 8 GiB budget, eight I/O workers | 1.15–1.54 decode tok/s in a representative session | A larger expert cache improved warm-turn hit rate and reduced SSD traffic. |

Two lessons stand out. First, optimization changed the identity of the bottleneck several times: generic matrix math, Python orchestration, repeated metadata work, file opening, and finally cache misses and prefill. Second, low-level but unglamorous changes—caching resolved paths and retaining handles—produced some of the largest practical gains.

The Q4 representation also matters. Storing the 30B checkpoint in approximately 30 GiB is larger than the physical RAM but small enough for the laptop's SSD. Q4 reduces traffic relative to Q8, while the native kernel avoids materializing expanded copies of complete matrices.

## 5. Current end-to-end result

The following command produced the representative session reported here:

```powershell
py -3.12 .\chat.py --backend streamed --model-dir .\models\qwen3-30b-a3b-tokenql-q4 --ram-budget-mb 8192 --weight-buffer-mb 128 --max-context 2048 --matmul int8 --threads 4 --io-workers 8 --thinking off --max-tokens 32 --stats
```

| Prompt | Output tokens | Total time | Overall output rate | Prefill | Decode rate | Decode cache hit | Logical weight reads |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hi` | 12 | 21.73 s | 0.55 tok/s | 13 tokens / 12.00 s | 1.20 tok/s | 83.4% | 6.64 GiB |
| `what is photosynthesis?` | 32 | 34.44 s | 0.93 tok/s | 19 tokens / 5.95 s | 1.15 tok/s | 86.4% | 5.03 GiB |
| `what is the capital of france?` | 32 | 27.38 s | 1.17 tok/s | 21 tokens / 5.77 s | 1.54 tok/s | 91.9% | 3.41 GiB |

The trend is internally consistent: the expert cache warms from one turn to the next, decode cache hit rate rises, logical reads fall, and throughput improves. It is also important not to hide the cold-start experience behind the best number. The first short greeting required 21.73 seconds overall even though its decode phase exceeded one token per second.

These figures are observational results from one machine and a short conversational run. They are not yet a statistically rigorous benchmark. Some development runs used a fixed seed of 42 for controlled comparisons, but the table above should be treated as a representative interactive trace, not a mean with confidence intervals.

## 6. Why time to first token remains high

Autoregressive decode processes one new token at a time, but prefill must evaluate every prompt token before the first output token is available. For MoE inference, those prompt tokens may collectively route to many experts in every layer. On a cold start, few of those experts are resident, so prefill has to read a broad working set from SSD.

Later chat turns are faster for two reasons. The exact previously evaluated token prefix and its disk-backed KV data are retained, so TokenQL evaluates only the new turn suffix. At the same time, commonly reused experts have accumulated in the compressed cache.

Increasing the managed budget from 4 GiB to 8 GiB substantially improved warm throughput, but it did not reliably eliminate cold prefill because an empty larger cache is still empty. The next latency target is therefore prompt processing, not merely decode GEMV speed. Potential directions include more effective layer-major batched prefill, storage-aware expert layout learned from real routing traces, and safe cache prewarming.

## 7. Experiments that did not work

Negative results were among the most useful parts of this project because they prevented attractive but ineffective mechanisms from becoming defaults.

### 7.1 Previous-token expert prediction

An initial predictor assumed that the previous token's expert routes would be useful for the next token. It achieved only 43.4% route accuracy. More importantly, it raised warm-turn weight traffic by approximately 45–52% and slowed execution. Incorrect guesses competed with exact reads for SSD bandwidth and cache capacity.

A conservative stability predictor was then built. It selected at most two experts that appeared in both of the two previous route sets and launched them through a separate speculative queue. Its top-two precision reached approximately 81.5% at a two-token distance and 84.6–87.4% at a three-token distance. Nevertheless, only five of 834 candidates caused useful I/O: predictable experts were usually already cached. In one comparison, decode was 0.79 tok/s with prediction versus 0.85 tok/s without it. High prediction precision did not translate into high system value, so expert prediction remains disabled by default.

### 7.2 Coalescing nearby expert reads

Because all experts for one layer are stored in a contiguous pack file, combining nearby ranges seemed likely to reduce read calls. A layout profiler showed the tradeoff:

- One full min-to-max range per layer would read 3.14 times the exact bytes when cold and up to 5.70 times when warm.
- Combining only byte-adjacent selected experts had 1.00× byte amplification and could theoretically remove 30.1% of cold read calls.
- Allowing one unselected expert between combined ranges reduced calls further but introduced approximately 1.14–1.19× byte amplification in observed layouts.

Measured performance contradicted the syscall-count intuition. Adjacent-only coalescing increased cold prefill from 7.26 to 12.61 seconds and continuation prefill from 7.36 to 8.75 seconds in a controlled test. The one-gap policy was worse again: cold/continuation prefill rose to 15.80/12.80 seconds, first-turn reads increased from 8.01 to 8.73 GiB, and two requests grew from 63.85 to 84.65 seconds.

Persistent file handles had already made individual reads relatively cheap. Coalescing added buffer copies and scheduling work, while broader spans consumed SSD bandwidth with unused bytes. The code remains available as an experimental flag, but coalescing is off by default.

### 7.3 More threads are not automatically better

Four compute threads outperformed three on this four-core/eight-thread CPU, but
the I/O-worker results did not have one universal winner. At the 8 GiB budget,
eight workers reduced first-turn latency from 21.90 to 20.47 seconds, while four
workers produced marginally higher warm decode throughput (1.22–1.25 versus
1.18–1.19 tok/s) and a slightly faster second turn (33.51 versus 33.88
seconds). Sixteen workers regressed cold prefill because of contention. Queue
depth is hardware-specific and should be measured rather than assumed.

## 8. Relationship to prior work

TokenQL should be positioned as an independent engineering implementation, not as the invention of model offloading or quantized CPU inference.

FlexGen demonstrated high-throughput language-model inference under constrained GPU memory by placing weights and state across GPU, CPU, and disk, and by searching for an offloading policy.[^3] DeepSpeed-MoE developed scalable and efficient inference techniques for sparse mixture-of-experts models.[^4] llama.cpp provides mature quantized inference, CPU kernels, memory mapping, and CPU/GPU hybrid execution.[^5] vLLM's PagedAttention addresses a different but related memory-management problem by reducing fragmentation and redundant KV-cache storage in serving workloads.[^6]

TokenQL's contribution is the measured integration of several ideas for one constrained setting:

- a SQL-like control surface for prediction and explicit token mutation;
- a custom offset-addressable Q4 model format and converter;
- a manual Qwen3-MoE CPU decoder;
- bounded shared-weight and expert caches;
- persistent SSD-backed KV state across chat turns;
- exact packed-Q4 AVX-512 kernels; and
- a documented optimization trail that includes unsuccessful prediction and coalescing experiments.

This combination is useful and reproducible, but no priority or “first” claim is made. A fair comparison with llama.cpp or Ollama on the same laptop, model quantization, context, output length, and sampling settings is still required before making comparative performance claims.

## 9. Reproduction

Clone the public source repository and install its Python dependencies:

```powershell
git clone https://github.com/eiomra/tokenql.git
cd tokenql
py -3.12 -m pip install -r requirements.txt
```

After converting Qwen3-30B-A3B to the TokenQL Q4 format, build the native
kernel:

```powershell
py -3.12 .\build_native.py
```

Then launch the measured configuration:

```powershell
py -3.12 .\chat.py --backend streamed --model-dir .\models\qwen3-30b-a3b-tokenql-q4 --ram-budget-mb 8192 --weight-buffer-mb 128 --max-context 2048 --matmul int8 --threads 4 --io-workers 8 --thinking off --max-tokens 32 --seed 42 --stats
```

For a proper benchmark, the process should be repeated several times in both cold-cache and warm-cache conditions. Results should report:

- time to first token and prefill tokens per second;
- decode tokens per second, excluding prefill;
- end-to-end output tokens per second;
- physical process RSS and peak working set, separately from the managed weight budget;
- logical weight bytes requested and physical disk bytes read;
- expert-cache hit rate, evictions, and I/O wait time;
- CPU utilization, SSD model and throughput, energy use, and thermal state; and
- mean, median, standard deviation, and sample count.

The current automated suite contains 14 passing tests, including small-model numerical checks, incremental KV-cache decoding, and forced expert-cache eviction. A native complete-model comparison against the Numba path produced the same argmax, cosine similarity near 1.0, and a maximum reported logit error of approximately 2.19×10⁻⁵. Those checks validate implementation consistency; they are not a full model-quality evaluation against the original unquantized checkpoint.

## 10. Limitations and next work

The runtime is research-grade. Its present limitations include:

- **Slow cold time to first token.** A 5–12 second prefill remains noticeable, and cold launches can be slower.
- **Storage dependence.** SSD latency, bandwidth, operating-system caching, and thermal behavior materially affect results.
- **Single-machine evidence.** The endpoint has not yet been reproduced on a broad hardware set.
- **No matched mature-runtime baseline yet.** Comparisons with llama.cpp and Ollama are required.
- **Limited statistical evaluation.** Current traces are short and do not yet include repeated-trial distributions.
- **Quantization-quality tradeoffs.** Q4 changes logits and requires task-level quality testing.
- **Model scope.** The converter targets dense Qwen2/Qwen2.5 and text-only Qwen3-MoE safetensors, not arbitrary architectures or multimodal variants.
- **Managed budget is not total RAM.** Future reports should capture complete process and system memory with operating-system counters.

The next practical work is to build an automated benchmark harness, record cold and warm trials, compare against llama.cpp/Ollama, and profile prefill at layer and expert granularity. If the project is submitted as an academic systems paper rather than a technical article, the evaluation should also include multiple SSDs and CPUs, model sizes, context lengths, batch sizes, ablation studies, energy measurements, and response-quality tests.

## 11. Conclusion

TokenQL demonstrates that a laptop does not have to hold an entire 30.05 GiB Q4 MoE checkpoint in application-managed RAM to execute it. By keeping model packs on SSD, caching recurring compressed experts, persisting KV state, using asynchronous exact reads, and executing packed weights with native CPU kernels, the runtime moved from approximately 0.03 tok/s to sustained decode above 1 tok/s, with a measured warm-turn peak of 1.54 tok/s.

The most important conclusion is not the peak number. The project shows where bounded-memory inference actually spends time and how easily a plausible optimization can fail. Prediction accuracy was not enough when predicted experts were already cached. Fewer I/O calls were not enough when coalescing increased copying and byte traffic. Conversely, removing repeated path resolution and file reopening—small orchestration details—made decisive improvements.

This is not yet a replacement for mature inference engines, and it does not make SSD equivalent to RAM. It is a functioning, inspectable foundation for studying how sparse language models can trade memory capacity for storage traffic on ordinary hardware.

---

## References

[^1]: Qwen Team, [Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B) and [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
[^2]: Hugging Face, [Safetensors documentation](https://huggingface.co/docs/safetensors/main/index).
[^3]: Ying Sheng et al., [FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](https://arxiv.org/abs/2303.06865).
[^4]: Samyam Rajbhandari et al., [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596).
[^5]: ggml-org, [llama.cpp](https://github.com/ggml-org/llama.cpp) and its [server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
[^6]: Woosuk Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
