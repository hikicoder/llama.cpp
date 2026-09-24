# dsv41-ada-medvram: faster prefill for DeepSeek-V4.1-Flash on a ~20 GB GPU

This branch is a fork of [JigSawPT/llama.cpp](https://github.com/JigSawPT/llama.cpp), branch
`dsv41-porte`, which runs DeepSeek-V4.1-Flash (552B MoE) by streaming its experts from NVMe. All
of the model port, the streaming engine and the host-RAM tier are JigSawPT's work (see
[README-dsv41.md](README-dsv41.md)). This fork changes how **prompt processing** streams experts,
so that a machine with a mid-size GPU can prefill at a usable speed.

The original llama.cpp README follows [further below](#llamacpp).

## Results in short

| | before | after |
|---|---|---|
| prompt processing, real prompt | 0.2-0.45 tokens/s in use (0.9-2.1 in short 512-token tests) | 14.4 tokens/s over a 7130-token prompt (3 GB/s cap, 70 C target); 23.9 tokens/s on its first 2048 tokens |
| time to first token | ~1.5 h in use (at those rates, a prompt of about 1,100-2,400 tokens) | 3 min 36 s for a 2410-token prompt (uncapped, drive throttling); 8 min 15 s for a 7130-token prompt |
| a 7130-token prompt | 4.4-9.9 h at 0.2-0.45 tokens/s | 8 min 15 s |
| generation | 0.9 tokens/s | 1.90-2.01 tokens/s from the third turn on (1.26 and 1.36 in the first two) |
| drive throughput during prefill | ~1 GB/s | 7.3-7.4 GB/s (the drive's maximum) uncapped, until it overheated; ~3.2 GB/s under a 3 GB/s cap |
| drive temperature over a 57-minute session | - | peak 71.8 C with a 3 GB/s cap and a 70 C target; uncapped it had throttled at 74 C |
| ubatch used | 128 (512 did not fit) | 2048 |
| cancelling a request mid-prompt | the slot stayed busy until the batch finished | the slot was free 23 ms after the server received the cancel |

Test machine: a 20 GB Ampere workstation GPU (about 17 GB usable next to the desktop), a 12-core
desktop CPU, 128 GB RAM, and the GGUF on a PCIe 4.0 x4 NVMe with a passive heatsink and no airflow.
The model file is 502 GB; its routed experts are 289 GB (40 layers x 384 experts x 3 projections of
6.27 MB).

## Why prefill was slow

Prefill ran at 0.2-0.45 tokens/s in real use (0.9-2.1 in short 512-token tests), about 1.5 hours to
the first token, while the drive, the CPU and the GPU all looked idle.

**The drive was not the bottleneck.** A read benchmark on the model file showed that scattered 6 MiB
`O_DIRECT` reads, the size of one expert projection, reach 7.4 GB/s once two or more are in flight.
That is the same as 32 MiB sequential reads. Merging or sorting reads was not going to help.

| read pattern (O_DIRECT) | threads | throughput |
|---|---|---|
| random 6 MiB | 1 | 5.3 GB/s |
| random 6 MiB | 2, 4 or 9 | 7.4 GB/s |
| 18 random 6 MiB reads (one wave's worth) | 9 | 6.8 GB/s, done in 16 ms |
| sequential 32 MiB | 1 / 2 | 6.8 / 7.4 GB/s |

**The wave pipeline was.** When a ubatch touches more experts than the GPU cache holds, the upstream
streaming code runs the expert GEMMs in "waves" of 6 experts. Each wave:

1. waits for its own 18 reads (6 experts x 3 projections), while at most the next wave's 18 are
   preloaded,
2. uploads them one at a time to the GPU,
3. syncs the GPU and runs the expert GEMM over **every** token of the ubatch, masking out all pairs
   except those 6 experts,
4. only then starts the next wave.

So the drive never had more than one or two waves of reads queued, and sat idle during every upload,
sync and GEMM. For scale: a 362-token ubatch touches 130-237 of a layer's 384 experts (measured),
which at 6 per wave is 22-40 waves per layer and about 1,200 per ubatch. The drive averaged about
1 GB/s.

**The ubatch was stuck at 128.** A ubatch streams a large share of the 289 GB of experts however few
tokens it carries: on average 47% of each layer's experts at 362 tokens and 79% at 2044 tokens
(measured). So bigger ubatches are the other large lever. But `-ub 512` asked for 8721 MiB of CUDA
scratch. The culprit was attention in layers 20-39: flash attention falls back to
the CPU for this model and gets disabled, so the score matrix is materialized as n_kv x n_tokens x
64 heads in F32, which is 4192 MiB per layer at 32k context and 512 tokens, plus as much again for
its softmax.

## What was changed

- **One pass per layer instead of waves** (`src/llama-moe-stream.cpp`, `src/llama-graph.cpp`). The
  GPU expert caches of all 40 layers are allocated as one pool; each layer's decode cache is a slice
  of it. During prefill, the layer being computed borrows the pool as a region with one slot per
  expert. All experts the ubatch routes to are queued at once in file order, land in RAM (the
  host tier, or pinned staging), upload to their slots, and the expert GEMM runs **once** with the
  real expert ids. Decode still uses the small per-layer caches; those that share slots with the
  region (32 of 40 layers at 12 slots) are dropped by each prefill pass and refill on demand.
  `LLAMA_MOE_STREAM_SWEEP=0` restores the wave path.
- **Attention in query chunks** (`src/llama-graph.cpp`). Non-flash attention processes queries in
  chunks so one score matrix stays under `LLAMA_ATTN_CHUNK_MIB` (default 1024). Query rows are
  independent, so this is mathematically the same computation with a lower peak memory.
- **Cancel that works mid-prompt** (`tools/server/`). A client cancel sets off the context's abort
  callback, the streaming waits poll it, the decode returns early, and the server clears the slot's
  partially processed prompt so a later request cannot reuse tokens that never reached the cache.
- **Drive heat controls** (off by default). `--moe-stream-read-max <GB/s>` caps drive reads;
  `--moe-stream-temp-max <C>` lowers the cap further to hold a temperature, read from the drive's
  hwmon sensor, with a slope term so it starts slowing before the target rather than after.
  Pacing works in half-second slices (full speed, then idle), because spacing individual reads
  evenly (one every ~2 ms at 3 GB/s) made the power stages whine audibly.
- **Smaller changes.** The host tier is pinned in 1 GiB chunks (a refused pin used to silently turn
  the whole tier pageable). Prefill fills take only free host-tier slots, so one pass over every
  layer does not evict the part of the tier that the next ubatch will hit. The minimum GPU cache
  drops from 18 to 12 slots per layer, which frees VRAM for the larger ubatch.
  `LLAMA_MOE_STREAM_SWEEP_LOG=1` prints one line per layer per ubatch.

## Results observed

**VRAM.** CUDA scratch at 32k context, measured with `llama-fit-params`:

| `-ub` | before | after |
|---|---|---|
| 128 | 1338 MiB | 1338 MiB |
| 512 | 8721 MiB | 2267 MiB |
| 1024 | - | 2675 MiB |
| 2048 | - | 4340 MiB |

With `-ub 2048`, a 12-slot cache (8606 MiB) fits; 14 slots (10041 MiB) ran out of memory in the CUDA
pool that cuBLAS matmuls allocate from, which the estimate does not count.

**Prefill.** A 2410-token prompt made of public llama.cpp documentation, `-ub 2048`, 32 GiB host tier.
The server splits such a prompt into 362 + 2044 + 4 tokens (upstream checkpoint logic).

- The 362-token part took 20.1 s (18.0 tokens/s). Every layer's reads ran at 7.3-7.4 GB/s,
  0.32-0.59 s per layer, with 55 ms between layers on average.
- In a run with temperature logging, the drive went from 52 C to 71 C within those 20 s. The
  2044-token part started at full speed; reads were still at 7.26 GB/s 28 s into the prompt, fell
  to 3.8 GB/s at 33 s, and were down to 0.48 GB/s at 61 s, about 15x slower, with the drive at 74 C.
- The whole prompt took 216 s (11.1 tokens/s) and read 336 GB from the drive, uncapped, with the
  drive throttling for most of it. With the drive kept cool, the same prompt should take about 60 s
  (~40 tokens/s); that is an estimate, not a measurement.

**Heat.** With a passive heatsink and no airflow, the drive heats about 1 C per second at full
speed and cools about 4 C per minute when idle. The first temperature controller reacted to the
reading alone and overshot: it engaged at 68.8 C and the drive still reached 75 C. The current one
projects the slope 6 s ahead and paces in half-second slices.

**Coil whine.** The first pacing version spaced single reads evenly, about one every 2 ms at a
3 GB/s cap, and the machine's power stages whined and crackled audibly while it ran. Pacing in
half-second slices (full speed, then idle) removed the noise completely, at the same average rate.

**A long session with a cap.** `--moe-stream-read-max 3 --moe-stream-temp-max 70`, `-ub 2048`,
32 GiB host tier, one chat that grew from 7.1k to 10.3k tokens of context over 57 minutes:

| request | new prompt tokens | prompt time | prompt speed | generated | generation speed |
|---|---|---|---|---|---|
| 1 | 7130 | 495 s | 14.4 tokens/s | 166 | 1.26 tokens/s |
| 2 | 59 | 17.6 s | 3.4 tokens/s | 197 | 1.36 tokens/s |
| 3 | 98 | 23.7 s | 4.1 tokens/s | 214 | 2.01 tokens/s |
| 4 | 167 | 29.3 s | 5.7 tokens/s | 318 | 1.95 tokens/s |
| 5 | 150 | 27.5 s | 5.5 tokens/s | 263 | 1.93 tokens/s |
| 6 | 67 | 18.2 s | 3.7 tokens/s | 292 | 1.96 tokens/s |
| 7 | 179 | 29.2 s | 6.1 tokens/s | 237 | 1.96 tokens/s |
| 8 | 65 | 18.9 s | 3.4 tokens/s | 260 | 1.90 tokens/s |
| 9 | 100 | 22.3 s | 4.5 tokens/s | 309 | 1.90 tokens/s |

- The drive peaked at 70.8 C during the 7130-token prompt and at 71.8 C over the whole session,
  below the 74 C at which it had throttled uncapped. Pacing engaged 34 times and let go each time,
  the first time 39 s into the first prompt.
- The first 2048 tokens ran at 23.9 tokens/s, almost entirely at the 3 GB/s cap. The temperature
  limit then held the rest of the prompt lower: the whole 7130-token prompt read 737 GB at an average
  of 1.58 GB/s and ran at 14.4 tokens/s.
- Follow-up turns only process their new tokens (the rest is cached), in two passes over the layers
  (the new tokens, then the last 4). Each turn read 52-85 GB from the drive at about 3.2 GB/s, so
  59-179 new tokens took 18-29 s. This is the fixed cost of a turn; faster reads shorten it directly.
- Generation ran at 1.90-2.01 tokens/s from the third turn on, about twice the 0.9 tokens/s seen on
  the same machine before these changes; the first two turns generated at 1.26 and 1.36 tokens/s.
- At the 0.2-0.45 tokens/s seen before these changes, a prompt of this size would have needed 4.4 to
  9.9 hours before the first token. Here it came after 8 min 15 s, 32-72x faster.
- Over the session, prefill read 1282 GB from the drive, averaging 2.0 GB/s while it waited on it,
  and took 20% of its slabs from the host tier instead (51,725 of 256,302).

**Cancel.** Killing the client in the middle of a 2044-token batch aborted the decode: the server
logged the cancel, the running layer's wait stopped 20 ms later, and the slot was free 23 ms after
the cancel. Before, the slot stayed busy until the whole batch had been read.

**Not verified yet:** outputs are not compared token-for-token against the wave path. Answers were
coherent over a long session.

## Running it

Build as usual (Linux, CUDA):

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

Example for a ~20 GB GPU, GGUF on NVMe:

```
LLAMA_MOE_STREAM_SWEEP_LOG=1 build/bin/llama-server -m DeepSeek-V4.1-Flash-MXFP4-engram.gguf \
  -ngl 99 -c 32768 -nkvo -b 2048 -ub 2048 --parallel 1 --moe-stream-io-threads 9 \
  -ot 'attn_output=CPU,attn_q_b=CPU,token_embd=CPU,output=CPU' \
  --moe-stream --moe-stream-cache 12s --moe-stream-l2 32 --moe-stream-direct \
  --moe-stream-read-max 3 --moe-stream-temp-max 70 --reasoning off
```

| option | meaning |
|---|---|
| `--moe-stream-cache 12s` | GPU expert slots per layer; the pool must hold one layer's 384 experts (40 x 12 = 480) |
| `-b 2048 -ub 2048` | each pass over the experts serves 2048 tokens |
| `--moe-stream-read-max N` | cap drive reads at N GB/s (0 = off) |
| `--moe-stream-temp-max C` | hold the drive at C degrees Celsius (0 = off); sensor override: `LLAMA_MOE_STREAM_TEMP_SENSOR` |
| `LLAMA_MOE_STREAM_SWEEP_LOG=1` | one line per layer: experts, wait, GB/s, drive temperature |
| `LLAMA_ATTN_CHUNK_MIB` | cap on one attention score matrix, MiB (default 1024, 0 = never chunk) |
| `LLAMA_MOE_STREAM_SWEEP=0` | use the old wave path |

## Credits

- [JigSawPT](https://github.com/JigSawPT/llama.cpp) for the `dsv41-porte` branch this is built on:
  the DeepSeek-V4.1 port, the engram tables, the host tier and the measurements behind them.
- nibor1896 for [Crow](https://github.com/nibor1896/Crow), whose `moe-stream` patch series is the
  base of the streaming path.
- DeepSeek for the model; [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for everything
  underneath.
- The changes in this fork were made with AI assistance (Claude, via Cursor).

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev branches](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-features.md) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
