# Uncensoring Project Maya (GLM-5.3-Flash / Maya S24)

> A complete guide on removing refusal directions from **GLM-5.3-Flash** when using the **Project Maya** inference engine (including quants such as `Maya-S24`, `Maya-S`, `Maya-M`, etc.) using **in-flight directional refusal ablation (orthogonal projection)** without modifying or re-quantizing the weights on disk.

---

## 📑 Table of Contents

- [Why Not Re-Quantize?](#why-not-re-quantize)
- [How In-Flight Ablation Works](#how-in-flight-ablation-works)
- [C++ / CUDA Kernel Overview](#c--cuda-kernel-overview)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Download the Refusal Vector](#1-download-the-refusal-vector)
  - [2. Launch Command](#2-launch-command)
  - [3. (Optional) Launching with a JSON Configuration File](#3-optional-launching-with-a-json-configuration-file)
- [Verification & Expected Logs](#verification--expected-logs)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Credits & References](#credits--references)

---

## ⚡ Why Not Re-Quantize?

Uncensoring models typically requires fine-tuning or applying abliteration scripts directly to weight checkpoints and re-quantizing them. For a 321B MoE architecture like **GLM-5.3-Flash**:

1. **Massive Size**: Checkpoints are ~90–150 GB in quantized formats. Downloading or storing multiple duplicates wastes storage.
2. **Quantization Degradation**: Converting or re-quantizing non-standard quantized layers can introduce perplexity degradation and break Maya's custom quantized format layouts.
3. **Loss of Fast Paths**: Maya relies on tightly calibrated expert tiering across VRAM, RAM, and NVMe SSD. Incompatible quantizations break its native kernel optimizations.

By removing refusal directions **in-flight** through residual stream orthogonal projection, we keep the original high-performance Maya quant untouched while achieving clean, uncensored responses.

---

## 🧠 How In-Flight Ablation Works

During forward passes through transformer blocks, refusal signals live within a low-dimensional subspace of the residual stream activations. 

Project Maya implements orthogonal projection directly after the token mixer and FFN computations:

$$\mathbf{x}' = \mathbf{x} - \text{scale} \cdot (\mathbf{x} \cdot \mathbf{v}) \mathbf{v}$$

Where:
- $\mathbf{x}$ is the residual stream activation vector for a token.
- $\mathbf{v}$ is a unit vector representing the refusal direction for that layer.
- $\text{scale}$ controls the projection strength (`1.0` removes 100% of the refusal component).

Because residual stream writes are linear combinations, projecting the directions out of the block outputs ensures every subsequent state remains orthogonal to the refusal direction.

---

## 💻 C++ / CUDA Kernel Overview

In Project Maya, the ablation logic is implemented in `src/core/glm_ablate.cuh`:

```cuda
namespace {
__global__ void rows_kernel(float* __restrict__ X, const float* __restrict__ V, int E, int K, float scale) {
    float* x = X + (size_t) blockIdx.x * E;
    __shared__ float red[256];
    const int tid = threadIdx.x;
    for (int k = 0; k < K; ++k) {
        const float* v = V + (size_t) k * E;
        float acc = 0.0f;
        for (int e = tid; e < E; e += 256) acc += x[e] * v[e];
        red[tid] = acc;
        __syncthreads();
        for (int n = 128; n > 0; n >>= 1) {
            if (tid < n) red[tid] += red[tid + n];
            __syncthreads();
        }
        const float c = red[0] * scale;
        __syncthreads();
        for (int e = tid; e < E; e += 256) x[e] -= c * v[e];
        __syncthreads();
    }
}
}
```

The operation executes in GPU shared memory with virtually zero compute overhead.

---

## 🚀 Step-by-Step Guide

### 1. Download the Refusal Vector

A 45-layer calibrated refusal direction vector is available via Hugging Face (`alesha-pro`):

```bash
mkdir -p /models/Maya-S24/vector
cd /models/Maya-S24/vector

curl -L -o glm53-refusal-per-block-3dir.layers45.f32 \
  https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-Maya-M-GGUF/resolve/main/vector/glm53-refusal-per-block-3dir.layers45.f32
```

Verify the file integrity:
```bash
ls -lh glm53-refusal-per-block-3dir.layers45.f32
# Size must be exactly 2,211,840 bytes (552,960 float32 values: 45 layers × 3 directions × embedding dimension)
```

---

### 2. Launch Command

Set the environment variables and run Project Maya directly:

```bash
# 1. Enable in-flight refusal ablation
export STRATA_GLM_ABLATE="/models/Maya-S24/vector/glm53-refusal-per-block-3dir.layers45.f32"
export STRATA_GLM_ABLATE_LAYERS="45"
export STRATA_GLM_ABLATE_SCALE="1.0"

# 2. Launch Project Maya server
./maya.sh --port 8080
```

> **One-liner command:**
> ```bash
> STRATA_GLM_ABLATE="/models/Maya-S24/vector/glm53-refusal-per-block-3dir.layers45.f32" \
> STRATA_GLM_ABLATE_LAYERS="45" \
> STRATA_GLM_ABLATE_SCALE="1.0" \
> ./maya.sh --port 8080
> ```

---

### 3. (Optional) Launching with a JSON Configuration File

If you use a customized configuration file (e.g. `strata_config_uncensored.json`) for custom layer splits or dual-GPU setups:

```bash
# Set your environment variables
export STRATA_GLM_ABLATE="/models/Maya-S24/vector/glm53-refusal-per-block-3dir.layers45.f32"
export STRATA_GLM_ABLATE_LAYERS="45"
export STRATA_GLM_ABLATE_SCALE="1.0"

# Launch the server with the custom configuration
.venv/bin/python3 -m serve.server \
  --engine strata \
  --fit-max-tokens \
  --config strata_config_uncensored.json \
  --host 0.0.0.0 \
  --port 8080
```

Or define the environment variables directly inside `strata_config_uncensored.json`:

```json
{
  "exe": "/home/maxbozza/project-maya/build/strata",
  "cwd": "/home/maxbozza/project-maya",
  "args": [
    "--glm-pack",
    "/models/Maya-S24/pack",
    "--max-context",
    "131072",
    "--kv",
    "int8",
    "--kv-streaming",
    "on",
    "--layer-split",
    "20",
    "--expert-cache-per-layer",
    "--vram-reserve-mib",
    "512",
    "--vram-reserve-later-mib",
    "640"
  ],
  "tokenizer": "/models/Maya-S24/pack/tokenizer",
  "model_name": "Maya-S24-Uncensored",
  "host": "0.0.0.0",
  "port": 8080,
  "gpu": "0,1",
  "env": {
    "STRATA_GLM_RAM_RESIDENT": "1",
    "STRATA_GLM_LAT8": "1",
    "STRATA_GLM_SPEC_CANCEL": "1",
    "STRATA_GLM_SPEC_SAMPLE": "1",
    "STRATA_GLM_MTP_KEEP": "1",
    "STRATA_GLM_TRUNK_KEEP_MAP": "8888888888888888888855555555555555555555555558",
    "STRATA_GLM_CPU_LANE0": "4",
    "STRATA_GLM_CPU_LANE1": "0",
    "STRATA_GLM_ABLATE": "/models/Maya-S24/vector/glm53-refusal-per-block-3dir.layers45.f32",
    "STRATA_GLM_ABLATE_LAYERS": "45",
    "STRATA_GLM_ABLATE_SCALE": "1.0"
  }
}
```

---

## 🔍 Verification & Expected Logs

When starting the engine, inspect the console output:

```text
glm ablate: 552960 values from /models/Maya-S24/vector/glm53-refusal-per-block-3dir.layers45.f32 (norm^2 135.0000), per layer, scale 1.00
```

- When `glm ablate: ...` appears, the kernel is active and applying the orthogonal projection to all residual stream writes.
- If you see `glm ablate: cannot open ... - running the stock model`, check the vector file path and permissions.

---

## ❓ Troubleshooting & FAQ

#### Does this degrade code generation or mathematical reasoning?
No. Because refusal vectors are isolated along a tiny orthogonal subspace (3 directions across 45 blocks), standard capabilities, tool-calling, and reasoning remain intact.

#### Can I adjust the intensity of the uncensoring?
Yes. Adjust `STRATA_GLM_ABLATE_SCALE`:
- `0.0`: Stock model behavior (no ablation).
- `0.5`: Mild suppression of preachy disclaimers.
- `1.0`: Standard complete ablation.

---

## 📜 Credits & References

- [Project Maya](https://github.com/mw00/project-maya) by mw00 and PeasantSmith.
- [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.ai.
- Refusal direction extraction & vector by [alesha-pro](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-Maya-M-GGUF).
