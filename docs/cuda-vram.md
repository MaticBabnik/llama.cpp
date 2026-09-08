# Estimating CUDA VRAM use

This guide estimates the peak CUDA device memory used by llama.cpp inference. It applies to a particular model, build, GPU, and command line. Treat the result as a capacity-planning estimate, not a promise: CUDA driver allocations, allocator alignment, kernel choices, and model architectures can change the final value.

The most reliable measurement is the startup memory breakdown:

```sh
llama-fit-params -m model.gguf ...
```

The tool projects device memory before loading the normal application. After a model is loaded, llama.cpp also prints a breakdown with `model`, `context`, `compute`, `self`, and `unaccounted` memory. Use that result to calibrate an estimate for the target GPU and build.

## Model weights

This is usually the largest fixed allocation. It contains the GGUF tensors placed on a CUDA device, including the selected transformer layers and normally the output layer.

The exact size is the sum of the allocated offloaded tensors, including backend buffer alignment. The GGUF file size is a useful upper-bound-like starting point only when all its tensors are offloaded; it is not an exact VRAM size.

Size is affected by:

- Model architecture, layer count, hidden dimensions, vocabulary size, and MoE expert tensors.
- GGUF quantization type. Lower-bit quantizations reduce weight memory, but their stored size is not simply parameter count times bits per parameter.
- `--n-gpu-layers` / `--gpu-layers`, tensor overrides, and whether the output layer is offloaded.
- `--split-mode`, `--tensor-split`, and `--device` when using several GPUs.
- LoRA adapters and any additional loaded model, such as a speculative draft model.

In layer split mode, model tensors reside on the device that owns their layer. In tensor split mode, splittable tensors are distributed across participating devices. Small replicated tensors and the main GPU's scratch work make per-device use unequal even with an even split.

## KV cache

The KV cache stores attention keys and values for tokens already processed. For long contexts it is usually the largest allocation after weights.

For a standard transformer attention cache, an unaligned estimate in bytes is:

```text
V_kv ~= N_cells * N_streams * sum_over_cached_layers(
    D_kv_k(layer) * B_k + D_kv_v(layer) * B_v
)
```

Where:

- `N_cells` is the allocated KV capacity, normally based on `--ctx-size`.
- `N_streams` is the number of cache streams used by the architecture.
- `D_kv_k` and `D_kv_v` are the key and value dimensions per token. With grouped-query attention these are based on KV heads, not query heads.
- `B_k` and `B_v` are the stored bytes per element for `--cache-type-k` and `--cache-type-v`.

For F16 or BF16, `B` is normally 2 bytes; for F32 it is 4 bytes. Quantized cache types need their actual ggml row size, including block metadata, rather than a simple fractional byte count. Add backend-buffer alignment and metadata to the result.

The cache capacity is allocated up front, so the active prompt length does not normally reduce its VRAM use. In `llama-server`, concurrent slots share or divide the configured cache pool according to the server configuration; size the pool for the total requested capacity, not one request. `--parallel` can therefore increase the required KV capacity. Recurrent, state-space, hybrid, and special attention architectures can use different memory layouts, so use their reported context allocation instead of this transformer-only expansion.

KV size is affected mainly by context capacity, cached layer count, KV-head dimensions, cache data types, and server concurrency. It is distributed with layers in layer split mode and split across participating devices in tensor split mode.

## Compute and temporary buffers

Compute buffers hold intermediate tensors and backend workspaces while llama.cpp executes a graph. They are temporary, but their maximum size is retained as an allocation and can be substantial during prompt processing.

Their size depends on:

- Physical micro-batch size, `--ubatch-size`, which is a primary prompt-processing memory control.
- Logical batch size, `--batch-size`, sequence count, prompt shape, and whether embeddings, logits, or multimodal inputs are requested.
- Model dimensions, attention type, MoE routing, and encoder or decoder graph structure.
- Flash Attention, CUDA Graphs, MMQ versus cuBLAS kernels, GPU architecture, and CUDA/library versions.
- The active inference phase. Prompt prefill generally has the highest temporary-memory peak; token generation normally uses less.

Do not estimate this category only from parameter count or context size. Measure it with the intended `--batch-size`, `--ubatch-size`, and workload. Reducing `--ubatch-size` is normally the first way to reduce a prefill-only peak.

## Output, logits, and application buffers

Output-related buffers include logits, token and embedding outputs, sampling state, and application-owned input or output staging. They are usually smaller than model weights and KV cache, but can be material for a large vocabulary, many output tokens, large batches, embeddings, or multimodal workloads.

A dense logits tensor, when one is materialized, has an approximate data size of:

```text
V_logits ~= N_logits_rows * N_vocab * B_logits
```

Actual use depends on which logits llama.cpp requests and retains. Add this category explicitly for applications that retain outputs or run extra post-processing on CUDA.

## CUDA runtime, allocator, and graph overhead

Not all device memory is owned by named llama.cpp tensors. CUDA creates a context, loads modules, and may reserve memory through its allocator or CUDA Graph execution. Other processes using the GPU also reduce free memory.

This overhead has no portable fixed size. It depends on the GPU driver, CUDA runtime, linked libraries, build options, kernel path, display usage, and process history. Reserve a measured margin instead of assuming zero. The `self` and `unaccounted` fields in llama.cpp's memory breakdown expose part of this difference.

Unified memory can avoid an immediate CUDA out-of-memory error by falling back to system RAM, but it does not make the working set fit in physical VRAM and can greatly reduce performance. Do not count it as available VRAM capacity.

## Multi-GPU placement

For a multi-GPU estimate, calculate the total first, then estimate the allocation for each device. Weight and KV placement follows the selected split mode, while compute workspaces, replicated tensors, and runtime overhead are device-specific. The main GPU can use more memory for small tensors and temporary results.

Each device must fit its own peak allocation plus a safety margin. Total VRAM across devices is not sufficient when a single device receives an oversized layer, workspace, or fixed overhead.

## Universal peak-VRAM formula

For device `g`, estimate the peak runtime allocation as:

```text
V_peak(g) ~= V_runtime(g)
           + V_weights(g)
           + V_kv(g)
           + max_over_execution_phases(
                 V_compute(phase, g)
               + V_outputs(phase, g)
               + V_graph_and_allocator(phase, g)
             )
           + V_margin(g)
```

Use bytes for every term, then convert with `MiB = bytes / 2^20` or `GiB = bytes / 2^30`. `V_runtime` covers the CUDA context and fixed library allocations; `V_margin` reserves unmeasured driver, allocator, display, and fragmentation use. The `max` is important because prefill and generation use different temporary buffers.

For a single GPU with all layers offloaded, this becomes:

```text
V_peak ~= V_runtime + V_all_model_weights + V_kv
        + max(V_prefill_temporaries, V_decode_temporaries) + V_margin
```

To make the estimate operational, measure one run with the same model, CUDA build, GPU, cache types, context size, batch settings, and split mode. Take the reported fixed and temporary allocations, preserve a safety margin, and re-measure whenever any of those inputs changes.
