# train-my-model (`tmm`)

A hardware-agnostic, plugin-extensible CLI for training deep learning models through the [DLPack](https://dmlc.github.io/dlpack/latest/) tensor ABI. Models compiled with [Apache TVM](https://tvm.apache.org/), IREE, or any framework that emits a C-ABI shared library work out of the box.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Plugin System](#plugin-system)
- [Built-in Plugins](#built-in-plugins)
- [Environment Variables](#environment-variables)
- [Building from Source](#building-from-source)

---

## Overview

`tmm` handles the training machinery so your model only needs to implement a forward+backward function:

- **Dataset loading** — Arrow IPC and Parquet files via URI-addressed sources (`hf:`, `gh:`, `file:`, …)
- **Gradient accumulation** and optimizer step scheduling
- **LR scheduling** — constant, step, linear, cosine, cosine-with-warmup
- **Plugin-driven extensibility** — WASM and native shared-library plugins for custom data sources, model loaders, LR schedulers, optimizers, callbacks, and loggers
- **Interactive console UI** — live loss curves and metrics via FTXUI

`tmm` is **tensor-library agnostic**: it communicates with models exclusively through `DLTensor` / `DLPack`, so any framework that can produce a compatible shared library works without modification.

---

## Installation

### Pre-built packages

Download the latest release for your platform from the [Releases](../../releases) page.
Each release ships a Debian package (`.deb`), a macOS disk image (`.dmg`), a Windows NSIS installer (`.exe`), and plain archives (`.tar.gz` / `.zip`).

```sh
# Debian / Ubuntu
sudo dpkg -i tmm-<version>-Linux.deb

# macOS — open the .dmg and drag tmm to /usr/local/bin

# Windows — run the NSIS installer, or extract the ZIP next to a directory on PATH
```

Plugins are searched in the same directory as the `tmm` binary by default, so
installing via the package puts everything in the right place automatically.

---

## Quick Start

```sh
# 1. Load plugins (core provides hf: data source and TVM model loader)
# 2. Write a config file

cat > train.yml <<'EOF'
data:
  train:
    url: hf:owner/my-dataset
    split: train
    batch_size: 32
  validation:
    url: hf:owner/my-dataset
    split: validation

model:
  file: ./model.so      # path relative to this config file
  device: cpu

trainer:
  epochs: 10
  optimizer:
    type: adamw
    lr: 3.0e-4
  lr_scheduler:
    type: cosine_warmup
    warmup_steps: 100

plugins:
  - name: core          # provides hf: source + TVM loader + schedulers
EOF

# 3. Train
tmm fit train.yml
```

---

## Usage

```
tmm fit      <config.yml> [config2.yml …] [--set key=value …]
tmm validate <config.yml> [config2.yml …] [--set key=value …]
```

### Multiple config files

When more than one file is provided they are deep-merged left-to-right (later files win on scalar collisions, maps are merged recursively):

```sh
tmm fit model.yml trainer.yml

# Override a single value at the CLI
tmm fit model.yml trainer.yml --set trainer.optimizer.lr=5e-4
```

---

## Configuration Reference

All configuration files are YAML. Two schemas are accepted: a **new schema** (recommended) and a **legacy schema** (backward-compatible).

### New schema (recommended)

```yaml
# Schema version — optional, defaults to "1"
version: "1"

# ── Data ────────────────────────────────────────────────────────────────────
data:
  train:
    url: hf:owner/dataset       # URI — scheme provided by a plugin (hf:, gh:, file:, …)
    split: train
    batch_size: 32
    shuffle: true
    shuffle_buffer_size: 10000
    num_workers: 4
    prefetch: 2
    preprocessor:               # optional — applied in order before collation
      - type: bpe-tokenize
        config: '{"vocab":"vocab.json","max_length":512}'

  validation:                   # optional
    url: hf:owner/dataset
    split: validation
    batch_size: 32

# ── Model ────────────────────────────────────────────────────────────────────
model:
  file: ./model.so              # path resolved relative to this config file
  # path: /absolute/path        # alternative: absolute / build-relative path
  function: main                # exported function name; default: "main"
  device: cpu                   # cpu | cuda | cuda:1 | metal | vulkan | opencl | rocm
  device_id: 0

# ── Trainer ──────────────────────────────────────────────────────────────────
trainer:
  epochs: 10
  gradient_accumulation_steps: 1

  optimizer:
    type: adamw                 # provided by core plugin; any plugin can add more
    lr: 3.0e-4
    weight_decay: 1.0e-2
    beta1: 0.9
    beta2: 0.999
    eps: 1.0e-8
    amsgrad: false

  lr_scheduler:
    type: cosine_warmup         # constant | step | linear | cosine | cosine_warmup
    warmup_steps: 100
    min_lr: 0.0
    step_size: 1                # step scheduler only
    gamma: 0.1                  # step scheduler only
    total_steps: 0              # 0 = infer from epochs × batches

  early_stopping:               # optional; injects an early-stopping callback
    monitor: val_loss
    patience: 5
    mode: min
    min_delta: 0.0

  checkpoint:                   # optional; injects a checkpoint callback
    directory: ./checkpoints
    every_n_epochs: 1
    keep_top_k: 3
    monitor: val_loss
    mode: min

  callbacks:                    # explicit callback list (takes priority)
    - type: early_stopping
      config: '{"monitor":"val_loss","patience":5,"mode":"min"}'
    - type: core::checkpoint
      config: '{"directory":"./checkpoints","every_n_epochs":1}'

# ── Plugins ──────────────────────────────────────────────────────────────────
plugins:
  - name: core                  # load by name — searched next to binary or TMM_PLUGIN_PATH
  - name: python                # Python model loader (requires Python 3 at runtime)
  - name: console-ui            # FTXUI terminal dashboard
    config: '{"interactive":true}'
  - path: ./my_plugin.wasm      # load by explicit path
    config: '{"key":"value"}'
    optional: true              # warn instead of fail if the file is missing
```

### Legacy schema

The old schema is still accepted as a fallback when no `data:` or `trainer:` key is present:

```yaml
dataset:
  uri: hf:owner/dataset
  config: causality detection   # HuggingFace named config
  split: train
  batch_size: 32

validation:
  uri: hf:owner/dataset
  split: validation

model:
  path: ./model.so
  function: main
  device: cpu

optimizer:
  type: adamw
  lr: 3.0e-4

scheduler:
  type: cosine_warmup
  warmup_steps: 100

training:
  epochs: 10
  gradient_accumulation_steps: 4
  grad_clip_norm: 1.0
  seed: 42

plugins:
  - name: core
```

---

## Plugin System

Plugins extend `tmm` with custom data sources, model loaders, LR schedulers, optimizers, callbacks, and loggers. Two plugin formats are supported:

- **WebAssembly (`.wasm`)** — portable, sandboxed, runs on every platform without recompilation.
- **Native shared library (`.so` / `.dylib` / `.dll`)** — full OS access, used for plugins that need OS threads or GPU access (e.g. `console-ui`).

### Plugin search

`tmm` searches for a plugin named `"core"` in this order:

1. Same directory as the `tmm` binary (works without any configuration after a package install).
2. Each directory listed in `TMM_PLUGIN_PATH` (colon-separated on Unix, semicolon-separated on Windows).

Within each directory, both `tmm_<name>.so` (or `.dll`/`.dylib`) and `tmm_<name>.wasm` are tried. A native plugin is preferred over a WASM plugin when both exist.

Alternatively, specify an absolute path in the config with `plugins: [{path: /path/to/plugin.wasm}]`.

### Writing a plugin

Any language that compiles to WebAssembly or produces a native shared library with C linkage can be used (C, C++, Rust, Zig, …).

A plugin must export three C functions:

```c
#include <tmm/plugins/abi.h>

// Required — return metadata; called before tmm_plugin_init.
tmm_plugin_info* tmm_plugin_get_info(void);

// Required — register capabilities; called once after loading.
tmm_error tmm_plugin_init(const tmm_host_api* host, const char* cfg, uint32_t cfg_len);

// Optional — free resources; called on unload.
void tmm_plugin_teardown(void);
```

`tmm_plugin_init` receives the host API and the JSON config from the `config:` field in the YAML. Use `host->register_source`, `host->register_scheduler`, etc. to publish capabilities to the runtime.

**Lifecycle hooks** (all optional — return non-zero from epoch_end to trigger early stop):

| Export | Signature | Called when |
|--------|-----------|-------------|
| `tmm_on_fit_begin`       | `(ctx_json, ctx_len)`                          | Training starts |
| `tmm_on_epoch_begin`     | `(epoch, total_epochs)`                        | Each epoch starts |
| `tmm_on_batch_begin`     | `(batch, total_batches)`                       | Each batch starts |
| `tmm_on_loss_computed`   | `(loss_f32) -> float`                          | After loss; return value replaces loss |
| `tmm_on_batch_end`       | `(batch, loss_f32, metrics_json, metrics_len)` | After optimizer step |
| `tmm_on_epoch_end`       | `(epoch, metrics_json, metrics_len) -> int32`  | After epoch; non-zero = stop |
| `tmm_on_validation_end`  | `(metrics_json, metrics_len)`                  | After validation pass |
| `tmm_on_model_loaded`    | `(info_json, info_len)`                        | After model is loaded |
| `tmm_on_metric`          | `(name, name_len, value_f32, step_i32)`        | Each metric logged |
| `tmm_on_log`             | `(level, msg, msg_len)`                        | Each log message |
| `tmm_on_fit_end`         | `(metrics_json, metrics_len)`                  | Training ends |

**Minimal C example:**

```c
#include <tmm/plugins/abi.h>
#include <stdio.h>

static tmm_plugin_info g_info = { TMM_ABI_VERSION, "my-plugin", "0.1.0", "" };

tmm_plugin_info* tmm_plugin_get_info(void) { return &g_info; }

tmm_error tmm_plugin_init(const tmm_host_api* host, const char* cfg, uint32_t len) {
    (void)host; (void)cfg; (void)len;
    return TMM_OK;
}

int32_t tmm_on_epoch_end(uint32_t epoch, const char* json, uint32_t len) {
    printf("epoch %u done\n", epoch);
    return 0; // non-zero triggers early stop
}
```

---

## Built-in Plugins

| Plugin | Format | Description |
|--------|--------|-------------|
| `core` | Native | Dataset sources: `hf:` (HuggingFace), `gh:` (GitHub), `gl:` (GitLab), `bb:` (Bitbucket), `sr:` (generic git). TVM model loader. LR schedulers: `constant`, `step`, `linear`, `cosine`, `cosine_warmup`. Callbacks: `early_stopping`, `checkpoint`. Loggers: `tensorboard`, `wandb`. |
| `python` | Native | Python model loader (requires Python 3 at runtime; PyTorch optional for full support). |
| `console-ui` | Native + WASM | FTXUI terminal dashboard with live loss curves and metrics. Native build supports interactive mode (`"interactive":true`); WASM build renders ANSI output on each event. |
| `nlp-tasks` | WASM | NLP task metadata and preprocessing utilities. |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TMM_PLUGIN_PATH` | Colon-separated (Unix) or semicolon-separated (Windows) list of directories searched for plugins by name. Searched after the directory containing the `tmm` binary. |

---

## Building from Source

**Requirements:** CMake ≥ 3.24, a C++23-capable compiler, Git.

```sh
git clone https://github.com/your-org/train-my-model.git
cd train-my-model
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `TMM_BUILD_TESTS` | `ON` | Build the Catch2 test suite |
| `TMM_BUILD_EXTENSIONS` | `OFF` | Build the bundled plugins (`core`, `python`, `console-ui`, `nlp-tasks`) |
| `TMM_BUILD_PACKAGE` | `OFF` | Configure CPack to produce a distributable package (DEB / DMG / NSIS+ZIP) |
| `TMM_WASM_TOOLCHAIN` | _(auto)_ | Path to the WASI SDK CMake toolchain file. When empty and `TMM_BUILD_EXTENSIONS=ON`, the SDK is fetched automatically on Linux/macOS. |
| `TMM_ENABLE_COVERAGE` | `OFF` | Instrument for gcov/llvm-cov coverage (GCC / Clang only) |

### Building with plugins

```sh
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTMM_BUILD_EXTENSIONS=ON
cmake --build build
```

### Running tests

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DTMM_BUILD_TESTS=ON
cmake --build build --target tmm_tests
ctest --test-dir build --output-on-failure
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
