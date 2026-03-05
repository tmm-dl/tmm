# Train My Model (tmm)

A hardware-agnostic, plugin-extensible command-line tool for training deep learning models compiled with [Apache TVM](https://tvm.apache.org/). Inspired by the PyTorch Lightning CLI philosophy — configuration-driven, composable, and reproducible.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
  - [Model Config](#model-config)
  - [Trainer Config](#trainer-config)
  - [Dataset Config](#dataset-config)
  - [Sweep Config](#sweep-config)
- [Plugin System](#plugin-system)
  - [Installing Plugins](#installing-plugins)
  - [Writing a Plugin](#writing-a-plugin)
  - [Built-in Plugins](#built-in-plugins)
- [Hyperparameter Sweeps](#hyperparameter-sweeps)
- [TVM FFI Integration](#tvm-ffi-integration)
- [Examples](#examples)
- [Environment Variables](#environment-variables)

---

## Overview

`tmm` trains models that follow the **forward-backward convention**: a compiled TVM function that accepts model parameters and a batch of inputs, and returns both the forward outputs and the gradients for each parameter. `tmm` handles everything else:

- Dataset loading via the [HuggingFace Dataset Card](https://huggingface.co/docs/datasets/dataset_card) specification
- Loss computation and gradient accumulation
- Optimizer state and learning rate scheduling
- Checkpointing and metric logging
- Extensibility via **WebAssembly plugins**
- Hyperparameter sweeps

`tmm` is **tensor-library agnostic**: it communicates with your model exclusively through the [TVM FFI](https://tvm.apache.org/docs/reference/api/doxygen/runtime_8h.html) (`DLTensor` / `DLPack`), so any framework that can produce a TVM-compatible shared library (TVM, IREE, custom MLIR pipelines) works out of the box.

---

## Installation

### Pre-built Binaries

Download the latest release for your platform from the [Releases](../../releases) page:

```sh
# Linux x86_64
curl -L https://github.com/your-org/train-my-model/releases/latest/download/tmm-linux-x86_64.tar.gz | tar xz
sudo mv tmm /usr/local/bin/
```

### Building from Source

**Requirements:** CMake >= 3.25, C++20 compiler, TVM runtime installed.

```sh
git clone --recurse-submodules https://github.com/your-org/train-my-model.git
cd train-my-model
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
sudo cmake --install build
```

Optional CMake flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-DTMM_WITH_WASM=ON` | `ON` | Enable WebAssembly plugin runtime (wasmtime) |
| `-DTMM_WITH_ARROW=ON` | `ON` | Enable Apache Arrow / Parquet dataset loading |
| `-DTMM_BUILD_TESTS=OFF` | `OFF` | Build test suite |

---

## Quick Start

```sh
# 1. Compile your model to a TVM shared library (see TVM FFI Integration section)
#    Result: model.so exports `forward_backward(params..., input) -> (output, grads...)`

# 2. Create a config file
cat > train.yml <<'EOF'
model:
  module: ./model.so
  function: forward_backward
  parameters:
    - name: fc1_weight
      shape: [784, 256]
      dtype: float32
    - name: fc1_bias
      shape: [256]
      dtype: float32

dataset:
  path: ./my-dataset          # directory with dataset_info.json
  task: image-classification
  split: train
  batch_size: 64

loss:
  type: cross_entropy

trainer:
  epochs: 20
  optimizer:
    type: adam
    learning_rate: 3.0e-4
  lr_scheduler:
    type: cosine_annealing
  gradient_accumulation_steps: 4
  checkpoint:
    every_n_epochs: 5
    output_dir: ./checkpoints
EOF

# 3. Train
tmm fit train.yml
```

---

## Usage

```
tmm <command> [options] <config(s)...>

Commands:
  fit       Train a model
  validate  Run validation loop only
  predict   Run inference on a dataset split
  sweep     Run a hyperparameter sweep
  plugin    Manage plugins (list, install, info)

Global options:
  -v, --verbose         Increase log verbosity (repeatable: -vv, -vvv)
  -q, --quiet           Suppress all output except errors
  --log-file PATH       Write logs to file
  --device DEVICE       Override device (e.g. cuda:0, cpu, metal)
  -h, --help            Show help
  --version             Print version
```

### Multiple Config Files

When multiple config files are provided they are merged in order (later files override earlier ones). This allows separating model and trainer concerns:

```sh
tmm fit model.yml trainer.yml

# Override a single value at the command line
tmm fit model.yml trainer.yml --set trainer.optimizer.learning_rate=1e-3
```

### Resuming Training

```sh
tmm fit train.yml --resume ./checkpoints/epoch_10.ckpt
```

---

## Configuration Reference

All configuration files are YAML. Keys marked **required** have no default.

### Model Config

```yaml
model:
  # Path to compiled TVM shared library or TVMScript module.  Required.
  module: ./model.so

  # Name of the exported forward-backward function.  Required.
  # Signature: (param_0, param_1, ..., input) -> (output, grad_0, grad_1, ...)
  function: forward_backward

  # Initial parameter values. Required.
  parameters:
    - name: layer1_weight         # logical name used for checkpointing / logging
      shape: [256, 784]
      dtype: float32
      init:
        type: kaiming_uniform     # kaiming_uniform | xavier_uniform | zeros | ones | constant | file
        # type: file
        # path: ./pretrained/layer1_weight.npy
    - name: layer1_bias
      shape: [256]
      dtype: float32
      init:
        type: zeros

  # Target device for model execution.  Default: cpu
  device: cpu                     # cpu | cuda:<n> | metal | vulkan | opencl
```

### Trainer Config

```yaml
trainer:
  # Number of full passes over the training set.  Default: 10
  epochs: 20

  optimizer:
    # Optimizer algorithm.  Default: sgd
    type: adam                    # sgd | adam | adamw | rmsprop
    learning_rate: 3.0e-4
    weight_decay: 1.0e-5
    # Additional optimizer-specific keys (betas, eps, momentum, …) passed through.
    betas: [0.9, 0.999]

  lr_scheduler:
    type: cosine_annealing        # constant | step | linear | cosine_annealing | warmup_cosine
    warmup_steps: 500             # (warmup_cosine only)
    min_lr: 1.0e-6
    step_size: 10                 # (step only)
    gamma: 0.1                    # (step only)

  # Accumulate gradients over N batches before applying an optimizer step.  Default: 1
  gradient_accumulation_steps: 4

  # Gradient clipping.  Disabled if omitted.
  gradient_clipping:
    max_norm: 1.0
    norm_type: 2.0                # l2 | linf

  checkpoint:
    output_dir: ./checkpoints
    every_n_epochs: 5
    save_top_k: 3                 # Keep only the best k checkpoints by validation loss
    monitor: val_loss

  # Validation frequency.  Default: 1 (every epoch)
  val_every_n_epochs: 1

  # Random seed for reproducibility.  Default: unset (non-deterministic)
  seed: 42

  # Plugins to load.  Paths or names of installed plugins.
  plugins:
    - name: console-ui
    - name: tensorboard
      config:
        log_dir: ./tb_logs
    - path: /opt/tmm/plugins/custom_early_stopping.wasm
      config:
        patience: 5
        monitor: val_loss
```

### Dataset Config

`tmm` expects datasets that follow the [HuggingFace Dataset Card](https://huggingface.co/docs/datasets/dataset_card) specification. The dataset directory must contain a `dataset_info.json` (or a `README.md` with YAML front matter).

```yaml
dataset:
  # Local path or HuggingFace Hub identifier (e.g. "ylecun/mnist")
  path: ./my-dataset

  # Dataset task — determines which features are inputs vs. labels.  Required.
  task: image-classification    # image-classification | text-classification |
                                #   token-classification | image-segmentation |
                                #   regression | question-answering | custom

  split: train                  # train | validation | test | train[:80%]

  # For task: custom — explicitly map feature names to roles.
  features:
    input: image
    label: label

  batch_size: 64

  # Number of data-loading worker threads.  Default: 4
  num_workers: 4

  # Shuffle training data each epoch.  Default: true
  shuffle: true

  # Preprocessing pipeline (applied in order, CPU-side before batching).
  transforms:
    - type: normalize
      mean: [0.485, 0.456, 0.406]
      std:  [0.229, 0.224, 0.225]
    - type: resize
      size: [224, 224]
    - type: random_horizontal_flip   # Training augmentation — ignored during val/test
      p: 0.5

  # Load the dataset lazily (streaming).  Required for datasets too large for RAM.
  streaming: false
```

### Sweep Config

Hyperparameter sweeps are configured in the same YAML file or a separate file passed to `tmm sweep`:

```yaml
sweep:
  # Search strategy.  Default: random
  method: bayesian              # grid | random | bayesian

  # Maximum number of trials.  Default: unlimited (grid finishes naturally)
  max_trials: 50

  # Objective metric to minimize/maximize.
  metric:
    name: val_loss
    goal: minimize              # minimize | maximize

  # Base trainer / model / dataset config files to apply the sweep on top of.
  base_configs:
    - model.yml
    - trainer.yml

  # Parameter search space.
  parameters:
    trainer.optimizer.learning_rate:
      distribution: log_uniform
      min: 1.0e-5
      max: 1.0e-2

    trainer.optimizer.type:
      values: [adam, adamw, sgd]

    trainer.gradient_accumulation_steps:
      values: [1, 2, 4, 8]

    dataset.batch_size:
      values: [32, 64, 128]

  # Run trials in parallel (requires multiple devices or a cluster config).
  parallelism: 1
```

Run a sweep:

```sh
tmm sweep sweep.yml
# or inline with fit configs
tmm sweep model.yml trainer.yml --sweep-config sweep_params.yml
```

---

## Plugin System

`tmm` plugins are **WebAssembly (WASM) binaries**. Because WASM is a portable compilation target, a single `.wasm` file runs identically on Linux, macOS, Windows, and any architecture — no re-compilation needed when distributing plugins.

Plugins hook into the training lifecycle and can:

- Modify training behavior (early stopping, gradient surgery)
- Render interactive UIs (FTXUI-based console dashboard)
- Forward metrics to external services (TensorBoard, Weights & Biases)
- Implement custom learning rate schedules or loss functions

### Installing Plugins

```sh
# Install from a .wasm file
tmm plugin install ./my_plugin.wasm

# List installed plugins
tmm plugin list

# Show plugin metadata and available hooks
tmm plugin info console-ui
```

Plugins are stored in `~/.tmm/plugins/` by default. Override with `$TMM_PLUGIN_DIR`.

### Writing a Plugin

Plugins export a set of well-known functions that `tmm` calls at lifecycle points. Any language that compiles to WASM can be used (C, C++, Rust, AssemblyScript, Zig, …).

**Plugin lifecycle hooks** (all optional):

| Export | Called when |
|--------|-------------|
| `tmm_init(config_json_ptr, config_len) -> i32` | Plugin loaded; receives JSON config from `trainer.plugins[].config` |
| `tmm_on_train_begin(ctx_ptr, ctx_len)` | Training loop starts |
| `tmm_on_epoch_begin(epoch, total_epochs)` | Each epoch starts |
| `tmm_on_batch_begin(batch, total_batches)` | Each batch starts |
| `tmm_on_loss_computed(loss_f32) -> f32` | After loss is computed; return value replaces loss |
| `tmm_on_batch_end(batch, loss_f32, metrics_json_ptr, metrics_json_len)` | After optimizer step |
| `tmm_on_epoch_end(epoch, metrics_json_ptr, metrics_json_len) -> i32` | After epoch; return non-zero to stop training early |
| `tmm_on_validation_end(metrics_json_ptr, metrics_json_len)` | After validation loop |
| `tmm_on_train_end(metrics_json_ptr, metrics_json_len)` | Training loop complete |
| `tmm_destroy()` | Plugin unloaded |

**Host imports** available to plugins:

| Import | Description |
|--------|-------------|
| `tmm::log(level, msg_ptr, msg_len)` | Write to tmm's logger |
| `tmm::emit_metric(name_ptr, name_len, value_f64, step_i64)` | Emit a scalar metric |
| `tmm::get_param(name_ptr, name_len, out_ptr, out_len) -> i32` | Read a parameter tensor |
| `tmm::set_hyperparameter(key_ptr, key_len, value_ptr, value_len)` | Modify a config value between epochs |
| `tmm::request_stop()` | Request graceful training termination |
| `tmm::alloc(size) -> ptr` | Allocate memory in the plugin's linear memory |

**Minimal Rust plugin example:**

```rust
// src/lib.rs
#[no_mangle]
pub extern "C" fn tmm_init(cfg: *const u8, len: usize) -> i32 { 0 }

#[no_mangle]
pub extern "C" fn tmm_on_epoch_end(epoch: i32, _metrics: *const u8, _len: usize) -> i32 {
    // Return 1 to trigger early stop
    0
}
```

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[profile.release]
opt-level = "s"
```

```sh
cargo build --target wasm32-unknown-unknown --release
# Output: target/wasm32-unknown-unknown/release/my_plugin.wasm
```

### Built-in Plugins

The following plugins are distributed alongside `tmm`:

| Plugin | Description |
|--------|-------------|
| `console-ui` | Full-screen FTXUI terminal dashboard with live loss curves and parameter norms |
| `tensorboard` | Emit metrics to a TensorBoard event file |
| `wandb` | Log metrics and hyperparameters to Weights & Biases |
| `early-stopping` | Stop training when a monitored metric stops improving |
| `csv-logger` | Append per-epoch metrics to a CSV file |
| `model-summary` | Print parameter count and shape table at training start |

---

## Hyperparameter Sweeps

`tmm sweep` orchestrates multiple training runs over a search space:

```sh
tmm sweep model.yml trainer.yml --sweep-config sweep.yml
```

- **Grid search**: exhaustively tries all combinations
- **Random search**: samples uniformly from the space for `max_trials` runs
- **Bayesian search**: uses a Gaussian Process surrogate to focus on promising regions

Results are written to `./sweep_results/` (override with `--output-dir`):

```
sweep_results/
  summary.json          # all trials, sorted by objective metric
  trial_0001/
    config.yml          # effective config for this trial
    metrics.json
    checkpoints/
  trial_0002/
    ...
```

Parallel sweeps across multiple local devices:

```yaml
sweep:
  parallelism: 4        # runs 4 trials concurrently, one per device
  devices: [cuda:0, cuda:1, cuda:2, cuda:3]
```

---

## TVM FFI Integration

`tmm` communicates with user models through the [DLPack](https://dmlc.github.io/dlpack/latest/) standard (`DLTensor`) which TVM's runtime exposes. No TVM Python runtime is required at inference/training time — only `libtvm_runtime`.

### Expected Function Signature

Your compiled module must export a function with the following contract:

```
forward_backward(param_0, param_1, ..., param_N, input_batch)
  -> (output_batch, grad_0, grad_1, ..., grad_N)
```

- All tensors are `DLTensor` (or `tvm::runtime::NDArray`)
- `param_i` are the trainable parameters (same order as declared in config)
- `input_batch` is the current mini-batch (after preprocessing)
- `output_batch` is the model prediction (used for loss computation)
- `grad_i` are the loss gradients with respect to `param_i`

The loss is computed by `tmm` using the configured `loss.type`. If you prefer to compute the loss inside your compiled function, set `loss.type: external` and return the scalar loss as the first output.

### Compiling with TVM

```python
import tvm
from tvm import relax

# ... define your model graph in Relax or TE ...
# Attach gradient pass
with tvm.transform.PassContext(opt_level=3):
    mod = relax.transform.Gradient()(mod)
    lib = relax.build(mod, target="llvm")

lib.export_library("model.so")
```

### Using Other Frameworks

Any framework that can output a shared library exporting a function callable via the C ABI with `DLTensor` arguments is compatible. Examples:

- **IREE**: compile with `iree-compile --iree-hal-target-backends=llvm-cpu`
- **Custom MLIR**: lower through `bufferization` + `llvm` dialects, emit a C-ABI wrapper
- **Manually**: write a C++ function that calls your framework and marshal tensors as `DLTensor`

---

## Examples

### MNIST MLP

```sh
cd examples/mnist-mlp
tmm fit config.yml
```

### Image Classification with Sweep

```sh
cd examples/imagenet-sweep
tmm sweep model.yml --sweep-config sweep.yml
```

### Using the Console UI Plugin

```yaml
trainer:
  plugins:
    - name: console-ui
      config:
        refresh_rate_ms: 100
        show_parameter_norms: true
```

### Custom Early Stopping Plugin

```yaml
trainer:
  plugins:
    - name: early-stopping
      config:
        monitor: val_loss
        patience: 10
        min_delta: 1.0e-4
        mode: min
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TMM_PLUGIN_DIR` | `~/.tmm/plugins` | Directory scanned for `.wasm` plugin files |
| `TMM_CHECKPOINT_DIR` | `./checkpoints` | Default checkpoint output directory |
| `TMM_LOG_LEVEL` | `info` | Log level: `trace`, `debug`, `info`, `warn`, `error` |
| `TMM_DEVICE` | `cpu` | Default device (overridden by config or `--device`) |
| `TMM_CACHE_DIR` | `~/.tmm/cache` | Cache for downloaded datasets and TVM modules |
| `TVM_HOME` | _(auto-detect)_ | Path to TVM installation |

---

## License

Apache 2.0. See [LICENSE](LICENSE).
