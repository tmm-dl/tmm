# TODO

Implementation roadmap for `ttm` (train-my-model). Tasks are grouped by component and ordered roughly by dependency/priority. All XDG paths follow the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/):

| Purpose | Path |
|---|---|
| Configuration | `$XDG_CONFIG_HOME/ttm/` (`~/.config/ttm/`) |
| Persistent data (dataset index) | `$XDG_DATA_HOME/ttm/` (`~/.local/share/ttm/`) |
| Cache (cloned repos, prepared datasets) | `$XDG_CACHE_HOME/ttm/` (`~/.cache/ttm/`) |
| Runtime state (PID files, lock files) | `$XDG_RUNTIME_DIR/ttm/` |
| Log files | `$XDG_STATE_HOME/ttm/` (`~/.local/state/ttm/`) |

---

## Infrastructure

- [ ] Implement XDG directory resolution helper (`ttm::xdg`) with fallbacks per spec
- [ ] Add structured logging via spdlog: verbosity levels, optional log file under `$XDG_STATE_HOME/ttm/logs/`
- [ ] Set up Catch2 testing framework; create `tests/` directory with CMakeLists.txt
- [ ] Add CI pipeline (GitHub Actions): build, lint (clang-tidy), format check (clang-format), test
- [ ] Establish error-handling conventions: typed errors / `std::expected<T, Error>` throughout
- [ ] Add `version` subcommand printing build version and dependency versions

---

## Config

The config system is the spine of the tool; most other components depend on it.

### Loading
- [ ] Choose and integrate a YAML library (e.g., [yaml-cpp](https://github.com/jbeder/yaml-cpp) or [rapidyaml](https://github.com/biojppm/rapidyaml)) via CPM
- [ ] Load a single configuration file (explicit path or auto-discovery: `./ttm.yaml`, `$XDG_CONFIG_HOME/ttm/config.yaml`)
- [ ] Load and deep-merge multiple config files (later files override earlier; same semantics as Helm values)
- [ ] Support `--config` flag (repeatable) and `--set key=value` CLI overrides (highest priority)

### Composition
- [ ] Implement cross-file references: `${ref:path.to.value}` resolves within the merged config
- [ ] Implement environment variable expansion: `${env:MY_VAR}` and `${env:MY_VAR:-default}`
- [ ] Support config includes: `!include path/to/partial.yaml` (useful for splitting large configs)
- [ ] Validate merged config against a schema (could leverage JSON Schema + a C++ validator, or hand-rolled)

### Sections (at minimum)
- [ ] `trainer:` — epochs, batch size, precision, device, gradient accumulation steps, clip norm, logging interval
- [ ] `model:` — path/URI to TVM compiled module, input/output shapes and dtypes
- [ ] `dataset:` — source URI(s), split ratios, seed, transforms list
- [ ] `optimizer:` — type, learning rate, weight decay, momentum, betas
- [ ] `scheduler:` — type, warmup steps, total steps, min lr
- [ ] `plugins:` — list of plugin URIs to load and their per-plugin configs
- [ ] `sweep:` — parameter search space, strategy, budget, metric to optimize

---

## Plugins

Tasks (fit, validate, predict, sweep) are themselves first-class plugins. Anyone can register new tasks, optimizers, LR schedulers, dataset sources, or callbacks.

### Core Plugin System
- [ ] Define stable C ABI for WASM plugins (function signatures, memory layout, versioning)
- [ ] Implement `PluginManager`: discovery, load ordering, dependency resolution between plugins
- [ ] Implement WASM plugin loader using WAMR: sandbox instantiation, per-plugin linear memory, capability grants
- [ ] Support native shared-library plugins (`.so`/`.dylib`/`.dll`) as a fallback for trusted plugins
- [ ] Implement plugin lifecycle hooks: `ttm_plugin_init`, `ttm_plugin_teardown`, ABI version negotiation
- [ ] Plugin configuration: each plugin receives its own config subtree from the merged config
- [ ] Plugin capability model: declare required capabilities (filesystem access, network, GPU) in plugin manifest

### Plugin Sources & Registry
Plugins are **project-local only** — modelled after npm. The project config (`plugins:` section in `ttm.yaml`) is the manifest declaring plugins and version ranges (like `package.json`). A `ttm.lock` file pins exact content hashes / git SHAs for reproducible installs (like `package-lock.json`). Downloaded WASM binaries are cached globally under `$XDG_CACHE_HOME/ttm/plugins/<content-hash>/` and shared across projects that use the same content hash.

- [ ] Design `ttm.lock` format: maps plugin name → URI + exact content hash + resolved download URL
- [ ] `ttm plugin install [<uri> [--as name] [--save-dev]]` — without args: fetch all plugins declared in manifest (like `npm install`); with a URI: add to manifest and lock file
- [ ] `ttm plugin update [name]` — re-resolve version constraint, update lock file entry
- [ ] `ttm plugin remove <name>` — remove from manifest and lock file
- [ ] `ttm plugin list` — show installed plugins (from lock file) and their cache status
- [ ] `ttm plugin fetch` — fetch all locked plugins into cache without running a task (CI-friendly)
- [ ] Load plugins from local filesystem path (absolute or relative to config file)
- [ ] Download and cache plugins from Git URLs via libgit2 (into `$XDG_CACHE_HOME/ttm/plugins/<content-hash>/`)
- [ ] Support shorthand URIs: `gh:owner/repo[@ref]`, `gl:`, `bb:`, `hf:` (Hugging Face Hub spaces/plugins), `sr:` (SourceHut), `gitea:` (self-hosted)
- [ ] OCI/ORAS registry support (`oras://registry/image:tag`) — emerging standard for WASM artifact distribution

### Plugin Extension Points (registries, called by core at runtime)
- [ ] **Task registry** — register top-level CLI subcommands (e.g., `fit`, `validate`, `predict`, `sweep`, or custom tasks)
- [ ] **Dataset source registry** — register handlers for custom URI schemes
- [ ] **Transform registry** — register named data transforms (called during dataset preparation)
- [ ] **Optimizer registry** — register custom optimizers
- [ ] **LR scheduler registry** — register custom schedulers
- [ ] **Callback/event registry** — register on_epoch_start, on_batch_end, on_train_end, etc.
- [ ] **Metric registry** — register custom evaluation metrics

### Built-in Plugins (shipped with ttm)

#### Tasks (required — replaces hard-coded subcommands)
- [ ] `task-fit` — training loop
- [ ] `task-validate` — evaluation loop
- [ ] `task-predict` — inference / export predictions
- [ ] `task-sweep` — hyperparameter search orchestration

#### UI / Logging
- [ ] `console-ui` — rich terminal progress bars and metrics table (using e.g. FTXUI or a simpler approach)
- [ ] `csv-logger` — append epoch/step metrics to a CSV file under `$XDG_STATE_HOME/ttm/runs/`
- [ ] `tensorboard-logger` — write TensorBoard event files
- [ ] `json-logger` — newline-delimited JSON log (JSONL) for easy post-processing

#### Experiment Tracking (optional, one per tracker)
- [ ] `mlflow-logger` — log to MLflow tracking server (via HTTP REST)
- [ ] `wandb-logger` — Weights & Biases integration
- [ ] `aim-logger` — Aim experiment tracker

#### Dataset Sources (built-in, as plugins)
- [ ] `source-local` — local files / directories (glob patterns)
- [ ] `source-git` — arbitrary Git repository (with libgit2)
- [ ] `source-huggingface` — Hugging Face Hub datasets API
- [ ] `source-webdataset` — sharded `.tar` WebDataset streaming
- [ ] `source-s3` / `source-gcs` / `source-azure` — cloud object storage
- [ ] `source-dvc` — DVC remote (reads `.dvc` files, fetches from remote)
- [ ] `source-kaggle` — Kaggle API (`kaggle://competition-or-dataset`)
- [ ] `source-openml` — OpenML benchmark catalog
- [ ] `source-sql` — SQL database via ODBC

#### Transforms (built-in, as plugins)
- [ ] `transform-tokenize` — text tokenization (sentencepiece / tiktoken compatible)
- [ ] `transform-normalize` — feature normalization (mean/std, min-max)
- [ ] `transform-shuffle` — deterministic shuffle with seed
- [ ] `transform-batch` — batching with optional padding
- [ ] `transform-image-decode` — JPEG/PNG decode to tensor

---

## Datasets

### Loading & Sources
- [ ] Implement the dataset URI router: parse scheme (e.g., `file://`, `gh:`, `hf:`, `s3://`) and dispatch to the appropriate registered source plugin
- [ ] `file://` / bare path — load files from disk using Arrow's filesystem abstraction
  - [ ] Apache Arrow IPC / Feather
  - [ ] Apache Parquet
  - [ ] CSV / TSV (via Arrow CSV reader)
  - [ ] JSONL (via Arrow JSON reader)
  - [ ] Folder of files (glob: `data/**/*.parquet`)
- [ ] Git repositories: clone/fetch via libgit2 into `$XDG_CACHE_HOME/ttm/datasets/<hash>/`
  - [ ] Shorthand URI expansion: `gh:owner/repo[@ref]`, `gl:`, `bb:`, `hf:`, `sr:`, `gitea://host/`
  - [ ] Sparse checkout (only the relevant subdirectory if specified)
  - [ ] Incremental updates: `git fetch` + fast-forward if already cached
  - [ ] Dataset versioning: pin to a commit SHA in config for reproducibility
- [ ] Dataset splits: `train`, `val`, `test` — either from config or auto-split with seed
- [ ] Multi-source datasets: concatenate or interleave multiple sources with optional weights

### Caching & Preparation
- [ ] Raw data cache: store downloaded/cloned data in `$XDG_CACHE_HOME/ttm/datasets/raw/<content-hash>/`
- [ ] Prepared data cache: store transform output in `$XDG_CACHE_HOME/ttm/datasets/prepared/<config-hash>/`
- [ ] Cache invalidation: recompute if config hash changes (dataset URI + transforms + version)
- [ ] Lock file during preparation to prevent concurrent re-computation
- [ ] `ttm dataset prepare` — explicit preparation command (also run implicitly before training)
- [ ] `ttm dataset inspect` — show schema, row count, sample rows, cache status
- [ ] `ttm dataset clear-cache [uri]`

### Data Pipeline
- [ ] Design `DatasetIterator` / `DataLoader` C++ abstraction (DLPack tensors as the exchange format)
- [ ] Apply transforms in order from config; transforms can be plugins (see Plugin section)
- [ ] Prefetch + background loading (producer/consumer with a thread pool)
- [ ] Deterministic shuffling with global seed for reproducibility
- [ ] Streaming mode: iterate without loading all data into memory (Arrow RecordBatch streaming)

### Formats to Support (beyond Arrow)
The following formats appear frequently in the ML ecosystem and should be covered by source/transform plugins:

| Format | Notes |
|---|---|
| WebDataset (`.tar` shards) | De facto standard for large-scale vision/NLP; supports streaming |
| LMDB | Used by many frameworks for fast random-access preprocessed data |
| HDF5 / `.h5` | Common in scientific ML (e.g., medical imaging, genomics) |
| Zarr | Cloud-native N-dim arrays; growing in biomed / geospatial |
| NumPy `.npz` | Ubiquitous in academic code |
| SafeTensors | Hugging Face weight file format; can also store dataset shards |
| Lance | Modern columnar format for ML (LanceDB ecosystem); supports versioning |
| TFDS | TensorFlow Datasets catalog (`tfds://dataset_name`) |
| OpenML | Benchmark datasets via REST API |
| DVC remotes | S3, GCS, Azure, SSH — data treated like versioned code |
| Kaggle API | `kaggle://` shorthand |
| Roboflow | Vision datasets (detection, segmentation, classification) |
| SQL / ODBC | Tabular data directly from databases |

---

## Core

### Model Loading (TVM)
- [ ] Implement `ModelLoader`: load a compiled TVM module (`.so` + `.json` + `.params` or a single `.tar`)
- [ ] Expose named input/output tensors with shapes and dtypes (from model metadata)
- [ ] Device management: CPU, CUDA, Metal, Vulkan — select via config `device: cuda:0`
- [ ] DLPack as the tensor exchange format between dataset, model, and optimizer
- [ ] Validate model input shapes against dataset output shapes at startup (fail fast)

### Training Loop (task-fit plugin, but core provides the harness)
- [ ] Implement the forward pass: feed batch → run TVM module → get logits/outputs
- [ ] Implement the backward pass: compute gradients via TVM's autograd or external loss+grad
- [ ] Basic training loop: for epoch in epochs: for batch in dataloader: forward → loss → backward → step
- [ ] Gradient accumulation: accumulate over N micro-batches before optimizer step
- [ ] Mixed precision: FP16 / BF16 forward pass with FP32 master weights
- [ ] Gradient clipping: clip by global norm
- [ ] Per-step and per-epoch metrics: loss, throughput (samples/sec), learning rate

### Optimizers
- [ ] SGD (with momentum, weight decay, Nesterov)
- [ ] Adam / AdamW (eps, betas, weight decay, amsgrad variant)
- [ ] AdaGrad, RMSProp
- [ ] Plugin-extensible: custom optimizers register via the optimizer registry

### LR Schedulers
- [ ] Constant
- [ ] Linear warmup + constant
- [ ] Linear warmup + linear decay
- [ ] Cosine annealing with warm restarts
- [ ] ReduceLROnPlateau (metric-driven)
- [ ] Plugin-extensible

### Checkpointing
- [ ] Save checkpoint: model weights, optimizer state, scheduler state, epoch, step, RNG state
- [ ] Save to `$XDG_STATE_HOME/ttm/checkpoints/<run-id>/` or user-specified path
- [ ] Load checkpoint: resume training from a checkpoint (automatic latest-checkpoint detection)
- [ ] Best-model checkpointing: keep top-K checkpoints by a monitored metric
- [ ] Export: convert a checkpoint to a standalone TVM compiled module for deployment
- [ ] SafeTensors format for weight storage (safe, fast, supports partial loading)

### Validation Loop (task-validate plugin)
- [ ] Separate eval dataloader (no shuffle, no augmentation)
- [ ] Collect predictions and targets; compute registered metrics
- [ ] Support custom metrics via plugin registry

### Inference / Prediction (task-predict plugin)
- [ ] Load model + weights; run on a dataset or single input
- [ ] Write outputs to a file (CSV, Arrow IPC, JSONL) or stdout
- [ ] Batch inference with configurable batch size

### Hyperparameter Sweep (task-sweep plugin)
- [ ] Grid search: enumerate all combinations
- [ ] Random search: sample N combinations uniformly
- [ ] Bayesian optimization: Gaussian Process surrogate (Eigen-based) with expected improvement acquisition
- [ ] Early stopping: median stopping rule, Hyperband (successive halving)
- [ ] Parallel trials: run N trials concurrently (subprocess per trial)
- [ ] Report best config and metric to stdout / log

---

## CLI

Built on CLI11; subcommands correspond to registered task plugins.

- [ ] `ttm fit [--config ...] [--set key=val] [checkpoint]` — train a model
- [ ] `ttm validate [--config ...] [checkpoint]` — evaluate on val/test split
- [ ] `ttm predict [--config ...] [--output file] [checkpoint] [input]` — run inference
- [ ] `ttm sweep [--config ...]` — hyperparameter search
- [ ] `ttm plugin install <uri> [--as name]` — add plugin to project config
- [ ] `ttm plugin list` — show plugins declared in current project config and their cache status
- [ ] `ttm plugin update [name]` — update pinned version in project config
- [ ] `ttm plugin remove <name>` — remove plugin from project config
- [ ] `ttm plugin fetch` — pre-fetch all project plugins into cache
- [ ] `ttm dataset prepare [--config ...]` — explicitly prepare datasets
- [ ] `ttm dataset inspect <uri>` — show schema and statistics
- [ ] `ttm dataset clear-cache [uri]` — remove cached datasets
- [ ] `ttm config dump [--config ...]` — print final merged config (useful for debugging)
- [ ] `ttm version` — print versions
- [ ] Dynamic subcommand registration: task plugins add their own subcommands at startup
- [ ] Shell completion scripts (bash, zsh, fish) via CLI11

---

## Notes

### On task-as-plugin design
`fit`, `validate`, `predict`, and `sweep` are all built-in plugins that ship with ttm. Core only provides the harness (config, XDG paths, plugin manager, dataset router, DLPack types). This means third parties can implement entirely new training paradigms (e.g., reinforcement learning, contrastive learning, RLHF) as plugins without forking ttm.

### On dataset source coverage
The dataset URI router should treat sources as an open registry. Priority order when resolving a bare path with no scheme: `file://` → warn and ask the user to be explicit. Any URI scheme unknown to core is dispatched to a plugin that registered that scheme.

### On credentials
Credentials for private dataset sources (Git SSH keys, HF tokens, S3 credentials, Kaggle API key) should follow platform conventions:
- Git: respect `~/.ssh/`, `.netrc`, `GIT_SSH_COMMAND`, `GITHUB_TOKEN`, etc.
- HF: `HF_TOKEN` env var or `~/.cache/huggingface/token`
- S3/GCS/Azure: standard SDK credential chains (env vars, `~/.aws/credentials`, instance metadata)
- Kaggle: `~/.config/kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`
- All credentials read-only from environment; never stored in config files (which may be committed)
