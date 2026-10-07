# scUNVEIL training

This repository contains the data-preparation and pretraining pipeline used to
train scUNVEIL on human single-cell RNA-sequencing data from the CZ CELLxGENE
Census. It covers the complete path from downloading and filtering Census data
to training the model and building a publication bundle for the separate
scUNVEIL inference package.

The repository is intentionally research-oriented. The numbered scripts and
notebooks are the canonical description of the training procedure; the goal of
this README is to make their required order, inputs, and outputs unambiguous.

## Pipeline overview

```text
CZ CELLxGENE Census
        |
        v
filtered per-dataset AnnData files
        |
        v
global gene-frequency ordering
        |
        v
randomly shuffled training shards
        |
        v
scUNVEIL pretraining and float32 checkpoints
        |
        v
float16 checkpoint export and loss comparison
        |
        v
streamed full PCA of model embeddings
        |
        v
publish/ bundle for the inference repository
```

During pretraining, each raw count vector is split by binomial thinning. The
model receives the log-transformed retained counts and predicts the normalized
distribution of the held-out counts. The final normalized hidden representation
is used as the cell embedding. The output projection uses the transpose of the
same gene-to-embedding weight matrix as the input projection.

## Repository structure

```text
.
├── .devcontainer/          Standalone GPU container configuration
├── src/
│   ├── prepare_data/       Numbered CELLxGENE preparation scripts
│   └── training/           Model code, pretraining notebook, export script
├── data/                   Mounted data directory; not tracked by Git
└── results/                Mounted training-output directory; not tracked by Git
```

The `data/` and `results/` directories are deliberately excluded from Git
because a full run produces hundreds of gigabytes of data and large model
checkpoints.

## Requirements

- A Linux host with Docker
- Either Visual Studio Code with the Dev Containers extension or the Docker
  CLI
- An NVIDIA GPU with a working NVIDIA Container Toolkit installation
- Internet access for building the image and querying the CELLxGENE Census
- Substantial local storage; expect the complete workflow to require hundreds
  of gigabytes

Pretraining is designed for a high-memory GPU and is not practical as a CPU
workflow. Exact storage and runtime depend on the Census snapshot, hardware,
and number of retained checkpoints.

## Environment setup

The standalone repository environment is defined by
[`.devcontainer/Dockerfile`](.devcontainer/Dockerfile) and
[`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json). The
container currently uses the TensorFlow 2.17 GPU/Jupyter image and installs
the CELLxGENE client, AnnData, scikit-learn, W&B, and the other Python
dependencies used here. The development workspace can also run these scripts
if it provides the same packages and mounts at `/workspace/data` and
`/workspace/results`.

Create persistent host directories for data and results, then expose their
locations as environment variables. For example, add the following to
`~/.bashrc`:

```bash
export SCUNVEIL_DATA_DIR="/path/to/scunveil-data"
export SCUNVEIL_RESULTS_DIR="/path/to/scunveil-results"
```

Reload the shell configuration, then create the mounted directories if they do
not already exist. The training code expects the `training/` results directory
to exist before a run is created:

```bash
source ~/.bashrc
mkdir -p "$SCUNVEIL_DATA_DIR" "$SCUNVEIL_RESULTS_DIR/training"
```

### Option A: VS Code Dev Container

Open this repository in VS Code from that environment and select **Dev
Containers: Reopen in Container**. The Dev Container configuration builds the
image, enables GPU access, and creates the required mounts automatically:

```text
$SCUNVEIL_DATA_DIR     -> /workspace/data
$SCUNVEIL_RESULTS_DIR  -> /workspace/results
```

### Option B: Docker CLI

VS Code is not required. From the repository root, build the same image
directly:

```bash
docker build \
  -f .devcontainer/Dockerfile \
  -t scunveil-training \
  .
```

Then start an interactive container with the same workspace location, GPU
access, and persistent data and results mounts:

```bash
docker run --rm -it \
  --gpus all \
  -p 8888:8888 \
  -v "$PWD:/workspace" \
  -v "$SCUNVEIL_DATA_DIR:/workspace/data" \
  -v "$SCUNVEIL_RESULTS_DIR:/workspace/results" \
  -w /workspace \
  scunveil-training \
  bash
```

Scripts can be run directly from this shell. To work with the notebooks in a
browser, start Jupyter Lab inside the container and open the URL it prints:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Both approaches provide the same repository layout. All source code assumes
the `/workspace/data` and `/workspace/results` container paths; run the
remaining commands inside the selected container environment. In the combined
`scunveil-dev` workspace, the source lives at
`/workspace/subprojects/01_scunveil-training/`; use its `src/prepare_data` and
`src/training` directories in the commands below. The data and results paths
remain `/workspace/data` and `/workspace/results`.

## 1. Prepare the CELLxGENE training data

The preparation scripts are ordered by their numeric prefixes and should be
run from their own directory so that local imports resolve consistently:

```bash
cd /workspace/src/prepare_data
```

The Census snapshot (`2025-11-08`), dataset exclusions, and filtering
thresholds are defined in
[`constants.py`](src/prepare_data/constants.py).

### 1.1 Discover datasets

```bash
python 01_find_the_dataset_list.py
```

Finds human datasets containing primary observations and writes their dataset
identifiers to `data/dataset_list.tsv`.

### 1.2 Download and filter datasets

```bash
python 02_download_cellxgene.py
```

Downloads each dataset from the Census, retains primary cells passing the
configured count and detected-gene filters, converts the sparse count matrix to
compact integer storage, and writes one file per dataset under
`data/individual_datasets/`.

This stage is resumable: completed dataset files are skipped. Incomplete files
whose names end in `-uploading.h5ad` are removed when the script starts, and
empty datasets are moved to `individual_datasets/_trash/`.

### 1.3 Apply dataset exclusions

```bash
python 03_exclude_datasets.py
```

Moves the explicitly listed exclusions from `individual_datasets/` to
`individual_datasets/_exclude/`.

### 1.4 Calculate the gene ordering

```bash
python 04_count_genes.py
```

Counts the number of nonzero observations for every gene across the downloaded
datasets. It produces:

```text
data/umi_nonzero.npy
data/genes_argsort.npy
```

`genes_argsort.npy` orders genes from most to least frequently detected and is
used consistently by the remaining stages.

### 1.5 Generate shuffled training shards

```bash
python 05_generate_shuffled_shards.py
```

Randomly mixes cells across source datasets, divides them into 1,000 equally
sized AnnData shards, applies the global gene ordering, and retains the
observation metadata used by the project. Shards are written to:

```text
data/shuffled_shards/*.h5ad
```

The script discards the small remainder needed to make every shard the same
size. It is a large, I/O-intensive operation; make sure the destination has
enough free space before starting it. Existing shard paths may be overwritten
if this stage is rerun.

### 1.6 Save dataset provenance

```bash
python 06_saving_ds_info.py
```

Reads the dataset IDs actually present in the finished training shards, checks
that excluded IDs are absent, and retrieves their Census metadata. By default
it writes:

```text
data/ds_info.csv
data/Supplementary_Table_S1_training_datasets.csv
```

Set `SCUNVEIL_METADATA_OUTPUT_ROOT` to write these two files elsewhere while
leaving the training shards untouched.

### 1.7 Save ordered gene metadata

```bash
python 07_save_var_separately.py
```

Writes the AnnData gene metadata in training order to:

```text
data/var_sorted.csv
```

## 2. Pretrain scUNVEIL

Open [`001_run_pretraining.ipynb`](src/training/001_run_pretraining.ipynb) in
the development container and run its cells from top to bottom. The notebook
must run with `/workspace/src/training` as its working directory.

The notebook:

1. defines the run configuration;
2. creates a timestamped experiment directory;
3. opens the prepared shards in backed mode;
4. constructs and compiles the model;
5. trains it using the transcript-thinning objective; and
6. saves a checkpoint after each training epoch.

The run configuration explicitly includes `emb_dim`, `ff_dim`, `n_layers`, and
`n_genes`. Keep `ff_dim` in the saved configuration: export and inference must
reconstruct exactly the architecture used for the checkpoint.

Each run is stored under:

```text
results/training/<experiment_id>/
├── config.json
├── src/                 Snapshot of the training Python modules
└── weights/
    └── <epoch>.weights.h5
```

The full configuration lives in the notebook and is copied to `config.json`
when the experiment is created. Model and data-loader implementations are in
[`model.py`](src/training/model.py),
[`dataset.py`](src/training/dataset.py), and
[`data_operations.py`](src/training/data_operations.py).

### Optional Weights & Biases logging

The current notebook has `CONFIG['use_wandb'] = True`. Set it to `False` if you
do not want to log a run. To use W&B, log in from a terminal inside the same
container and as the same user as the notebook:

```bash
wandb login
```

W&B uses the saved credentials automatically; no `wandb_login.json` file is
needed. The project is selected by `CONFIG['project']`. To select a specific
team/account, optionally set `WANDB_ENTITY` in the environment before starting
the notebook kernel; otherwise W&B uses its configured/default entity.

Credentials saved inside a container may need to be established again after
rebuilding it unless its credential storage is persisted.

## 3. Build the inference publish bundle

Edit the constants at the top of
[`002_export_publish_package.py`](src/training/002_export_publish_package.py):
set `WEIGHTS_PATH` to the original float32 `.weights.h5` checkpoint you want
to publish, and choose `N_LOSS_CELLS`, `N_PCA_CELLS`, `BATCH_SIZE`, and
`MAX_LOSS_INCREASE`. The script has no command-line arguments. Run it from the
training directory so its local imports resolve:

```bash
cd /workspace/src/training
python 002_export_publish_package.py
```

The script opens training shards in backed mode and then:

1. copies the checkpoint weights to float16 in bounded HDF5 chunks, excluding
   optimizer state;
2. loads the float32 and float16 weights into separate models and compares
   their loss on identical freshly thinned batches from the **training** shard
   distribution, rejecting an increase above `MAX_LOSS_INCREASE`;
3. computes all PCA components from a streamed float64 embedding covariance
   matrix, using the float16-exported weights, and checks the stored component
   orientation; and
4. copies the run configuration and ordered gene metadata into one publish
   directory, along with an export report.

The loss comparison tests rounding on training-style samples, not performance
on an independent test dataset. Embeddings are processed by batch and are not
stored as intermediate shard files or clipped. The PCA matrix has principal
components in its **columns**; the inference package applies the saved mean
and matrix in that orientation.

The current script defaults to 100,000 cells for the loss check and 1,000,000
cells for PCA. It first writes into a temporary directory and renames that
directory only after a successful export. It refuses to overwrite an existing
`publish/` directory.

```text
results/training/<experiment_id>/publish/
├── config.json
├── var_sorted.csv
├── weights.weights.h5
├── pca_mean.npy
├── pca_mat.npy
└── export_report.json
```

For the separately released inference package, upload the **contents** of
`publish/` directly to `models/<version>/` in the scUNVEIL Hugging Face model
repository. The five files other than `export_report.json` are used by the
inference loader; the report records the source checkpoint and export checks.
The inference package chooses its default model version in its own code.

## Expected data layout

After completing the workflow, the relevant files have this structure:

```text
data/
├── dataset_list.tsv
├── individual_datasets/
├── umi_nonzero.npy
├── genes_argsort.npy
├── shuffled_shards/
├── ds_info.csv
├── Supplementary_Table_S1_training_datasets.csv
└── var_sorted.csv

results/
└── training/
    └── <experiment_id>/
        ├── config.json
        ├── src/
        ├── weights/
        └── publish/
```

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
