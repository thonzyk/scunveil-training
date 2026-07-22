# scUNVEIL training

This repository contains the data-preparation and pretraining pipeline used to
train scUNVEIL on human single-cell RNA-sequencing data from the CZ CELLxGENE
Census. It covers the complete path from downloading and filtering Census data
to training the model and exporting embeddings with a PCA transformation.

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
scUNVEIL pretraining
        |
        v
cell embeddings
        |
        v
PCA matrix and mean
```

During pretraining, each raw count vector is split by binomial thinning. The
model receives the log-transformed retained counts and predicts the normalized
distribution of the held-out counts. The final normalized hidden representation
is used as the cell embedding.

## Repository structure

```text
.
├── .devcontainer/          Reproducible GPU development environment
├── src/
│   ├── prepare_data/       Numbered CELLxGENE preparation scripts
│   └── training/           Model code and ordered training notebooks
├── data/                   Mounted data directory; not tracked by Git
└── results/                Mounted training-output directory; not tracked by Git
```

The `data/` and `results/` directories are deliberately excluded from Git
because a full run produces hundreds of gigabytes of data and large model
checkpoints.

## Requirements

- A Linux host with Docker
- Visual Studio Code with the Dev Containers extension for the recommended
  workflow, or the Docker CLI for the editor-independent alternative
- An NVIDIA GPU with a working NVIDIA Container Toolkit installation
- Internet access for building the image and querying the CELLxGENE Census
- Substantial local storage; expect the complete workflow to require hundreds
  of gigabytes

Pretraining is designed for a high-memory GPU and is not practical as a CPU
workflow. Exact storage and runtime depend on the Census snapshot, hardware,
and number of retained checkpoints.

## Environment setup

The supported environment is defined by
[`.devcontainer/Dockerfile`](.devcontainer/Dockerfile) and
[`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json). The
container provides TensorFlow with GPU support, Jupyter, the CELLxGENE client,
AnnData, scikit-learn, and the remaining Python dependencies used here.

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

### Recommended: VS Code Dev Container

Open this repository in VS Code from that environment and select **Dev
Containers: Reopen in Container**. The Dev Container configuration builds the
image, enables GPU access, and creates the required mounts automatically:

```text
$SCUNVEIL_DATA_DIR     -> /workspace/data
$SCUNVEIL_RESULTS_DIR  -> /workspace/results
```

### Alternative: Docker CLI

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
remaining commands inside the selected container environment.

## 1. Prepare the CELLxGENE training data

The preparation scripts are ordered by their numeric prefixes and should be
run from their own directory so that local imports resolve consistently:

```bash
cd /workspace/src/prepare_data
```

The Census snapshot and filtering thresholds are defined in
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

Randomly mixes cells across source datasets, divides them into equally sized
AnnData shards, applies the global gene ordering, and retains the observation
metadata used by the project. Shards are written to:

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

Retrieves Census metadata for the selected datasets and writes:

```text
data/ds_info.csv
data/Supplementary_Table_S1_training_datasets.csv
```

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

Each run is stored under:

```text
results/training/<experiment_id>/
├── config.json
└── weights/
    └── <epoch>.weights.h5
```

The full configuration lives in the notebook and is copied to `config.json`
when the experiment is created. Model and data-loader implementations are in
[`model.py`](src/training/model.py),
[`dataset.py`](src/training/dataset.py), and
[`data_operations.py`](src/training/data_operations.py).

### Optional Weights & Biases logging

Training runs without W&B by default. To enable it, create
`src/training/wandb_login.json` before running the notebook:

```json
{
  "key": "YOUR_WANDB_API_KEY",
  "entity": "YOUR_WANDB_ENTITY"
}
```

The file is ignored by Git. Its location is resolved relative to the notebook's
working directory, which is another reason to run the notebook from
`/workspace/src/training`.

## 3. Generate cell embeddings

Open [`002_generate_embeddings.ipynb`](src/training/002_generate_embeddings.ipynb).
In its first cell, select the completed `EXPERIMENT_ID` and the desired
checkpoint through `WEIGHTS_ID`, then run the notebook from top to bottom.

The notebook reconstructs the model from the saved `config.json`, loads the
selected checkpoint, and exports the final hidden representation for cells in
the shuffled shards. Embeddings are stored as clipped float16 NumPy arrays:

```text
data/shuffled_shards_emb/<shard_id>.npy
```

As currently written, the notebook processes the first 50 shuffled shards. If
more embedding shards are required, adjust the shard slice in the final cell
deliberately and ensure sufficient storage is available.

## 4. Fit and export PCA

Open [`003_calculate_PCA.ipynb`](src/training/003_calculate_PCA.ipynb), select
the same experiment and checkpoint in its first cell, and run all cells.

The notebook fits PCA on a subset of the generated embeddings, checks
reconstruction on a held-out subset, and saves the complete PCA transformation
next to the selected checkpoint:

```text
results/training/<experiment_id>/weights/<checkpoint>_pca_mat.npy
results/training/<experiment_id>/weights/<checkpoint>_pca_mean.npy
```

Together, the model checkpoint, PCA matrix, PCA mean, ordered gene metadata,
and experiment configuration describe the exported representation.

## Expected data layout

After completing the workflow, the relevant files have this structure:

```text
data/
├── dataset_list.tsv
├── individual_datasets/
├── umi_nonzero.npy
├── genes_argsort.npy
├── shuffled_shards/
├── shuffled_shards_emb/
├── ds_info.csv
├── Supplementary_Table_S1_training_datasets.csv
└── var_sorted.csv

results/
└── training/
    └── <experiment_id>/
        ├── config.json
        └── weights/
```

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
