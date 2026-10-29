# CEL
Concept Extraction Library (with hydra)

## Structure of the Repository
- **modules**: Contains the core code for what you want to do. e.g. training and testing models.
- **data**: Directory to store datasets and related files.
  - `config/`: Configuration files for Hydra.
    - `train_model.yaml`: (e.g) Default configuration for training.
- **env setup**: Environment setup files.
  - `Dockerfile`: Docker configuration for containerized environment.
  - `requirements.txt`: Python dependencies.
- **runs**: Contains scripts for experimental runs, create datasets, train models, analyze them, etc.
  - `task10_train_model.py`: Main script for training the model.
  - `task20_concept_extraction.py`: Main script for fitting a CE method and producing basic report


# Reproducing CENDRe — Code Submission

This document accompanies the code submission for the paper *CENDRe:
Concept Extraction with Natural Domain Representations*. It documents
the exact commands used to produce every figure and metric in the main
text and appendix.

The codebase is the **Concept Extraction Library (CEL)**, a
Hydra-managed pipeline located at
`supplementary_material/CEL-main/`. CEL is a general-purpose CE
toolbox; this README only covers the configurations needed to
reproduce the paper. For a tour of the library and environment setup,
see `supplementary_material/CEL-main/README.md`.

## 1. Environment

Follow `supplementary_material/CEL-main/README.md` §"Setting Up the
Virtual Environment". In short:

```bash
cd supplementary_material/CEL-main
conda create --prefix ./.venv python=3.10.4
conda activate ./.venv
pip install -r ./env_setup/requirements.txt
```

A Dockerfile is also provided at `env_setup/Dockerfile`.

Hardware in the paper: a single workstation with two NVIDIA A100-40GB
GPUs partitioned into four ~20 GB MIG slices. Each run fits inside one
slice. The CENDRe variants run their clustering stage on CPU
because k-means / HDBSCAN over 50 LAD micro-centroids is faster
without the GPU dispatch overhead.

## 2. Dataset paths

Edit `data/config/path/<your_path>.yaml` to point at your local dataset
and output folders, then add `path=<your_path>` to every command (or
make it the default in `data/config/config.yaml`). Schema:

```yaml
name: mypath
base_path_datasets: /path/to/Datasets
base_path_models:   ./data/models
base_path_ce:       ./data/ConceptExtraction
base_path:          ./data
results: ${dataset.name}_${transforms.name}_${model.name}_${training.name}_${training.seed}
```

All datasets, synthetic and natural, are loaded from disk via
`datasets.Dataset.load_from_disk` from paths under
`${path.base_path_datasets}` (synthetic under `synthetic/<name>`,
natural under `others/<name>`).

| Dataset      | Source |
|--------------|--------|
| Synthetic (`syntheticLocal`, `syntheticFrequency`, `syntheticLmc`, `syntheticLconf`) | Released as a prebuilt HuggingFace `Dataset` archive at <https://zenodo.org/records/20054181> |
| CWRU         | <https://engineering.case.edu/bearingdatacenter> (Smith & Randall 2015) |
| BearingPD    | <https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter> (Lessmeier et al. 2016) |

Download the synthetic archive from the link above and unpack it into
`${base_path_datasets}/synthetic/`, so that each variant ends up at
`${base_path_datasets}/synthetic/<variant_name>/` (the path the
synthetic dataset configs in `data/config/dataset/` expect). Place the
CWRU and BearingPD archives under `${base_path_datasets}/others/`
following the layout expected by `data/config/dataset/CWRU_Bearing.yaml`
and `data/config/dataset/BearingPD.yaml`.

## 3. Reproducing the paper

All commands assume the working directory is
`supplementary_material/CEL-main/`. Each experiment config in
`data/config/experiment/` defines a Hydra multirun sweep over models
(InceptionTime10, ResNet1D18, DenseNet1D121), seeds (0, 1, 2), and the
relevant dataset variants for that family.

Common flags:

| Flag | Effect |
|------|--------|
| `-m` / `--multirun` | execute the sweep defined in the experiment config |
| `mode=force`        | re-run even if cached outputs already exist |

The reported runs were executed locally on the workstation described
in §1. CEL also ships Slurm and joblib launchers in
`data/config/launcher/` (add `+launcher=slurmgpu` or `+launcher=joblib`
to a command to use them); they are not required to reproduce the
paper.

The pipeline is: **train models → extract concepts → compute metrics →
generate plots/tables**. Bundles are split by CE method:

- `_CE0` / `_metrics0`: ECLAD-ts and MultiVISION (fixed-K sweep over {2, 4, 6, 8, 10})
- `_CE1` / `_metrics1`: CENDRe with silhouette aggregation and HDBSCAN (auto-K from 50 micro-clusters)
- `_CE2` / `_metrics2`: CENDRe with k-means (fixed-K sweep)

### 3.1 Train models

Three architectures × three seeds × all variants in each dataset family.

```bash
python runs/task10_train_model.py -m +experiment=syntheticL_train
python runs/task10_train_model.py -m +experiment=syntheticF_train
python runs/task10_train_model.py -m +experiment=syntheticLmc_train
python runs/task10_train_model.py -m +experiment=syntheticLconf_train
python runs/task10_train_model.py -m +experiment=Bearing_train
```

`Bearing_train` runs on one of `BearingCWRU` or `BearingPD` at a time;
switch the `dataset:` field in
`data/config/experiment/Bearing_train.yaml` and re-run for the other
dataset.

### 3.2 Concept extraction

```bash
# CE0: ECLAD-ts, MultiVISION
python runs/task20_concept_extraction.py -m +experiment=syntheticL_CE0
python runs/task20_concept_extraction.py -m +experiment=syntheticF_CE0
python runs/task20_concept_extraction.py -m +experiment=syntheticLmc_CE0
python runs/task20_concept_extraction.py -m +experiment=syntheticLconf_CE0

# CE1: CENDRe (silhouette, HDBSCAN)
python runs/task20_concept_extraction.py -m +experiment=syntheticL_CE1
python runs/task20_concept_extraction.py -m +experiment=syntheticF_CE1
python runs/task20_concept_extraction.py -m +experiment=syntheticLmc_CE1
python runs/task20_concept_extraction.py -m +experiment=syntheticLconf_CE1

# CE2: CENDRe (k-means)
python runs/task20_concept_extraction.py -m +experiment=syntheticL_CE2
python runs/task20_concept_extraction.py -m +experiment=syntheticF_CE2
python runs/task20_concept_extraction.py -m +experiment=syntheticLmc_CE2
python runs/task20_concept_extraction.py -m +experiment=syntheticLconf_CE2

# Natural data: BearingCWRU and BearingPD
python runs/task20_concept_extraction.py -m +experiment=Bearing_CE0
```

`Bearing_CE0` covers all CE methods reported on the natural-data
experiments. The dataset is selected via the `dataset:` field of
`data/config/experiment/Bearing_CE0.yaml` (and similarly for
`Bearing_train.yaml`); switch between `BearingCWRU` and `BearingPD` and
re-run to produce the results for each.

### 3.3 Metrics (synthetic only)

sRC and sIC against the ground-truth primitive masks.

```bash
# metrics0: ECLAD-ts, MultiVISION
python runs/task30_synthetic_metrics.py -m +experiment=syntheticL_metrics0
python runs/task30_synthetic_metrics.py -m +experiment=syntheticF_metrics0
python runs/task30_synthetic_metrics.py -m +experiment=syntheticLmc_metrics0
python runs/task30_synthetic_metrics.py -m +experiment=syntheticLconf_metrics0

# metrics1: CENDRe (silhouette, HDBSCAN)
python runs/task30_synthetic_metrics.py -m +experiment=syntheticL_metrics1
python runs/task30_synthetic_metrics.py -m +experiment=syntheticF_metrics1
python runs/task30_synthetic_metrics.py -m +experiment=syntheticLmc_metrics1
python runs/task30_synthetic_metrics.py -m +experiment=syntheticLconf_metrics1

# metrics2: CENDRe (k-means)
python runs/task30_synthetic_metrics.py -m +experiment=syntheticL_metrics2
python runs/task30_synthetic_metrics.py -m +experiment=syntheticF_metrics2
python runs/task30_synthetic_metrics.py -m +experiment=syntheticLmc_metrics2
python runs/task30_synthetic_metrics.py -m +experiment=syntheticLconf_metrics2
```

The natural-data evaluations (CWRU, BearingPD) are qualitative and do
not pass through `task30`; their figures are produced directly from the
extracted concepts.

### 3.4 Plots, tables, scatter, optimal K

Aggregation over all metrics runs.

```bash
# Boxplots of sRC and sIC (Figs. in main text and appendix)
python runs/task31_plot_metrics.py -m +experiment=syntheticL_plots
python runs/task31_plot_metrics.py -m +experiment=syntheticF_plots
python runs/task31_plot_metrics.py -m +experiment=syntheticLmc_plots
python runs/task31_plot_metrics.py -m +experiment=syntheticLconf_plots
python runs/task31_plot_metrics.py -m +experiment=syntheticAll_plots

# Alignment tables
python runs/task32_alignment_tables.py -m +experiment=syntheticL_tables
python runs/task32_alignment_tables.py -m +experiment=syntheticF_tables
python runs/task32_alignment_tables.py -m +experiment=syntheticLmc_tables
python runs/task32_alignment_tables.py -m +experiment=syntheticLconf_tables

# Concept scatter plots
python runs/task33_concept_scatter.py -m +experiment=syntheticL_scatter
python runs/task33_concept_scatter.py -m +experiment=syntheticF_scatter
python runs/task33_concept_scatter.py -m +experiment=syntheticLmc_scatter
python runs/task33_concept_scatter.py -m +experiment=syntheticLconf_scatter

# Optimal-K analysis (silhouette / HDBSCAN behaviour across datasets)
python runs/task34_optimal_k.py -m

# Runtime summary (Table A.X)
python runs/task32_runtime_summary.py -m
```

## 4. Expected runtimes

Mean concept-extraction runtimes per (architecture, method) on a single
A100 MIG slice (seconds, averaged over 3 seeds on the
`syntheticLocal` square-triangle variant):

|                 | ECLAD-ts | MultiVISION | CENDRe-kMeans | CENDRe-silhouette | CENDRe-HDBSCAN |
|-----------------|---------:|------------:|--------------:|------------------:|---------------:|
| InceptionTime10 |      102 |         120 |           118 |                18 |             15 |
| ResNet1D-18     |       88 |         103 |            91 |                11 |             11 |
| DenseNet1D-121  |      251 |         321 |           227 |                40 |             35 |

Total compute budget for the project, including method development and
preliminary experiments not in the paper, is approximately 20,000 CPU
core-hours, estimated from HPC scheduler logs (a separate cluster from
the workstation used for the reported runs).

## 5. Known issues

- **CENDRe-HDBSCAN occasional convergence failure.** On 4/108
  `syntheticLocal` and 5/108 `syntheticLconf` runs (concentrated on
  `triangle-circle` bases with InceptionTime10 or ResNet1D18), HDBSCAN
  labels all 50 micro-centroids as noise and no concept is produced.
  The corresponding rows are excluded from the boxplot statistics in
  the paper. Recovery, when needed, is via per-dataset tuning of
  `min_cluster_size` / `min_samples` in
  `data/config/CE/CENDRE.yaml`.

## 6. Citation

```bibtex
% TODO: add bibtex entry on acceptance.
```

