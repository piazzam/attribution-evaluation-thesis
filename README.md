# Reliability of Attribution Evaluation Metrics — Tabular and Synthetic Experiments

Code accompanying the tabular and synthetic-data experiments presented in the thesis chapter **“On the Reliability of Attribution Evaluation Metrics”**.

The corresponding **image experiments** are available in a separate repository: [`IMAGE_REPOSITORY_URL`](IMAGE_REPOSITORY_URL).

## Scope

This repository studies how attribution-evaluation outcomes change with methodological choices such as feature-removal strategy, signed vs. absolute attribution ranking, evaluation metric, explanation target, and model quality.

The real-tabular experiments use seven datasets: Adult Income, Breast Cancer, COMPAS, German Credit, Glass, Wine Quality (red), and Digits. The explainers are TreeSHAP, KernelSHAP, and LIME. The main classifier is a random forest with 50 trees and maximum depth 6; shallow and untrained variants are included for the sanity-check experiment.

The synthetic experiments consider **15 controlled binary-classification scenarios** in which the signal-carrying features are known by construction, allowing attribution rankings to be compared with an external ground truth. The scenarios vary dimensionality, signal sparsity, redundancy/repetition, class imbalance, label noise, class separability, decision-boundary fragmentation, cluster geometry, and feature scale.

## Repository structure

```text
.
├── scripts/
│   ├── download_data.py
│   ├── run_binary_tabular.py
│   ├── run_multiclass_tabular.py
│   └── run_synthetic.py
├── src/
│   ├── tabular_binary.py
│   ├── tabular_multiclass.py
│   └── synthetic_binary.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The datasets are **not redistributed** with the repository.

```bash
python scripts/download_data.py
```

The helper downloads the external UCI/ProPublica datasets and converts them to the local formats expected by the loaders. Breast Cancer Wisconsin and Digits are loaded directly from `scikit-learn`.

## Running the experiments

Real binary datasets:

```bash
python scripts/run_binary_tabular.py
```

Real multiclass datasets:

```bash
python scripts/run_multiclass_tabular.py
```

Synthetic scenarios:

```bash
python scripts/run_synthetic.py
```

Outputs are written to `results/`. The long-running real-data scripts include checkpoint/resume logic: completed configurations are skipped when a run is restarted.

## Experimental protocol implemented here

For real tabular data, each dataset is split into 70% training and 30% test data with stratification. The main experiments compare TreeSHAP, KernelSHAP, and LIME under six representations of unavailable features: default values, uniform sampling, product of marginals, marginal sampling, surrogate models, and separate models.

Deletion and addition progressively remove/add one feature at a time and summarize the target-class probability curve with its AUC. A random-order baseline averages five random feature permutations. Faithfulness correlation samples 20 random feature subsets per instance and computes the Pearson correlation between summed attribution and the corresponding decrease in target-class probability.

The public configuration exposes the two ranking conventions discussed in the chapter (`signed`, `absolute`). Binary experiments additionally include predicted, true, and positive target classes; multiclass experiments use predicted and true classes.

For the 15 synthetic scenarios, attribution rankings can additionally be evaluated against the known signal-carrying feature set using ground-truth ROC-AUC and precision@k.

## Reproducibility notes

Random seeds are fixed in the supplied implementation. Random forests use `n_jobs=1` to avoid parallelization overhead in the many small `predict_proba` calls used by removal-based evaluation. Some configurations, especially Adult Income and the `separate_models` removal strategy, can be computationally expensive.

## Research context and supervision

This work was developed as part of a PhD thesis at the **MIND Laboratory**, Department of Informatics, Systems and Communication (DISCo), **University of Milano-Bicocca**, under the supervision of **Prof. Enza Messina** and **Prof. Mauro Passacantando**.

## Acknowledgements

Special thanks to **Antonio Consolo** for his valuable contribution to this work.

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for the full licence text.
