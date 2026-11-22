# RSSPCA ORL face recognition

Python implementation of Wang et al. (2025), evaluated against standard PCA with the same train/test splits and Euclidean 1-nearest-neighbor classifier.

## Run

Use Python 3.9 or newer. Download the authors' [ORL.mat](https://github.com/yuzhounh/RSSPCA/blob/main/data/ORL.mat) from the upstream repository. The original ORL images are credited to AT&T Laboratories Cambridge.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py /path/to/ORL.mat
```

The default run uses seeds 0, 1 and 2 and evaluates 1–30 components. To check the pipeline with a smaller run:

```sh
python experiment.py /path/to/ORL.mat --seeds 0 --components 2 --output smoke-results
```

The output directory contains `accuracy.csv` (test accuracy by split, method and component count) and `runs.json` (split indices, validation trials, chosen parameters, fit times, convergence diagnostics, versions and dataset checksum). Accuracy is recorded as a fraction. The RSSPCA fit time includes computing the standard PCA directions used for initialization and excludes hyperparameter search.

[Original paper](https://doi.org/10.1371/journal.pone.0323281) · [Authors' MATLAB implementation](https://github.com/yuzhounh/RSSPCA)
