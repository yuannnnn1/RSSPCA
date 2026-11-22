"""Run a paired PCA/RSSPCA recognition experiment on the authors' ORL MAT file."""

import argparse
import csv
import hashlib
import itertools
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.io import loadmat

from rsspca import fit_rsspca, grid_laplacian, pca_components

ETA1 = (1e-3, 1e-1, 1e1)
ETA2 = (1e-1, 1e1, 1e3)


def load_orl(path):
    data = loadmat(path, variable_names=["x", "label"])
    if "x" not in data or "label" not in data:
        raise ValueError("Expected the authors' ORL MAT variables x and label.")
    images = np.asarray(data["x"], dtype=np.float64)
    raw_labels = data["label"].ravel()
    if images.ndim != 3 or images.shape[2] != 400 or raw_labels.size != 400:
        raise ValueError("Expected 400 images in height × width × 400 layout and 400 labels.")
    if not np.isfinite(images).all() or not np.isfinite(raw_labels).all():
        raise ValueError("Images and labels must be finite.")
    labels, counts = np.unique(raw_labels, return_counts=True)
    if labels.size != 40 or not np.all(counts == 10):
        raise ValueError("Expected 40 subjects with 10 images each.")
    height, width, count = images.shape
    return images.transpose(2, 0, 1).reshape(count, -1), raw_labels, (height, width)


def split_by_subject(labels, indices, train_count, rng):
    training, testing = [], []
    for subject in np.unique(labels[indices]):
        order = rng.permutation(indices[labels[indices] == subject])
        training.extend(order[:train_count])
        testing.extend(order[train_count:])
    return np.array(training), np.array(testing)


def standardized_split(samples, training, evaluation):
    train = samples[training]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1
    return (train - mean) / scale, (samples[evaluation] - mean) / scale


def accuracy_curve(training, evaluation, train_labels, eval_labels, components):
    train_scores = np.einsum("ij,jk->ik", training, components)
    eval_scores = np.einsum("ij,jk->ik", evaluation, components)
    distances = np.zeros((len(evaluation), len(training)))
    accuracy = []
    for k in range(components.shape[1]):
        distances += (eval_scores[:, k, None] - train_scores[None, :, k]) ** 2
        predicted = train_labels[np.argmin(distances, axis=1)]
        accuracy.append(float(np.mean(predicted == eval_labels)))
    return accuracy


def tune(samples, labels, training, rng, laplacian, dimensions):
    fit_indices, val_indices = split_by_subject(labels, training, 5, rng)
    fit, val = standardized_split(samples, fit_indices, val_indices)
    initial = pca_components(fit, max(dimensions))
    trials = []
    for eta1, eta2 in itertools.product(ETA1, ETA2):
        result = fit_rsspca(fit, laplacian, initial, eta1, eta2)
        curve = accuracy_curve(fit, val, labels[fit_indices], labels[val_indices], result.components)
        score = float(np.mean([curve[k - 1] for k in dimensions]))
        trials.append({"eta1": eta1, "eta2": eta2, "validation_accuracy": score})
    winner = min(trials, key=lambda trial: (-trial["validation_accuracy"], trial["eta1"], trial["eta2"]))
    return winner, trials, fit_indices, val_indices


def run_split(samples, labels, laplacian, seed, count):
    rng = np.random.default_rng(seed)
    training, testing = split_by_subject(labels, np.arange(len(labels)), 7, rng)
    dimensions = [k for k in (5, 10, 20, 30) if k <= count] or [count]
    winner, trials, inner_train, validation = tune(samples, labels, training, rng, laplacian, dimensions)
    train, test = standardized_split(samples, training, testing)

    started = perf_counter()
    pca = pca_components(train, count)
    pca_seconds = perf_counter() - started
    started = perf_counter()
    robust = fit_rsspca(train, laplacian, pca, winner["eta1"], winner["eta2"])
    rsspca_seconds = pca_seconds + perf_counter() - started

    rows = []
    for name, components in (("PCA", pca), ("RSSPCA", robust.components)):
        curve = accuracy_curve(train, test, labels[training], labels[testing], components)
        rows.extend(
            {"seed": seed, "method": name, "components": k, "accuracy": score}
            for k, score in enumerate(curve, 1)
        )
    details = {
        "seed": seed,
        "training_indices": training.tolist(),
        "test_indices": testing.tolist(),
        "inner_training_indices": inner_train.tolist(),
        "validation_indices": validation.tolist(),
        "validation_dimensions": dimensions,
        "hyperparameter_trials": trials,
        "selected_parameters": winner,
        "fit_seconds": {"PCA": pca_seconds, "RSSPCA": rsspca_seconds},
        "rsspca_iterations": robust.iterations,
        "rsspca_converged": robust.converged,
    }
    return rows, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the authors' ORL.mat")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--components", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if not 1 <= args.components <= 30:
        parser.error("--components must be between 1 and 30")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")

    samples, labels, shape = load_orl(args.dataset)
    laplacian = grid_laplacian(*shape)
    metadata = {
        "dataset": args.dataset.name,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "image_shape": shape,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "runs": [],
    }
    rows = []
    for seed in args.seeds:
        print(f"Running seed {seed}", flush=True)
        scores, details = run_split(samples, labels, laplacian, seed, args.components)
        rows.extend(scores)
        metadata["runs"].append(details)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "accuracy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "method", "components", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "runs.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved accuracy.csv and runs.json to {args.output}")


if __name__ == "__main__":
    main()
