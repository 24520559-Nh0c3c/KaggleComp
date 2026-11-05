"""Tune XLM class multipliers on one OOF split and evaluate on the other."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from spam_pipeline import LABEL, encode_labels, load_frame, make_submission


def score(labels, predictions):
    return float(f1_score(labels, predictions, labels=[0, 1, 2, 3],
                          average="macro", zero_division=0))


def tune(labels, probabilities):
    # Class 0 is the reference; ties favor the unadjusted prediction.
    candidates = [1.0, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4, 1.6]
    best_weights = np.ones(4)
    best_score = score(labels, probabilities.argmax(axis=1))
    for factors in itertools.product(candidates, repeat=3):
        weights = np.array((1.0, *factors))
        current = score(labels, (probabilities * weights).argmax(axis=1))
        if current > best_score + 1e-12:
            best_score, best_weights = current, weights
    return best_weights


def main():
    root = Path(__file__).resolve().parent
    output = root / "spam_output"
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    train = load_frame(config["train"], True)
    dev = load_frame(config["dev"], True)
    test = load_frame(config["test"])
    sample = pd.read_csv(config["sample"], encoding="utf-8-sig")
    _, labels = encode_labels(train, dev)
    oof, dev_probs, test_probs = [np.load(output / f"xlm_{name}_probabilities.npy")
                                  for name in ("oof", "dev", "test")]
    for probabilities, frame in ((oof, train), (dev_probs, dev), (test_probs, test)):
        if probabilities.shape != (len(frame), 4) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid probability array or row count")
        if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
            raise ValueError("Probabilities must be nonnegative and sum to one")
    calibration, holdout = train_test_split(
        np.arange(len(labels)), test_size=0.5, stratify=labels, random_state=42)
    weights = tune(labels[calibration], oof[calibration])
    results = {"weights": weights.tolist(), "seed": 42}
    for name, targets, probs in (("calibration", labels[calibration], oof[calibration]),
                                 ("holdout", labels[holdout], oof[holdout]),
                                 ("dev", dev[LABEL].to_numpy(), dev_probs)):
        results[name] = {"baseline": score(targets, probs.argmax(axis=1)),
                         "adjusted": score(targets, (probs * weights).argmax(axis=1)),
                         "rows": len(targets)}
    submission = make_submission(test, sample, (test_probs * weights).argmax(axis=1))
    submission.to_csv(output / "submission_xlm_calibrated.csv", index=False)
    (output / "xlm_calibration_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()