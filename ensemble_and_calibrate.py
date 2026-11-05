"""Produce submission_10 by ensembling XLM-RoBERTa and DeBERTa-v3 with calibrated thresholds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from spam_pipeline import LABEL, load_frame, make_submission


def main():
    root = Path(__file__).resolve().parent
    output = root / "spam_output"
    
    dev = load_frame(root / "ds102_dev.csv", labeled=True)
    test = load_frame(root / "ds102_test.csv")
    sample = pd.read_csv(root / "ds102_samplesubmission.csv", encoding="utf-8-sig")
    dev_labels = dev[LABEL].to_numpy()

    # Load 5-fold probability arrays
    xlm_dev = np.load(output / "xlm_dev_probabilities.npy")
    xlm_test = np.load(output / "xlm_test_probabilities.npy")
    deberta_dev = np.load(output / "deberta_dev_probabilities.npy")
    deberta_test = np.load(output / "deberta_test_probabilities.npy")

    # Optimal ensemble weights: 87% XLM + 13% DeBERTa
    alpha = 0.13
    blended_dev = (1.0 - alpha) * xlm_dev + alpha * deberta_dev
    blended_test = (1.0 - alpha) * xlm_test + alpha * deberta_test

    # Calibration multipliers: preserve class 1 recall while boosting class 3 and 2
    weights = np.array([1.0, 0.9, 1.1, 1.6])
    calibrated_dev_probs = blended_dev * weights
    calibrated_test_probs = blended_test * weights

    dev_preds = calibrated_dev_probs.argmax(axis=1)
    test_preds = calibrated_test_probs.argmax(axis=1)

    # Metrics on DEV
    macro_f1 = float(f1_score(dev_labels, dev_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(dev_labels, dev_preds, average="weighted", zero_division=0))
    report = classification_report(dev_labels, dev_preds, output_dict=True, zero_division=0)

    print(f"DEV Macro F1: {macro_f1:.5f} (weighted F1: {weighted_f1:.5f})")
    for c in range(4):
        c_f1 = report[str(c)]["f1-score"]
        print(f"  Class {c} F1: {c_f1:.5f} (support: {report[str(c)]['support']})")

    # Save submission_10.csv
    submission_10 = make_submission(test, sample, test_preds)
    sub10_path = output / "submission_10.csv"
    submission_10.to_csv(sub10_path, index=False)
    print(f"Saved {sub10_path}")

    # Metrics log
    metrics_data = {
        "ensemble": "0.87 * XLM-RoBERTa + 0.13 * DeBERTa-v3",
        "weights": weights.tolist(),
        "dev": {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report": report
        },
        "test_distribution": pd.Series(test_preds).value_counts().to_dict()
    }
    (output / "submission_10_metrics.json").write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print(f"Test class distribution: {metrics_data['test_distribution']}")


if __name__ == "__main__":
    main()
