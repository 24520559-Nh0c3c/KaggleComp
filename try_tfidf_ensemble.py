"""Evaluate a complementary TF-IDF model against the calibrated XLM baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from calibrate_xlm import score
from spam_pipeline import LABEL, encode_labels, load_frame, make_submission, make_texts


def features(frame):
    comments = make_texts(frame)
    products = frame["product_name"].fillna("").astype(str).tolist() if "product_name" in frame else [""] * len(frame)
    return comments, products


def blend(xlm, other, weights, ratio):
    return ((1 - ratio) * xlm + ratio * other) * weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "spam_output"
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    train = load_frame(config["train"], True)
    dev = load_frame(config["dev"], True)
    test = load_frame(config["test"])
    sample = pd.read_csv(config["sample"], encoding="utf-8-sig")
    _, labels = encode_labels(train, dev)
    weights = np.array(json.loads((output / "xlm_calibration_metrics.json").read_text(encoding="utf-8"))["weights"])
    xlm = [np.load(output / f"xlm_{name}_probabilities.npy") for name in ("oof", "dev", "test")]
    for array, frame in zip(xlm, (train, dev, test)):
        if array.shape != (len(frame), 4) or not np.isfinite(array).all():
            raise ValueError("Invalid XLM probabilities")
    if args.smoke:
        indices = np.concatenate([np.flatnonzero(labels == label)[:20] for label in range(4)])
        train, labels = train.iloc[indices], labels[indices]
        dev, test = dev.iloc[:40], test.iloc[:40]
        xlm = [xlm[0][indices], xlm[1][:40], xlm[2][:40]]

    train_text, train_product = features(train)
    dev_text, dev_product = features(dev)
    test_text, test_product = features(test)
    oof = np.zeros((len(train), 4))
    dev_probs = np.zeros((len(dev), 4))
    test_probs = np.zeros((len(test), 4))
    folds = StratifiedKFold(5, shuffle=True, random_state=config["seed"])
    for fold, (training, validation) in enumerate(folds.split(train_text, labels), 1):
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000)
        char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=150000)
        product = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30000)
        vectors = (word, char, product)
        sources = (train_text, train_text, train_product)
        train_matrix = hstack([v.fit_transform([s[i] for i in training]) for v, s in zip(vectors, sources)]).tocsr()
        model = LogisticRegression(C=4, max_iter=500, class_weight="balanced", random_state=42)
        model.fit(train_matrix, labels[training])

        def predict(text, products):
            matrix = hstack([v.transform(s) for v, s in zip(vectors, (text, text, products))]).tocsr()
            return model.predict_proba(matrix)

        oof[validation] = predict([train_text[i] for i in validation], [train_product[i] for i in validation])
        dev_probs += predict(dev_text, dev_product) / 5
        test_probs += predict(test_text, test_product) / 5
        print(f"Fold {fold}: TF-IDF macro F1 {score(labels[validation], oof[validation].argmax(axis=1)):.5f}", flush=True)

    calibration, holdout = train_test_split(np.arange(len(labels)), test_size=0.5, stratify=labels, random_state=42)
    ratios = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    ratio = max(ratios, key=lambda r: score(labels[calibration], blend(xlm[0][calibration], oof[calibration], weights, r).argmax(axis=1)))
    results = {"ratio": ratio, "weights": weights.tolist(), "smoke": args.smoke}
    for name, targets, base, extra in (("calibration", labels[calibration], xlm[0][calibration], oof[calibration]),
                                        ("holdout", labels[holdout], xlm[0][holdout], oof[holdout]),
                                        ("dev", dev[LABEL].to_numpy(), xlm[1], dev_probs)):
        results[name] = {"baseline": score(targets, (base * weights).argmax(axis=1)),
                         "blended": score(targets, blend(base, extra, weights, ratio).argmax(axis=1)),
                         "tfidf": score(targets, extra.argmax(axis=1))}
    print(json.dumps(results, indent=2), flush=True)
    if not args.smoke:
        (output / "tfidf_blend_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        np.save(output / "tfidf_oof_probabilities.npy", oof)
        np.save(output / "tfidf_dev_probabilities.npy", dev_probs)
        np.save(output / "tfidf_test_probabilities.npy", test_probs)
        if ratio > 0 and results["holdout"]["blended"] > results["holdout"]["baseline"] and results["dev"]["blended"] > results["dev"]["baseline"]:
            submission = make_submission(test, sample, blend(xlm[2], test_probs, weights, ratio).argmax(axis=1))
            submission.to_csv(output / "submission_xlm_tfidf.csv", index=False)
            print("Saved submission_xlm_tfidf.csv", flush=True)


if __name__ == "__main__":
    main()