"""Evaluate the saved XLM-R folds without retraining or overwriting the ensemble."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm

from spam_pipeline import LABEL, encode_labels, load_frame, make_submission, make_texts, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "spam_output"
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    model_index = config["models"].index("xlm-roberta-base") + 1
    train = load_frame(config["train"], True)
    dev = load_frame(config["dev"], True)
    test = load_frame(config["test"])
    sample = pd.read_csv(config["sample"], encoding="utf-8-sig")
    _, labels = encode_labels(train, dev)
    train_text, dev_text, test_text = map(make_texts, (train, dev, test))
    splits = list(StratifiedKFold(5, shuffle=True, random_state=config["seed"]).split(train_text, labels))
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(model, texts):
        encoded = tokenizer(texts, truncation=True, max_length=config["max_length"])
        items = [{key: value[i] for key, value in encoded.items()} for i in range(len(texts))]
        loader = DataLoader(items, batch_size=config["batch_size"],
                            collate_fn=lambda batch: tokenizer.pad(batch, return_tensors="pt"))
        result = []
        with torch.inference_mode():
            for batch in tqdm(loader, leave=False):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext():
                    logits = model(**batch).logits
                result.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(result)

    oof = np.zeros((len(train), 4), dtype=np.float32)
    dev_probs = np.zeros((len(dev), 4), dtype=np.float32)
    test_probs = np.zeros((len(test), 4), dtype=np.float32)
    for fold, (_, indices) in enumerate(splits, 1):
        model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base", num_labels=4, local_files_only=True).float().to(device)
        model.load_state_dict(torch.load(output / f"model_{model_index}_fold_{fold}.pt",
                                         map_location=device, weights_only=True))
        model.eval()
        if args.smoke:
            probs = predict(model, [train_text[i] for i in indices[:8]])
            assert probs.shape == (8, 4) and np.isfinite(probs).all()
            assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
            print("Smoke test passed", flush=True)
            return
        oof[indices] = predict(model, [train_text[i] for i in indices])
        dev_probs += predict(model, dev_text) / 5
        test_probs += predict(model, test_text) / 5
        print(f"Fold {fold}: OOF macro F1 {metrics(labels[indices], oof[indices])['macro_f1']:.5f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = {"oof": metrics(labels, oof), "dev": metrics(dev[LABEL].to_numpy(), dev_probs)}
    submission = make_submission(test, sample, test_probs.argmax(axis=1))
    assert submission.notna().all().all() and len(submission) == len(sample)
    np.save(output / "xlm_oof_probabilities.npy", oof)
    np.save(output / "xlm_dev_probabilities.npy", dev_probs)
    np.save(output / "xlm_test_probabilities.npy", test_probs)
    submission.to_csv(output / "submission_xlm.csv", index=False)
    (output / "xlm_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"XLM OOF macro F1: {results['oof']['macro_f1']:.5f}; dev macro F1: {results['dev']['macro_f1']:.5f}", flush=True)


if __name__ == "__main__":
    main()