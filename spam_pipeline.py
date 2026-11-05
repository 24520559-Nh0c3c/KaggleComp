"""Five-fold transformer classifier for the DS102 spam data.

Requires numpy, pandas, scikit-learn, torch, transformers and tqdm. Run with
``python spam_pipeline.py --help`` for configurable paths and hyperparameters.
The optional dev set is evaluated only after each fold has been trained.
"""

import argparse
import json
import random
import unicodedata
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold


LABEL = "spam_label"
DATA_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    train: str = str(DATA_DIR / "ds102_train.csv")
    test: str = str(DATA_DIR / "ds102_test.csv")
    sample: str = str(DATA_DIR / "ds102_samplesubmission.csv")
    dev: str | None = None
    output: str = "spam_output"
    models: list[str] = field(default_factory=lambda: [
        "xlm-roberta-base", "vinai/phobert-base", "bert-base-multilingual-cased"])
    epochs: int = 4
    batch_size: int = 8
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 3
    seed: int = 42


def find_column(frame, candidates):
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"missing columns: expected one of {candidates}")


def load_frame(path, labeled=False):
    frame = pd.read_csv(path, encoding="utf-8-sig", engine="python")
    if frame.empty:
        raise ValueError(f"{path}: empty dataset")
    find_column(frame, ("text", "review", "comment"))
    if labeled:
        label = find_column(frame, ("label", "spam_label"))
        if label != LABEL:
            frame = frame.rename(columns={label: LABEL})
    else:
        identifier = find_column(frame, ("ID", "id"))
        if identifier != "id":
            frame = frame.rename(columns={identifier: "id"})
    if labeled and frame[LABEL].isna().any():
        raise ValueError(f"{path}: missing labels")
    if not labeled and (frame["id"].isna().any() or frame["id"].duplicated().any()):
        raise ValueError(f"{path}: test IDs must be non-null and unique")
    return frame


def make_texts(frame):
    column = find_column(frame, ("text", "review", "comment"))
    return [" ".join("".join(" " if unicodedata.category(char).startswith("C") and char != "\u200d"
                           else char for char in str(value)).split())
            for value in frame[column].fillna("")]


def encode_labels(train, dev=None):
    classes = sorted(train[LABEL].unique().tolist())
    if classes != [0, 1, 2, 3]:
        raise ValueError("Expected four classes with integer labels 0, 1, 2, 3")
    labels = train[LABEL].to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=4)
    if counts.min() < 5:
        raise ValueError("Every class needs at least five training examples")
    if dev is not None and not dev[LABEL].isin(classes).all():
        raise ValueError("Dev contains labels absent from training")
    return classes, labels


def class_weights(labels):
    counts = np.bincount(labels, minlength=4)
    if np.any(counts == 0):
        raise ValueError("Each fold's training split must contain all classes")
    return len(labels) / (4.0 * counts)


def metrics(labels, probabilities):
    predictions = np.asarray(probabilities).argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=list(range(4)), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, predictions, labels=list(range(4)), average="weighted", zero_division=0)),
            "report": classification_report(labels, predictions, labels=list(range(4)),
                                            output_dict=True, zero_division=0)}


def accumulate_probabilities(oof, test_probs, val_indices, fold_oof, fold_test, model_count, fold_count=5):
    oof[val_indices] += fold_oof / model_count
    test_probs += fold_test / (model_count * fold_count)


def optimizer_step(scaler, optimizer, scheduler):
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= previous_scale:
        scheduler.step()


def make_submission(test, sample, predictions):
    if list(sample.columns) != ["ID", LABEL]:
        raise ValueError("Sample submission must contain exactly ID and spam_label")
    if sample["ID"].isna().any() or sample["ID"].duplicated().any():
        raise ValueError("Sample IDs must be non-null and unique")
    if len(predictions) != len(test):
        raise ValueError("Prediction count does not match test rows")
    identifier = find_column(test, ("ID", "id"))
    if test[identifier].isna().any() or test[identifier].duplicated().any():
        raise ValueError("Test IDs must be non-null and unique")
    if set(sample["ID"]) != set(test[identifier]):
        raise ValueError("Sample and test IDs do not match exactly")
    predictions = np.asarray(predictions)
    if not np.isin(predictions, [0, 1, 2, 3]).all():
        raise ValueError("Predictions must be integer labels 0 through 3")
    result = sample.copy()
    result[LABEL] = result["ID"].map(dict(zip(test[identifier], predictions.astype(int))))
    return result


def run(config):
    if min(config.epochs, config.batch_size, config.max_length, config.patience) < 1:
        raise ValueError("epochs, batch size, max length and patience must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Invalid optimizer parameters")
    if not 0 <= config.warmup_ratio <= 1:
        raise ValueError("warmup ratio must be between 0 and 1")
    if not isinstance(config.models, (list, tuple)) or not config.models or any(
            not isinstance(model, str) or not model.strip() for model in config.models):
        raise ValueError("models must be a non-empty list of model names")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("Install torch, transformers and tqdm to train the pipeline") from exc

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = load_frame(config.train, True), load_frame(config.test)
    dev = load_frame(config.dev, True) if config.dev else None
    sample = pd.read_csv(config.sample, encoding="utf-8-sig")
    classes, labels = encode_labels(train, dev)
    # Validate ID correspondence before expensive model downloads/training.
    make_submission(test, sample, np.zeros(len(test), dtype=int))
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    class TextDataset(Dataset):
        def __init__(self, texts, targets=None):
            self.encodings = tokenizer(texts, truncation=True, max_length=config.max_length)
            self.targets = targets

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, index):
            item = {key: value[index] for key, value in self.encodings.items()}
            if self.targets is not None:
                item["labels"] = int(self.targets[index])
            return item

    def collate(items):
        targets = [item.pop("labels") for item in items] if "labels" in items[0] else None
        batch = tokenizer.pad(items, padding=True, return_tensors="pt")
        if targets is not None:
            batch["labels"] = torch.tensor(targets, dtype=torch.long)
        return batch

    def loader(dataset, shuffle=False):
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate)

    def predict(model, dataset):
        model.eval()
        all_probabilities = []
        with torch.no_grad():
            for batch in loader(dataset):
                batch.pop("labels", None)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", enabled=True) if device.type == "cuda" else nullcontext():
                    logits = model(**batch).logits
                all_probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(all_probabilities)

    texts, test_texts = make_texts(train), make_texts(test)
    oof = np.zeros((len(train), 4), dtype=np.float32)
    test_probs = np.zeros((len(test), 4), dtype=np.float32)
    dev_probs = np.zeros((len(dev), 4), dtype=np.float32) if dev is not None else None
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    fold_metrics = []
    for model_index, model_name in enumerate(config.models):
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      test_dataset = TextDataset(test_texts)
      dev_dataset = TextDataset(make_texts(dev)) if dev is not None else None
      for fold, (tr_idx, val_idx) in enumerate(folds.split(texts, labels), 1):
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4).float().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights(labels[tr_idx]), dtype=torch.float32, device=device))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        train_dataset = TextDataset([texts[i] for i in tr_idx], labels[tr_idx])
        val_dataset = TextDataset([texts[i] for i in val_idx])
        train_loader = loader(train_dataset, shuffle=True)
        total_steps = len(train_loader) * config.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * config.warmup_ratio),
            num_training_steps=total_steps)
        checkpoint = output / f"model_{model_index + 1}_fold_{fold}.pt"
        best, stale = -1.0, 0
        for epoch in range(config.epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"{model_name} fold {fold} epoch {epoch + 1}", leave=False):
                targets = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=True) if device.type == "cuda" else nullcontext():
                    loss = criterion(model(**batch).logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_step(scaler, optimizer, scheduler)
            score = metrics(labels[val_idx], predict(model, val_dataset))["macro_f1"]
            print(f"{model_name} fold {fold} epoch {epoch + 1}: validation macro F1 {score:.5f}", flush=True)
            if score > best:
                best, stale = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= config.patience:
                    break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        fold_oof = predict(model, val_dataset)
        accumulate_probabilities(oof, test_probs, val_idx, fold_oof,
                                 predict(model, test_dataset), len(config.models))
        if dev_dataset is not None:
            dev_probs += predict(model, dev_dataset) / (5 * len(config.models))
        fold_metrics.append({"model": model_name, "fold": fold, **metrics(labels[val_idx], fold_oof)})
        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results = {"classes": classes, "folds": fold_metrics, "oof": metrics(labels, oof)}
    if dev is not None:
        dev_labels = dev[LABEL].map({value: i for i, value in enumerate(classes)}).to_numpy(dtype=np.int64)
        results["dev"] = metrics(dev_labels, dev_probs)
        np.save(output / "dev_probabilities.npy", dev_probs)
    np.save(output / "oof_probabilities.npy", oof)
    np.save(output / "test_probabilities.npy", test_probs)
    make_submission(test, sample, test_probs.argmax(axis=1)).to_csv(output / "submission_ensemble.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"OOF macro F1: {results['oof']['macro_f1']:.5f}; outputs: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--models", nargs="+", default=defaults.models)
    for name in ("train", "test", "sample", "dev", "output", "epochs", "batch_size", "max_length", "learning_rate", "weight_decay", "warmup_ratio", "patience", "seed"):
        default = getattr(defaults, name)
        parser.add_argument("--" + name.replace("_", "-"), default=default,
                            type=type(default) if default is not None else str)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()