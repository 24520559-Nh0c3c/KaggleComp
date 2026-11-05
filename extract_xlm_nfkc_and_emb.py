"""Run fast sorted FP16 inference on saved 5-fold XLM-RoBERTa checkpoints (model_3_fold_1..5.pt)
using NFKC-normalized comments, and extract 768-dim penultimate representations.

Saves:
  - spam_output/xlm_nfkc_oof.npy, xlm_nfkc_dev.npy, xlm_nfkc_test.npy
  - spam_output/xlm_emb_mlp_oof.npy, xlm_emb_mlp_dev.npy, xlm_emb_mlp_test.npy
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from spam_pipeline import LABEL, load_frame

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "spam_output"
SEED = 42
BATCH_SIZE = 64


def make_nfkc_texts(frame):
    out = []
    for val in frame["comment"].fillna(""):
        s = unicodedata.normalize("NFKC", str(val))
        s = " ".join("".join(" " if unicodedata.category(ch).startswith("C") and ch != "\u200d" else ch for ch in s).split())
        out.append(s)
    return out


def infer_probs_and_cls(model, encodings, tokenizer, device):
    model.eval()
    n = len(encodings["input_ids"])
    lengths = np.array([len(x) for x in encodings["input_ids"]])
    order = np.argsort(lengths)
    probs = np.zeros((n, 4), dtype=np.float32)
    cls_embs = np.zeros((n, 768), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            b_idx = order[start : start + BATCH_SIZE]
            items = [{"input_ids": encodings["input_ids"][i], "attention_mask": encodings["attention_mask"][i]} for i in b_idx]
            batch = tokenizer.pad(items, padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model.roberta(**batch)
                seq_out = outputs.last_hidden_state
                cls_vec = seq_out[:, 0, :]
                logits = model.classifier(seq_out)
            probs[b_idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            cls_embs[b_idx] = cls_vec.float().cpu().numpy()
    return probs, cls_embs


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")
    labels = train[LABEL].to_numpy()
    dev_labels = dev[LABEL].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", local_files_only=True)
    tr_enc = tokenizer(make_nfkc_texts(train), truncation=True, max_length=256)
    de_enc = tokenizer(make_nfkc_texts(dev), truncation=True, max_length=256)
    te_enc = tokenizer(make_nfkc_texts(test), truncation=True, max_length=256)

    oof_p = np.zeros((len(train), 4), dtype=np.float32)
    dev_p = np.zeros((len(dev), 4), dtype=np.float32)
    test_p = np.zeros((len(test), 4), dtype=np.float32)

    oof_emb = np.zeros((len(train), 768), dtype=np.float32)
    dev_emb = np.zeros((len(dev), 768), dtype=np.float32)
    test_emb = np.zeros((len(test), 768), dtype=np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=4, local_files_only=True).to(device)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, labels), 1):
        ckpt = OUTPUT / f"model_3_fold_{fold}.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

        val_enc = {
            "input_ids": [tr_enc["input_ids"][i] for i in val_idx],
            "attention_mask": [tr_enc["attention_mask"][i] for i in val_idx],
        }
        vp, ve = infer_probs_and_cls(model, val_enc, tokenizer, device)
        dp, de = infer_probs_and_cls(model, de_enc, tokenizer, device)
        tp, te = infer_probs_and_cls(model, te_enc, tokenizer, device)

        oof_p[val_idx] = vp
        oof_emb[val_idx] = ve
        dev_p += dp / 5.0
        dev_emb += de / 5.0
        test_p += tp / 5.0
        test_emb += te / 5.0
        print(f"Fold {fold} inferred ({time.time() - t0:.1f}s)", flush=True)

    old_oof = np.load(OUTPUT / "xlm_oof_probabilities.npy")
    old_dev = np.load(OUTPUT / "xlm_dev_probabilities.npy")
    print(f"Old XLM OOF F1:  {f1_score(labels, old_oof.argmax(axis=1), average='macro'):.5f} | DEV F1: {f1_score(dev_labels, old_dev.argmax(axis=1), average='macro'):.5f}")
    print(f"NFKC XLM OOF F1: {f1_score(labels, oof_p.argmax(axis=1), average='macro'):.5f} | DEV F1: {f1_score(dev_labels, dev_p.argmax(axis=1), average='macro'):.5f}")

    # Compute out-of-fold cosine similarity to class centroids in XLM 768-d embedding space!
    # Normalize embeddings
    oof_norm = oof_emb / np.maximum(np.linalg.norm(oof_emb, axis=1, keepdims=True), 1e-8)
    dev_norm = dev_emb / np.maximum(np.linalg.norm(dev_emb, axis=1, keepdims=True), 1e-8)
    test_norm = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-8)

    cos_cent_oof = np.zeros((len(train), 4), dtype=np.float32)
    cos_cent_dev = np.zeros((len(dev), 4), dtype=np.float32)
    cos_cent_test = np.zeros((len(test), 4), dtype=np.float32)

    emb_lr_oof = np.zeros((len(train), 4), dtype=np.float32)
    emb_lr_dev = np.zeros((len(dev), 4), dtype=np.float32)
    emb_lr_test = np.zeros((len(test), 4), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, labels), 1):
        for c in range(4):
            c_mask = tr_idx[labels[tr_idx] == c]
            cent = oof_norm[c_mask].mean(axis=0)
            cent /= max(np.linalg.norm(cent), 1e-8)
            cos_cent_oof[val_idx, c] = oof_norm[val_idx] @ cent
            cos_cent_dev[:, c] += (dev_norm @ cent) / 5.0
            cos_cent_test[:, c] += (test_norm @ cent) / 5.0

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(oof_norm[tr_idx])
        X_va_s = scaler.transform(oof_norm[val_idx])
        X_de_s = scaler.transform(dev_norm)
        X_te_s = scaler.transform(test_norm)

        clf = LogisticRegression(C=0.1, max_iter=400, random_state=SEED + fold)
        clf.fit(X_tr_s, labels[tr_idx])
        emb_lr_oof[val_idx] = clf.predict_proba(X_va_s)
        emb_lr_dev += clf.predict_proba(X_de_s) / 5.0
        emb_lr_test += clf.predict_proba(X_te_s) / 5.0

    print(f"Calibrated Emb-LR OOF F1: {f1_score(labels, emb_lr_oof.argmax(axis=1), average='macro'):.5f} | DEV F1: {f1_score(dev_labels, emb_lr_dev.argmax(axis=1), average='macro'):.5f}")

    np.save(OUTPUT / "xlm_nfkc_oof.npy", oof_p)
    np.save(OUTPUT / "xlm_nfkc_dev.npy", dev_p)
    np.save(OUTPUT / "xlm_nfkc_test.npy", test_p)

    np.savez_compressed(
        OUTPUT / "xlm_cls_features.npz",
        cos_cent_oof=cos_cent_oof,
        cos_cent_dev=cos_cent_dev,
        cos_cent_test=cos_cent_test,
        emb_lr_oof=emb_lr_oof,
        emb_lr_dev=emb_lr_dev,
        emb_lr_test=emb_lr_test,
    )
    print(f"Saved xlm_nfkc probabilities and xlm_cls_features.npz in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
