"""5-Fold Context-Aware PhoBERT-base Classifier for DS102 Spam Detection.

Unlike the comment-only baseline transformers, this model feeds native Vietnamese PhoBERT:
  1. NFKC-normalized comment (fixing NFD diacritics & stylized spam evasion fonts)
     with line breaks preserved as ' // '
  2. Rating + Category path + Product name (catching wrong-product & e-voucher spam)
  3. Best-matching 35-word window from NFKC-normalized product description
     (catching verbatim or partial description copy-paste anywhere in long descriptions)

Uses noisy length-bucketed mini-batches + FP16 mixed precision for fast training on 4GB VRAM,
strictly aligned on StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from spam_pipeline import LABEL, load_frame

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "spam_output"
MODEL_NAME = "vinai/phobert-base"
SEED = 42
MAX_LEN = 254
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
EPOCHS = 3
LR = 2.5e-5


def norm_nfkc(s):
    return unicodedata.normalize("NFKC", str(s))


def best_desc_window(c_str, d_str, win_words=35, step=10, max_chars=170):
    d_clean = " ".join(norm_nfkc(d_str).split())
    if not d_clean:
        return ""
    d_words = d_clean.split()
    if len(d_words) <= win_words:
        return d_clean[:max_chars]
    c_words = set(norm_nfkc(c_str).lower().split())
    if not c_words:
        return d_clean[:max_chars]

    best_idx = 0
    best_hits = 0
    for i in range(0, len(d_words) - win_words + 1, step):
        w_slice = d_words[i : i + win_words]
        hits = sum(1 for w in set(x.lower() for x in w_slice) if len(w) > 2 and w in c_words)
        if hits > best_hits:
            best_hits = hits
            best_idx = i
    if best_hits <= 2:
        return d_clean[:max_chars]
    return " ".join(d_words[best_idx : best_idx + win_words])[:max_chars]


def build_context_texts(df):
    out = []
    ratings = df["rating"].fillna(5).astype(int).tolist()
    cats = df["categories"].fillna("").tolist()
    prods = df["product_name"].fillna("").tolist()
    comms = df["comment"].fillna("").tolist()
    descs = df["description"].fillna("").tolist()

    for r, cat, p, c, d in zip(ratings, cats, prods, comms, descs):
        c_norm = norm_nfkc(c).replace("\r\n", "\n").replace("\r", "\n")
        c_norm = re.sub(r"\n+", " // ", c_norm)
        c_norm = " ".join(c_norm.split())[:420]

        cat_norm = re.sub(r"[\[\]']", "", norm_nfkc(cat))
        cat_norm = " ".join(cat_norm.split())[:65]
        p_norm = " ".join(norm_nfkc(p).split())[:105]
        d_win = best_desc_window(c, d, win_words=35, step=10, max_chars=160)

        text = f"{c_norm} </s> {r} sao | {cat_norm} | SP: {p_norm} </s> Mô tả: {d_win}"
        out.append(text)
    return out


def make_noisy_length_batches(indices, lengths, batch_size, rng):
    noise = rng.integers(-12, 13, size=len(indices))
    order = np.argsort(lengths[indices] + noise)
    sorted_idx = indices[order]
    batches = [sorted_idx[i : i + batch_size] for i in range(0, len(sorted_idx), batch_size)]
    rng.shuffle(batches)
    return batches


def predict_sorted(model, encodings, tokenizer, device):
    model.eval()
    n = len(encodings["input_ids"])
    lengths = np.array([len(x) for x in encodings["input_ids"]])
    order = np.argsort(lengths)
    probs = np.zeros((n, 4), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH_SIZE):
            b_idx = order[start : start + EVAL_BATCH_SIZE]
            items = [{"input_ids": encodings["input_ids"][i], "attention_mask": encodings["attention_mask"][i]} for i in b_idx]
            batch = tokenizer.pad(items, padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits
            probs[b_idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def best_calibrated_f1(y_true, probs):
    best = f1_score(y_true, probs.argmax(axis=1), average="macro")
    for w1 in (0.7, 0.85, 1.0, 1.15, 1.3):
        for w2 in (0.85, 1.0, 1.15):
            w = np.array([1.0, w1, w2, 1.0])
            s = f1_score(y_true, (probs * w).argmax(axis=1), average="macro")
            if s > best:
                best = s
    return best


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")
    labels = train[LABEL].to_numpy(dtype=np.int64)
    dev_labels = dev[LABEL].to_numpy(dtype=np.int64)

    print("Building context-aware NFKC inputs...")
    tr_texts = build_context_texts(train)
    de_texts = build_context_texts(dev)
    te_texts = build_context_texts(test)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tr_enc = tokenizer(tr_texts, truncation=True, max_length=MAX_LEN)
    de_enc = tokenizer(de_texts, truncation=True, max_length=MAX_LEN)
    te_enc = tokenizer(te_texts, truncation=True, max_length=MAX_LEN)
    tr_lengths = np.array([len(x) for x in tr_enc["input_ids"]])
    print(f"Tokenized: mean_len={tr_lengths.mean():.1f}, max_len={tr_lengths.max()}")

    oof_probs = np.zeros((len(train), 4), dtype=np.float32)
    dev_probs = np.zeros((len(dev), 4), dtype=np.float32)
    test_probs = np.zeros((len(test), 4), dtype=np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cls_weights = torch.tensor([1.0, 4.5, 2.0, 1.3], dtype=torch.float32, device=device)

    t_start = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(tr_texts, labels), 1):
        f_t0 = time.time()
        rng = np.random.default_rng(SEED + fold * 100)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=4, local_files_only=True
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.02)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        steps_per_epoch = (len(tr_idx) + BATCH_SIZE - 1) // BATCH_SIZE
        total_steps = steps_per_epoch * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        val_enc = {
            "input_ids": [tr_enc["input_ids"][i] for i in val_idx],
            "attention_mask": [tr_enc["attention_mask"][i] for i in val_idx],
        }
        ckpt_path = OUTPUT / f"phobert_ctx_fold_{fold}.pt"
        best_score = -1.0

        for epoch in range(1, EPOCHS + 1):
            ep_t0 = time.time()
            model.train()
            batches = make_noisy_length_batches(tr_idx, tr_lengths, BATCH_SIZE, rng)
            total_loss = 0.0

            for b_idx in batches:
                items = [{"input_ids": tr_enc["input_ids"][i], "attention_mask": tr_enc["attention_mask"][i]} for i in b_idx]
                targets = torch.tensor(labels[b_idx], dtype=torch.long, device=device)
                batch = tokenizer.pad(items, padding=True, return_tensors="pt")
                batch = {k: v.to(device) for k, v in batch.items()}

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    logits = model(**batch).logits
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= prev_scale:
                    scheduler.step()
                total_loss += loss.item()

            val_p = predict_sorted(model, val_enc, tokenizer, device)
            raw_f1 = f1_score(labels[val_idx], val_p.argmax(axis=1), average="macro")
            cal_f1 = best_calibrated_f1(labels[val_idx], val_p)
            ep_dt = time.time() - ep_t0
            print(f"[Fold {fold} | Ep {epoch}/{EPOCHS}] loss={total_loss/len(batches):.4f} | raw_F1={raw_f1:.5f} | cal_F1={cal_f1:.5f} | {ep_dt:.1f}s", flush=True)

            if cal_f1 > best_score:
                best_score = cal_f1
                torch.save(model.state_dict(), ckpt_path)

        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        oof_probs[val_idx] = predict_sorted(model, val_enc, tokenizer, device)
        dev_probs += predict_sorted(model, de_enc, tokenizer, device) / 5.0
        test_probs += predict_sorted(model, te_enc, tokenizer, device) / 5.0

        print(f"--> Finished Fold {fold} in {time.time() - f_t0:.1f}s (Best cal_F1={best_score:.5f})", flush=True)
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    oof_raw = f1_score(labels, oof_probs.argmax(axis=1), average="macro")
    dev_raw = f1_score(dev_labels, dev_probs.argmax(axis=1), average="macro")
    print("\n" + "=" * 65)
    print(f"Context-Aware PhoBERT Total Time: {(time.time() - t_start)/60:.2f} min")
    print(f"OOF Raw Macro F1: {oof_raw:.5f} | Calibrated OOF F1: {best_calibrated_f1(labels, oof_probs):.5f}")
    print(f"DEV Raw Macro F1: {dev_raw:.5f} | Calibrated DEV F1: {best_calibrated_f1(dev_labels, dev_probs):.5f}")
    print("=" * 65)
    print("\nClassification Report on DEV (Raw):")
    print(classification_report(dev_labels, dev_probs.argmax(axis=1), digits=4))

    np.save(OUTPUT / "phobert_ctx_oof.npy", oof_probs)
    np.save(OUTPUT / "phobert_ctx_dev.npy", dev_probs)
    np.save(OUTPUT / "phobert_ctx_test.npy", test_probs)
    print("Saved phobert_ctx_{oof,dev,test}.npy to spam_output/")


if __name__ == "__main__":
    main()
