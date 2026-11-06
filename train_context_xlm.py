"""5-Fold Context-Aware XLM-RoBERTa-base with [CLS]+Mean Pooling & Multi-Sample Dropout.

Trained on Synchronized Stratified 5-Fold across Train (14,306) + Dev (1,590) = 15,896 labeled rows:
  - Fold k trains on 80% Train + 80% Dev (12,717 human-labeled rows per fold, +11.1% more data).
  - Predicts 100% leak-free OOF on val_train (aligned with seed=42!) -> xlm_ctx_oof.npy
  - Predicts 100% leak-free OOF on val_dev -> xlm_ctx_dev.npy
  - Predicts 5-fold ensemble average on Test (trained on 100% of Train+Dev!) -> xlm_ctx_test.npy

Architecture upgrades over comment-only baseline XLM-RoBERTa:
  1. Rich structured NFKC input: Comment (preserving '//' newlines & emojis) + Rating + NumSold +
     Category breadcrumb + Product Name + Best-matching 35-word Description Window.
  2. Concatenated [CLS] + Attention-Masked Mean Pooling (1536-d) + LayerNorm + 5x Multi-Sample Dropout.
  3. Dampened Class-Weighted Cross-Entropy ([1.0, 4.5, 2.0, 1.3], label_smoothing=0.02).
"""

import argparse
import sys
sys.stdout.reconfigure(encoding="utf-8")
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from spam_pipeline import LABEL, load_frame

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "spam_output"
MODEL_NAME = "xlm-roberta-base"
SEED = 42
MAX_LEN = 256
BATCH_SIZE = 8
GRAD_ACCUM = 2
EVAL_BATCH_SIZE = 64
EPOCHS = 3
LR = 2.5e-5


def norm_nfkc(s):
    return unicodedata.normalize("NFKC", str(s))


def best_desc_window(c_str, d_str, win_words=35, step=10, max_chars=175):
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


def build_context_texts(df, variant="a"):
    if variant == "b":
        return build_context_texts_b(df)
    out = []
    ratings = df["rating"].fillna(5).astype(int).tolist()
    sold = df["num_sold"].fillna(0).astype(int).tolist()
    cats = df["categories"].fillna("").tolist()
    prods = df["product_name"].fillna("").tolist()
    comms = df["comment"].fillna("").tolist()
    descs = df["description"].fillna("").tolist()

    for r, s_cnt, cat, p, c, d in zip(ratings, sold, cats, prods, comms, descs):
        c_norm = norm_nfkc(c).replace("\r\n", "\n").replace("\r", "\n")
        c_norm = re.sub(r"\n+", " // ", c_norm)
        c_norm = " ".join(c_norm.split())[:440]

        cat_norm = re.sub(r"[\[\]']", "", norm_nfkc(cat))
        cat_norm = " ".join(cat_norm.split())[:70]
        p_norm = " ".join(norm_nfkc(p).split())[:110]
        d_win = best_desc_window(c, d, win_words=35, step=10, max_chars=170)

        text = f"{c_norm} </s> {r} sao | Đã bán {s_cnt} | {cat_norm} | SP: {p_norm} </s> Mô tả: {d_win}"
        out.append(text)
    return out


def build_context_texts_b(df):
    """Variant B: product-first ordering + longer (50-word) description window, for ensemble diversity."""
    out = []
    ratings = df["rating"].fillna(5).astype(int).tolist()
    cats = df["categories"].fillna("").tolist()
    prods = df["product_name"].fillna("").tolist()
    comms = df["comment"].fillna("").tolist()
    descs = df["description"].fillna("").tolist()

    for r, cat, p, c, d in zip(ratings, cats, prods, comms, descs):
        c_norm = norm_nfkc(c).replace("\r\n", "\n").replace("\r", "\n")
        c_norm = re.sub(r"\n+", " // ", c_norm)
        c_norm = " ".join(c_norm.split())[:400]
        cat_parts = [x.strip() for x in re.sub(r"[\[\]']", "", norm_nfkc(cat)).split(",") if x.strip()]
        cat_norm = " > ".join(cat_parts[-2:])[:50]
        p_norm = " ".join(norm_nfkc(p).split())[:90]
        d_win = best_desc_window(c, d, win_words=50, step=10, max_chars=260)
        out.append(f"{p_norm} | {cat_norm} | {r} sao </s> {c_norm} </s> {d_win}")
    return out


def keep_awake(enable=True):
    """Ask Windows not to sleep while training (documented SetThreadExecutionState API; auto-reverts on exit)."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if enable else ES_CONTINUOUS
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


class XLMRobertaContextClassifier(nn.Module):
    def __init__(self, model_name, num_labels=4):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.add_pooling_layer = False
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg)
        # Freeze 250k multilingual token embeddings (saves 2.3GB VRAM, prevents PCIe spilling & rare-token overfitting!)
        self.backbone.embeddings.word_embeddings.requires_grad_(False)
        h = cfg.hidden_size
        self.norm = nn.LayerNorm(h * 2)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])
        self.fc = nn.Linear(h * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_vec = seq_out[:, 0, :]
        mask_f = attention_mask.unsqueeze(-1).type_as(seq_out)
        mean_vec = (seq_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        feat = self.norm(torch.cat([cls_vec, mean_vec], dim=-1))
        logits = sum(self.fc(drop(feat)) for drop in self.dropouts) / len(self.dropouts)
        return logits


def make_noisy_length_batches(indices, lengths, batch_size, rng):
    noise = rng.integers(-14, 15, size=len(indices))
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
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variant", choices=["a", "b"], default="a")
    parser.add_argument("--tag", default="xlm_ctx", help="output prefix, e.g. xlm_ctx2")
    args = parser.parse_args()
    seed = args.seed
    tag = args.tag
    keep_awake(True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Smoke: {args.smoke}")

    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")

    if args.smoke:
        train = train.iloc[:320].reset_index(drop=True)
        dev = dev.iloc[:80].reset_index(drop=True)
        test = test.iloc[:80].reset_index(drop=True)

    labels_tr = train[LABEL].to_numpy(dtype=np.int64)
    labels_de = dev[LABEL].to_numpy(dtype=np.int64)
    n_tr = len(train)
    n_de = len(dev)

    print("Building context-aware NFKC inputs for Train, Dev, and Test...")
    tr_texts = build_context_texts(train, args.variant)
    de_texts = build_context_texts(dev, args.variant)
    te_texts = build_context_texts(test, args.variant)
    print("Example input:", tr_texts[0][:300])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_texts = tr_texts + de_texts
    full_labels = np.concatenate([labels_tr, labels_de])

    full_enc = tokenizer(full_texts, truncation=True, max_length=MAX_LEN)
    te_enc = tokenizer(te_texts, truncation=True, max_length=MAX_LEN)
    full_lengths = np.array([len(x) for x in full_enc["input_ids"]])
    print(f"Tokenized {len(full_texts)} train+dev texts: mean_len={full_lengths.mean():.1f}, max_len={full_lengths.max()}")

    oof_tr = np.zeros((n_tr, 4), dtype=np.float32)
    oof_de = np.zeros((n_de, 4), dtype=np.float32)
    test_probs = np.zeros((len(test), 4), dtype=np.float32)

    # Synchronized Stratified 5-Fold across Train (exact seed=42!) and Dev
    skf_tr = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(tr_texts, labels_tr))
    skf_de = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(de_texts, labels_de))

    cls_weights = torch.tensor([1.0, 4.5, 2.0, 1.3], dtype=torch.float32, device=device)
    n_folds = 1 if args.smoke else 5
    n_epochs = 1 if args.smoke else EPOCHS

    t_start = time.time()
    for fold in range(1, n_folds + 1):
        f_t0 = time.time()
        tr_idx_a, val_idx_a = skf_tr[fold - 1]
        tr_idx_b, val_idx_b = skf_de[fold - 1]

        comb_tr_idx = np.concatenate([tr_idx_a, n_tr + tr_idx_b])
        comb_val_idx = np.concatenate([val_idx_a, n_tr + val_idx_b])

        rng = np.random.default_rng(seed + fold * 100)
        model = XLMRobertaContextClassifier(MODEL_NAME, num_labels=4).to(device)

        # Differential learning rate over trainable parameters only
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = list(model.norm.parameters()) + list(model.fc.parameters())
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": LR, "weight_decay": 0.01},
            {"params": head_params, "lr": LR * 3.0, "weight_decay": 0.01},
        ])
        criterion = torch.nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.02)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        batches_per_epoch = (len(comb_tr_idx) + BATCH_SIZE - 1) // BATCH_SIZE
        steps_per_epoch = (batches_per_epoch + GRAD_ACCUM - 1) // GRAD_ACCUM
        total_steps = steps_per_epoch * n_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.08 * total_steps), num_training_steps=total_steps
        )

        val_enc = {
            "input_ids": [full_enc["input_ids"][i] for i in comb_val_idx],
            "attention_mask": [full_enc["attention_mask"][i] for i in comb_val_idx],
        }
        ckpt_path = OUTPUT / f"{tag}_fold_{fold}.pt"
        best_score = -1.0

        for epoch in range(1, n_epochs + 1):
            ep_t0 = time.time()
            model.train()
            batches = make_noisy_length_batches(comb_tr_idx, full_lengths, BATCH_SIZE, rng)
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step_i, b_idx in enumerate(batches, 1):
                items = [{"input_ids": full_enc["input_ids"][i], "attention_mask": full_enc["attention_mask"][i]} for i in b_idx]
                targets = torch.tensor(full_labels[b_idx], dtype=torch.long, device=device)
                batch = tokenizer.pad(items, padding=True, return_tensors="pt")
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = criterion(logits, targets) / GRAD_ACCUM
                scaler.scale(loss).backward()
                total_loss += loss.item() * GRAD_ACCUM

                if step_i % GRAD_ACCUM == 0 or step_i == len(batches):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    prev_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= prev_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            val_p = predict_sorted(model, val_enc, tokenizer, device)
            raw_f1 = f1_score(full_labels[comb_val_idx], val_p.argmax(axis=1), average="macro")
            cal_f1 = best_calibrated_f1(full_labels[comb_val_idx], val_p)
            ep_dt = time.time() - ep_t0
            max_vram = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
            print(f"[Fold {fold} | Ep {epoch}/{n_epochs}] loss={total_loss/len(batches):.4f} | raw_F1={raw_f1:.5f} | cal_F1={cal_f1:.5f} | VRAM={max_vram:.2f}GB | {ep_dt:.1f}s", flush=True)

            if cal_f1 > best_score:
                best_score = cal_f1
                if not args.smoke:
                    torch.save(model.state_dict(), ckpt_path)

        if not args.smoke:
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        val_best_p = predict_sorted(model, val_enc, tokenizer, device)
        oof_tr[val_idx_a] = val_best_p[: len(val_idx_a)]
        oof_de[val_idx_b] = val_best_p[len(val_idx_a) :]
        test_probs += predict_sorted(model, te_enc, tokenizer, device) / n_folds

        print(f"--> Finished Fold {fold} in {time.time() - f_t0:.1f}s (Best cal_F1={best_score:.5f})", flush=True)
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    if args.smoke:
        print("Smoke check passed cleanly!")
        return

    print("\n" + "=" * 68)
    print(f"Context-Aware XLM-RoBERTa Total Time: {(time.time() - t_start)/60:.2f} min")
    print(f"Train OOF (14,306) Raw F1: {f1_score(labels_tr, oof_tr.argmax(axis=1), average='macro'):.5f} | Cal F1: {best_calibrated_f1(labels_tr, oof_tr):.5f}")
    print(f"Dev   OOF  (1,590) Raw F1: {f1_score(labels_de, oof_de.argmax(axis=1), average='macro'):.5f} | Cal F1: {best_calibrated_f1(labels_de, oof_de):.5f}")
    print("=" * 68)
    print("\nClassification Report on Train OOF:")
    print(classification_report(labels_tr, oof_tr.argmax(axis=1), digits=4))
    print("\nClassification Report on Dev OOF:")
    print(classification_report(labels_de, oof_de.argmax(axis=1), digits=4))

    np.save(OUTPUT / f"{tag}_oof.npy", oof_tr)
    np.save(OUTPUT / f"{tag}_dev.npy", oof_de)
    np.save(OUTPUT / f"{tag}_test.npy", test_probs)
    print(f"Saved {tag}_{{oof,dev,test}}.npy to spam_output/")
    keep_awake(False)


if __name__ == "__main__":
    main()
