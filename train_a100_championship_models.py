"""A100 Championship Multi-Backbone Trainer for DS102 Spam Classification (Target: Rank #1 > 0.80819).

Trains 4 complementary Context-Aware Foundation Transformers on Synchronized Stratified 5-Fold
across Full Train (14,306) + Dev (1,590) = 15,896 human-labeled reviews:
  1. `xlm_ctx2`          : `xlm-roberta-base` (Variant B product-first, MAX_LEN=288, seed=2024)
  2. `xlm_large_ctx`     : `xlm-roberta-large` (24L 1024d, Variant C long-context, MAX_LEN=384, seed=42)
  3. `mdeberta_ctx`      : `microsoft/mdeberta-v3-base` (Disentangled Relative Attention, MAX_LEN=384, seed=102)
  4. `phobert_large_ctx` : `vinai/phobert-large` (24L 1024d Native Vietnamese BPE, MAX_LEN=254, seed=777)

Key Engineering & Grandmaster Modeling Features:
  - Automatic A100/L4/T4 hardware scaling: native BF16 Tensor Cores + TF32 on A100/L4; safe FP16/FP32 fallback on T4.
  - Real-time Per-Fold & Per-Model Google Drive persistence (`--cache-dir`):
      * Saves lightweight per-fold OOF/Dev/Test predictions (`~180 KB`) after EVERY fold to Google Drive.
      * Automatically resumes from exact completed fold/model if Colab runtime ever disconnects!
      * Cleans up multi-GB `.pt` weights immediately after each fold so Drive/disk quota never fills up.
  - Tiki boilerplate stripping + explicit platform (`Tiki` vs `Shopee`) conditioning + `[CLS]+Mean+Max` fp32 pooling
    + 5x Multi-Sample Dropout + Layerwise Learning Rate Decay (LLRD) + Mid-Epoch Sweet-Spot Checkpointing.
"""

import argparse
import gc
import json
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

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
TMP_CKPT_DIR = Path("/tmp/ds102_ckpts") if os.name != "nt" else (OUTPUT / "tmp_ckpts")

TIKI_BOILERPLATE_RE = re.compile(
    r"giá sản phẩm trên tiki đã bao gồm thuế.*",
    flags=re.IGNORECASE | re.DOTALL,
)

MODEL_SPECS = {
    "xlm_ctx2": {
        "model_name": "xlm-roberta-base",
        "variant": "b",
        "max_len": 288,
        "epochs": 3,
        "lr": 2.5e-5,
        "layer_decay": 0.96,
        "pool_mode": "cls_mean",
        "seed": 2024,
        "cls_weights": [1.0, 4.5, 2.0, 1.3],
        "a100_bs": 32,
        "l4_bs": 16,
        "t4_bs": 8,
        "eff_bs": 16,
    },
    "xlm_large_ctx": {
        "model_name": "xlm-roberta-large",
        "variant": "c",
        "max_len": 384,
        "epochs": 4,
        "lr": 1.5e-5,
        "layer_decay": 0.94,
        "pool_mode": "cls_mean_max",
        "seed": 42,
        "cls_weights": [1.0, 4.5, 2.0, 1.3],
        "a100_bs": 16,
        "l4_bs": 8,
        "t4_bs": 4,
        "eff_bs": 16,
    },
    "mdeberta_ctx": {
        "model_name": "microsoft/mdeberta-v3-base",
        "variant": "c",
        "max_len": 384,
        "epochs": 4,
        "lr": 2.2e-5,
        "layer_decay": 0.94,
        "pool_mode": "cls_mean_max",
        "seed": 102,
        "cls_weights": [1.0, 4.6, 2.05, 1.3],
        "a100_bs": 16,
        "l4_bs": 8,
        "t4_bs": 4,
        "eff_bs": 16,
    },
    "phobert_large_ctx": {
        "model_name": "vinai/phobert-large",
        "variant": "p",
        "max_len": 254,  # Strictly <= 254 because PhoBERT max_position_embeddings == 258 (pad_idx=1)
        "epochs": 4,
        "lr": 1.6e-5,
        "layer_decay": 0.94,
        "pool_mode": "cls_mean_max",
        "seed": 777,
        "cls_weights": [1.0, 4.5, 2.0, 1.3],
        "a100_bs": 24,
        "l4_bs": 12,
        "t4_bs": 8,
        "eff_bs": 16,
    },
}


def norm_nfkc(s):
    return unicodedata.normalize("NFKC", str(s))


def norm_nfc(s):
    return unicodedata.normalize("NFC", str(s))


def clean_desc_for_window(d_str):
    d_nfkc = norm_nfkc(d_str)
    d_stripped = TIKI_BOILERPLATE_RE.sub(" ", d_nfkc)
    return " ".join(d_stripped.split())


def detect_platform(d_raw):
    d_str = str(d_raw)
    d_nfkc_low = norm_nfkc(d_str).lower()
    if "giá sản phẩm trên tiki" in d_nfkc_low or (d_str != norm_nfc(d_str)):
        return "Tiki"
    return "Shopee"


def best_desc_window(c_str, d_str, win_words=45, step=10, max_chars=240):
    d_clean = clean_desc_for_window(d_str)
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


def build_context_texts(df, variant="c", sep="</s>"):
    out = []
    ratings = df["rating"].fillna(5).astype(int).tolist()
    sold = df["num_sold"].fillna(0).astype(int).tolist()
    revs = df["num_review"].fillna(0).astype(int).tolist()
    cats = df["categories"].fillna("").tolist()
    prods = df["product_name"].fillna("").tolist()
    comms = df["comment"].fillna("").tolist()
    descs = df["description"].fillna("").tolist()

    for r, s_cnt, rev_cnt, cat, p, c, d in zip(ratings, sold, revs, cats, prods, comms, descs):
        plat = detect_platform(d)
        c_norm = norm_nfkc(c).replace("\r\n", "\n").replace("\r", "\n")
        c_norm = re.sub(r"\n+", " // ", c_norm)
        c_norm = " ".join(c_norm.split())

        if variant == "b":
            cat_parts = [x.strip() for x in re.sub(r"[\[\]']", "", norm_nfkc(cat)).split(",") if x.strip()]
            cat_norm = " > ".join(cat_parts[-2:])[:55]
            p_norm = " ".join(norm_nfkc(p).split())[:95]
            d_win = best_desc_window(c, d, win_words=50, step=10, max_chars=260)
            text = f"{p_norm} | {cat_norm} | {plat} {r} sao {sep} {c_norm[:430]} {sep} {d_win}"
        elif variant == "p":
            cat_norm = " ".join(re.sub(r"[\[\]']", "", norm_nfkc(cat)).split())[:60]
            p_norm = " ".join(norm_nfkc(p).split())[:100]
            d_win = best_desc_window(c, d, win_words=35, step=10, max_chars=165)
            text = f"{c_norm[:380]} {sep} {plat} | {r} sao | Bán {s_cnt} | {cat_norm} | SP: {p_norm} {sep} Mô tả: {d_win}"
        else:  # variant == "c": Long-context 384-token format for Large / mDeBERTa backbones
            cat_norm = " ".join(re.sub(r"[\[\]']", "", norm_nfkc(cat)).split())[:95]
            p_norm = " ".join(norm_nfkc(p).split())[:140]
            d_win = best_desc_window(c, d, win_words=60, step=12, max_chars=320)
            text = (
                f"{c_norm[:720]} {sep} Sàn: {plat} | {r} sao | Đã bán {s_cnt} ({rev_cnt} đánh giá) | "
                f"{cat_norm} | SP: {p_norm} {sep} Mô tả: {d_win}"
            )
        out.append(text)
    return out


def load_tokenizer_safe(model_name):
    if "deberta" in model_name.lower():
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception:
            return AutoTokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


class ContextPoolingClassifier(nn.Module):
    def __init__(self, model_name, num_labels=4, pool_mode="cls_mean_max", backbone_config=None):
        super().__init__()
        if backbone_config is None:
            cfg = AutoConfig.from_pretrained(model_name)
            if hasattr(cfg, "add_pooling_layer"):
                cfg.add_pooling_layer = False
            self.backbone = AutoModel.from_pretrained(model_name, config=cfg)
        else:
            cfg = backbone_config
            self.backbone = AutoModel.from_config(cfg)

        # Freeze vocabulary lookup table to prevent rare-SKU/token memorization & speed up training
        if hasattr(self.backbone, "embeddings") and hasattr(self.backbone.embeddings, "word_embeddings"):
            self.backbone.embeddings.word_embeddings.requires_grad_(False)

        self.pool_mode = pool_mode
        h = cfg.hidden_size
        mult = 3 if pool_mode == "cls_mean_max" else 2
        self.norm = nn.LayerNorm(h * mult)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])
        self.fc = nn.Linear(h * mult, num_labels)

    def forward(self, input_ids, attention_mask):
        seq_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Upcast to fp32 before pooling & LayerNorm so DeBERTa/Large models never overflow in mixed precision
        seq_f = seq_out.float()
        cls_vec = seq_f[:, 0, :]
        mask_f = attention_mask.unsqueeze(-1).float()
        mean_vec = (seq_f * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)

        if self.pool_mode == "cls_mean_max":
            masked_for_max = seq_f.masked_fill(mask_f == 0, -1e4)
            max_vec = masked_for_max.max(dim=1).values
            feat = self.norm(torch.cat([cls_vec, mean_vec, max_vec], dim=-1))
        else:
            feat = self.norm(torch.cat([cls_vec, mean_vec], dim=-1))

        logits = sum(self.fc(drop(feat)) for drop in self.dropouts) / len(self.dropouts)
        return logits


def build_llrd_optimizer(model, base_lr, layer_decay=0.95, weight_decay=0.01):
    """Layerwise Learning Rate Decay (LLRD) across arbitrary HuggingFace encoder backbones."""
    no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    param_groups = []
    seen = set()

    def add_group(named_params, lr):
        decay_p, nodecay_p = [], []
        for n, p in named_params:
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if any(k in n for k in no_decay_keys):
                nodecay_p.append(p)
            else:
                decay_p.append(p)
        if decay_p:
            param_groups.append({"params": decay_p, "lr": lr, "weight_decay": weight_decay})
        if nodecay_p:
            param_groups.append({"params": nodecay_p, "lr": lr, "weight_decay": 0.0})

    # 1. Classifier head gets 3x base_lr
    head_named = list(model.norm.named_parameters(prefix="norm")) + list(model.fc.named_parameters(prefix="fc"))
    add_group(head_named, base_lr * 3.0)

    # 2. Transformer encoder layers (top layer -> base_lr, decaying downward)
    layers = None
    if hasattr(model.backbone, "encoder") and hasattr(model.backbone.encoder, "layer"):
        layers = model.backbone.encoder.layer
    if layers is not None:
        n_layers = len(layers)
        for depth_from_top, layer_idx in enumerate(reversed(range(n_layers))):
            lr_l = base_lr * (layer_decay ** depth_from_top)
            add_group(layers[layer_idx].named_parameters(prefix=f"layer.{layer_idx}"), lr_l)
        bottom_lr = base_lr * (layer_decay ** n_layers)
    else:
        bottom_lr = base_lr

    # 3. Remaining trainable backbone params (position/relative embeddings, encoder LayerNorm/conv, etc.)
    add_group(model.backbone.named_parameters(prefix="backbone"), bottom_lr)

    return torch.optim.AdamW(param_groups)


def make_noisy_length_batches(indices, lengths, batch_size, rng):
    noise = rng.integers(-14, 15, size=len(indices))
    order = np.argsort(lengths[indices] + noise)
    sorted_idx = indices[order]
    batches = [sorted_idx[i : i + batch_size] for i in range(0, len(sorted_idx), batch_size)]
    rng.shuffle(batches)
    return batches


def predict_sorted(model, encodings, tokenizer, device, amp_dtype, use_amp, eval_bs=128, max_pos_clamp=None):
    model.eval()
    n = len(encodings["input_ids"])
    lengths = np.array([len(x) for x in encodings["input_ids"]])
    order = np.argsort(lengths)
    probs = np.zeros((n, 4), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, eval_bs):
            b_idx = order[start : start + eval_bs]
            items = [
                {
                    "input_ids": encodings["input_ids"][i][:max_pos_clamp] if max_pos_clamp else encodings["input_ids"][i],
                    "attention_mask": encodings["attention_mask"][i][:max_pos_clamp] if max_pos_clamp else encodings["attention_mask"][i],
                }
                for i in b_idx
            ]
            batch = tokenizer.pad(items, padding=True, return_tensors="pt")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs[b_idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def best_calibrated_f1(y_true, probs):
    best = f1_score(y_true, probs.argmax(axis=1), average="macro")
    for w1 in (0.75, 0.9, 1.0, 1.12, 1.25):
        for w2 in (0.88, 1.0, 1.12):
            w = np.array([1.0, w1, w2, 1.0])
            s = f1_score(y_true, (probs * w).argmax(axis=1), average="macro")
            if s > best:
                best = s
    return float(best)


def detect_hardware_profile(spec, model_name):
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.float32, False, False, 8, 2, 32, "CPU"

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    gpu_name = props.name
    bf16_ok = bool(torch.cuda.is_bf16_supported())

    if bf16_ok:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        amp_dtype = torch.bfloat16
        use_amp = True
        use_scaler = False
    elif "deberta" in model_name.lower():
        # DeBERTa-v3 disentangled attention can overflow in fp16 on older T4 GPUs; fp32 is 100% NaN-safe
        amp_dtype = torch.float32
        use_amp = False
        use_scaler = False
    else:
        amp_dtype = torch.float16
        use_amp = True
        use_scaler = True

    if vram_gb >= 35.0:
        bs = spec["a100_bs"]
        eval_bs = 128
    elif vram_gb >= 20.0:
        bs = spec["l4_bs"]
        eval_bs = 96
    else:
        bs = spec["t4_bs"]
        eval_bs = 64

    grad_accum = max(1, spec["eff_bs"] // bs)
    prec_str = "BF16" if (use_amp and amp_dtype == torch.bfloat16) else ("FP16" if use_amp else "FP32")
    hw_str = f"{gpu_name} ({vram_gb:.1f} GB VRAM | {prec_str} | bs={bs}x{grad_accum})"
    return device, amp_dtype, use_amp, use_scaler, bs, grad_accum, eval_bs, hw_str


def sync_from_cache(cache_dir: Path | None):
    if cache_dir is None or not cache_dir.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    restored = []
    for p in cache_dir.glob("*.np*"):
        dst = OUTPUT / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
            restored.append(p.name)
    if restored:
        print(f"[Drive Cache] Restored {len(restored)} cached files into spam_output/: {', '.join(sorted(restored))}")


def save_array_dual(arr, filename, cache_dir: Path | None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_p = OUTPUT / filename
    np.save(out_p, arr)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_p, cache_dir / filename)


def valid_prob_file(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path)
        return arr.shape == (expected_rows, 4) and np.isfinite(arr).all() and float(arr.sum()) > expected_rows * 0.95
    except Exception:
        return False


def train_single_backbone(tag, spec, train_df, dev_df, test_df, cache_dir: Path | None, smoke=False, force=False):
    model_name = spec["model_name"]
    seed = spec["seed"]
    max_len = spec["max_len"]
    max_pos_clamp = 254 if "phobert" in model_name.lower() else None

    n_tr, n_de, n_te = len(train_df), len(dev_df), len(test_df)
    labels_tr = train_df[LABEL].to_numpy(dtype=np.int64)
    labels_de = dev_df[LABEL].to_numpy(dtype=np.int64)
    full_labels = np.concatenate([labels_tr, labels_de])

    oof_p_path = OUTPUT / f"{tag}_oof.npy"
    dev_p_path = OUTPUT / f"{tag}_dev.npy"
    tst_p_path = OUTPUT / f"{tag}_test.npy"

    if not smoke and not force:
        if (
            valid_prob_file(oof_p_path, n_tr)
            and valid_prob_file(dev_p_path, n_de)
            and valid_prob_file(tst_p_path, n_te)
        ):
            oof_tr = np.load(oof_p_path)
            oof_de = np.load(dev_p_path)
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                for p in (oof_p_path, dev_p_path, tst_p_path):
                    if not (cache_dir / p.name).exists():
                        shutil.copy2(p, cache_dir / p.name)
            raw_f1 = f1_score(labels_tr, oof_tr.argmax(axis=1), average="macro")
            full_raw = f1_score(full_labels, np.vstack([oof_tr, oof_de]).argmax(axis=1), average="macro")
            print(f"\n[SKIP {tag}] Already trained & verified! Train Raw F1={raw_f1:.5f} | Full 15,896 Raw F1={full_raw:.5f}")
            return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, amp_dtype, use_amp, use_scaler, bs, grad_accum, eval_bs, hw_str = detect_hardware_profile(spec, model_name)
    print("\n" + "=" * 78)
    print(f"TRAINING [{tag}] -> {model_name} | variant={spec['variant']} | max_len={max_len} | seed={seed}")
    print(f"Hardware Profile: {hw_str}")
    print("=" * 78, flush=True)

    tokenizer = load_tokenizer_safe(model_name)
    sep = tokenizer.sep_token if getattr(tokenizer, "sep_token", None) else "</s>"

    tr_texts = build_context_texts(train_df, variant=spec["variant"], sep=sep)
    de_texts = build_context_texts(dev_df, variant=spec["variant"], sep=sep)
    te_texts = build_context_texts(test_df, variant=spec["variant"], sep=sep)
    print(f"Sample formatted text: {tr_texts[0][:260]}...", flush=True)

    full_texts = tr_texts + de_texts
    full_enc = tokenizer(full_texts, truncation=True, max_length=max_len)
    te_enc = tokenizer(te_texts, truncation=True, max_length=max_len)
    full_lengths = np.array([len(x) for x in full_enc["input_ids"]])
    print(
        f"Tokenized {len(full_texts)} train+dev rows: mean_len={full_lengths.mean():.1f}, "
        f"p95_len={np.percentile(full_lengths, 95):.0f}, max_len={full_lengths.max()}",
        flush=True,
    )

    oof_tr = np.zeros((n_tr, 4), dtype=np.float32)
    oof_de = np.zeros((n_de, 4), dtype=np.float32)
    test_probs = np.zeros((n_te, 4), dtype=np.float32)

    skf_tr = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(tr_texts, labels_tr))
    skf_de = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(de_texts, labels_de))

    cls_weights = torch.tensor(spec["cls_weights"], dtype=torch.float32, device=device)
    n_folds = 1 if smoke else 5
    n_epochs = 1 if smoke else spec["epochs"]

    TMP_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    fold_cache_dir = (cache_dir / "fold_preds") if (cache_dir is not None and not smoke) else (OUTPUT / "fold_preds")
    if not smoke:
        fold_cache_dir.mkdir(parents=True, exist_ok=True)

    t_model_start = time.time()
    for fold in range(1, n_folds + 1):
        tr_idx_a, val_idx_a = skf_tr[fold - 1]
        tr_idx_b, val_idx_b = skf_de[fold - 1]

        fold_npz = fold_cache_dir / f"{tag}_fold{fold}.npz"
        if not smoke and not force and fold_npz.exists():
            try:
                cached = np.load(fold_npz)
                if (
                    cached["val_tr"].shape == (len(val_idx_a), 4)
                    and cached["val_de"].shape == (len(val_idx_b), 4)
                    and cached["test_fold"].shape == (n_te, 4)
                ):
                    oof_tr[val_idx_a] = cached["val_tr"]
                    oof_de[val_idx_b] = cached["val_de"]
                    test_probs += cached["test_fold"] / n_folds
                    f_cal = best_calibrated_f1(
                        np.concatenate([labels_tr[val_idx_a], labels_de[val_idx_b]]),
                        np.vstack([cached["val_tr"], cached["val_de"]]),
                    )
                    print(f"--> [Resume Fold {fold}/{n_folds}] Loaded from Drive cache! (cal_F1={f_cal:.5f})", flush=True)
                    continue
            except Exception as e:
                print(f"Warning: corrupt fold cache {fold_npz} ({e}), retraining fold {fold}...")

        f_t0 = time.time()
        comb_tr_idx = np.concatenate([tr_idx_a, n_tr + tr_idx_b])
        comb_val_idx = np.concatenate([val_idx_a, n_tr + val_idx_b])

        rng = np.random.default_rng(seed + fold * 100)
        model = ContextPoolingClassifier(
            model_name, num_labels=4, pool_mode=spec["pool_mode"]
        ).to(device)

        optimizer = build_llrd_optimizer(
            model, base_lr=spec["lr"], layer_decay=spec["layer_decay"], weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.02)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        batches_per_epoch = (len(comb_tr_idx) + bs - 1) // bs
        steps_per_epoch = (batches_per_epoch + grad_accum - 1) // grad_accum
        total_steps = steps_per_epoch * n_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.08 * total_steps), num_training_steps=total_steps
        )

        val_enc = {
            "input_ids": [full_enc["input_ids"][i] for i in comb_val_idx],
            "attention_mask": [full_enc["attention_mask"][i] for i in comb_val_idx],
        }
        ckpt_path = TMP_CKPT_DIR / f"{tag}_fold_{fold}.pt"
        best_score = -1.0

        def evaluate_and_checkpoint(tag_label, ep_loss_str, t_ref):
            nonlocal best_score
            val_p = predict_sorted(
                model, val_enc, tokenizer, device, amp_dtype, use_amp, eval_bs=eval_bs, max_pos_clamp=max_pos_clamp
            )
            raw_f1 = f1_score(full_labels[comb_val_idx], val_p.argmax(axis=1), average="macro")
            cal_f1 = best_calibrated_f1(full_labels[comb_val_idx], val_p)
            max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0
            marker = ""
            if cal_f1 > best_score:
                best_score = cal_f1
                marker = " *BEST*"
                if not smoke:
                    torch.save(model.state_dict(), ckpt_path)
            print(
                f"[{tag} F{fold} | {tag_label}] {ep_loss_str} | raw_F1={raw_f1:.5f} | cal_F1={cal_f1:.5f} "
                f"| VRAM={max_vram:.1f}GB | {time.time() - t_ref:.1f}s{marker}",
                flush=True,
            )
            model.train()

        for epoch in range(1, n_epochs + 1):
            ep_t0 = time.time()
            model.train()
            batches = make_noisy_length_batches(comb_tr_idx, full_lengths, bs, rng)
            mid_batch = len(batches) // 2
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step_i, b_idx in enumerate(batches, 1):
                items = [
                    {
                        "input_ids": full_enc["input_ids"][i][:max_pos_clamp] if max_pos_clamp else full_enc["input_ids"][i],
                        "attention_mask": full_enc["attention_mask"][i][:max_pos_clamp] if max_pos_clamp else full_enc["attention_mask"][i],
                    }
                    for i in b_idx
                ]
                targets = torch.tensor(full_labels[b_idx], dtype=torch.long, device=device)
                batch = tokenizer.pad(items, padding=True, return_tensors="pt")
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = criterion(logits, targets) / grad_accum

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += loss.item() * grad_accum

                if step_i % grad_accum == 0 or step_i == len(batches):
                    if use_scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        prev_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= prev_scale:
                            scheduler.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Mid-epoch checkpoint evaluation in epochs >= 2 catches sweet-spot before Class 1 overfitting
                if not smoke and epoch >= 2 and step_i == mid_batch and (step_i % grad_accum == 0):
                    evaluate_and_checkpoint(f"Ep {epoch - 0.5:.1f}/{n_epochs}", f"loss={total_loss/step_i:.4f}", ep_t0)

            evaluate_and_checkpoint(f"Ep {epoch:.1f}/{n_epochs}", f"loss={total_loss/len(batches):.4f}", ep_t0)

        if not smoke and ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        val_best_p = predict_sorted(
            model, val_enc, tokenizer, device, amp_dtype, use_amp, eval_bs=eval_bs, max_pos_clamp=max_pos_clamp
        )
        val_tr_p = val_best_p[: len(val_idx_a)]
        val_de_p = val_best_p[len(val_idx_a) :]
        fold_test_p = predict_sorted(
            model, te_enc, tokenizer, device, amp_dtype, use_amp, eval_bs=eval_bs, max_pos_clamp=max_pos_clamp
        )

        oof_tr[val_idx_a] = val_tr_p
        oof_de[val_idx_b] = val_de_p
        test_probs += fold_test_p / n_folds

        if not smoke:
            np.savez_compressed(fold_npz, val_tr=val_tr_p, val_de=val_de_p, test_fold=fold_test_p)
            if ckpt_path.exists():
                ckpt_path.unlink()  # Free multi-GB checkpoint immediately after fold inference!

        print(f"--> Finished [{tag}] Fold {fold}/{n_folds} in {time.time() - f_t0:.1f}s (Best cal_F1={best_score:.5f})", flush=True)
        del model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if smoke:
        print(f"[SMOKE PASS] {tag} completed cleanly!")
        return

    full_oof = np.vstack([oof_tr, oof_de])
    tr_raw = float(f1_score(labels_tr, oof_tr.argmax(axis=1), average="macro"))
    tr_cal = best_calibrated_f1(labels_tr, oof_tr)
    de_raw = float(f1_score(labels_de, oof_de.argmax(axis=1), average="macro"))
    full_raw = float(f1_score(full_labels, full_oof.argmax(axis=1), average="macro"))
    full_cal = best_calibrated_f1(full_labels, full_oof)
    elapsed_min = (time.time() - t_model_start) / 60.0

    print("\n" + "-" * 74)
    print(f"[{tag}] ({model_name}) Total Time: {elapsed_min:.2f} min")
    print(f"  Train OOF (14,306) Raw F1: {tr_raw:.5f} | Cal F1: {tr_cal:.5f}")
    print(f"  Dev   OOF  (1,590) Raw F1: {de_raw:.5f}")
    print(f"  Full  OOF (15,896) Raw F1: {full_raw:.5f} | Cal F1: {full_cal:.5f}")
    print("-" * 74)
    print(classification_report(full_labels, full_oof.argmax(axis=1), digits=4))

    save_array_dual(oof_tr, f"{tag}_oof.npy", cache_dir)
    save_array_dual(oof_de, f"{tag}_dev.npy", cache_dir)
    save_array_dual(test_probs, f"{tag}_test.npy", cache_dir)

    meta = {
        "tag": tag,
        "model_name": model_name,
        "variant": spec["variant"],
        "max_len": max_len,
        "seed": seed,
        "elapsed_minutes": round(elapsed_min, 2),
        "train_raw_f1": tr_raw,
        "train_cal_f1": tr_cal,
        "dev_raw_f1": de_raw,
        "full_15896_raw_f1": full_raw,
        "full_15896_cal_f1": full_cal,
    }
    meta_str = json.dumps(meta, indent=2)
    (OUTPUT / f"{tag}_metrics.json").write_text(meta_str, encoding="utf-8")
    if cache_dir is not None:
        (cache_dir / f"{tag}_metrics.json").write_text(meta_str, encoding="utf-8")
    print(f"Saved & backed up {tag}_{{oof,dev,test}}.npy to spam_output/ and Google Drive cache!", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xlm_ctx2", "xlm_large_ctx", "mdeberta_ctx", "phobert_large_ctx"],
        choices=list(MODEL_SPECS.keys()),
        help="Which championship Level-1 Transformer streams to train",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Persistent directory (e.g. /content/drive/MyDrive/DS102_Colab_Cache) for live per-fold & .npy backup",
    )
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test on small subset")
    parser.add_argument("--force", action="store_true", help="Retrain even if cached .npy files already exist")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is None and Path("/content/drive/MyDrive").exists():
        cache_dir = Path("/content/drive/MyDrive/DS102_Colab_Cache")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Persistent Cache Active: {cache_dir}")
        sync_from_cache(cache_dir)

    train_df = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev_df = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test_df = load_frame(ROOT / "ds102_test.csv")

    if args.smoke:
        train_df = train_df.iloc[:160].reset_index(drop=True)
        dev_df = dev_df.iloc[:40].reset_index(drop=True)
        test_df = test_df.iloc[:40].reset_index(drop=True)

    t0 = time.time()
    for tag in args.models:
        train_single_backbone(
            tag=tag,
            spec=MODEL_SPECS[tag],
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            cache_dir=cache_dir,
            smoke=args.smoke,
            force=args.force,
        )

    print(f"\nALL REQUESTED MODELS COMPLETED IN {(time.time() - t0)/60:.2f} MIN!")


if __name__ == "__main__":
    main()
