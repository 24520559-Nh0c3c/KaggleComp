"""Grandmaster Multi-Foundation Stacking Pipeline for DS102 (Submission 17 — Target: Rank #1 > 0.80819).

Integrates up to 10 Level-1 Probability Streams (including A100-trained Large & Multilingual Backbones):
  1. `xlge` (`xlm-roberta-large`, 24L 1024d, MAX_LEN=384, [CLS]+Mean+Max + LLRD)
  2. `mdeb` (`microsoft/mdeberta-v3-base`, Disentangled Relative Attention, MAX_LEN=384)
  3. `plge` (`vinai/phobert-large`, 24L 1024d Native Vietnamese BPE, MAX_LEN=254)
  4. `xctx` & `xctx2` (`xlm-roberta-base` Context Variants A & B)
  5. `pctx` (`vinai/phobert-base` Context)
  6. `xcom` (Comment-Only XLM-R Raw+NFKC TTA), `deb` (DeBERTa), `rob` (RoBERTa), `tfidf` (Word+Char TF-IDF)

Major Innovations Over Submission 16 (0.80557 LB):
  - Explicit Platform (`Tiki` vs `Shopee` — 10x Class-1 & 5x Class-3 prior shift!) & Shop-Promo / Shopee-Chip
    / Delivery-Only vs Product-Specific lexical-semantic features discovered via error analysis.
  - All 16 SVD semantic coordinates + Out-of-Fold KNN Neighbor Graph + Group LOO expectations on Large streams.
  - 6-Family Level-2 Ensemble on Full 15,896 Labeled Reviews (10-Fold):
      * 3-Profile Bagged XGBoost (d4 + d5 + d3)
      * 2-Profile Bagged LightGBM (d4 + d5)
      * LightGBM ExtraTrees (randomized split thresholds)
      * CatBoost Depth-6 & High-Reg Depth-5 Symmetric Trees
      * PyTorch Multi-Sample-Dropout Residual Meta-MLP on standardized Level-1 logits & shifts
  - Strict Nested 5-Fold Out-of-Fold Simplex Blend + Regularized Multiplier Calibration (eliminates legacy anchor dilution).
  - Automatic bidirectional Google Drive persistence (`--cache-dir`) + Kaggle CLI auto-submit & Leaderboard check.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from spam_pipeline import LABEL, load_frame, make_submission

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "spam_output"

C1_PATTERNS = [
    r"gửi gắm rất nhiều tâm tư", r"không thể trông đợi gì hơn", r"văn mẫu",
    r"tâm tư và tình cảm", r"người bán đã gửi gắm",
    r"dành cho ng tiêu dùng chúng tôi 1 sản phẩm", r"xinh đẹp tuyệt vời",
    r"tuyệt vời.*tuyệt vời", r"tuyệt vời.*đóng gói", r"chất lượng sản phẩm tuyệt vời",
]
BRAND_PROMO_PAT = re.compile(
    r"(tổng đại lý|chuyên cung cấp|chuyên sỉ|phân phối chính hãng|giá tận gốc|"
    r"chào mừng đến với cửa hàng|kính gửi khách hàng|thời gian giao hàng dự kiến|"
    r"tuyển sỉ|tuyển ctv|sỉ lẻ toàn quốc|đặt hàng ngay|cam kết hàng chính hãng)"
)
SHOPEE_CHIP_PHRASES = [
    "chất lượng sản phẩm tuyệt vời",
    "đóng gói sản phẩm rất đẹp và chắc chắn",
    "đóng gói sản phẩm đẹp",
    "shop phục vụ rất tốt",
    "rất đáng tiền",
    "thời gian giao hàng rất nhanh",
    "nên mua nhé mọi người",
]
C2_PATTERNS = [
    r"chưa (dùng|sử dụng|xài|sài|biết|thử|mở|khui|uống|ăn|đeo|mặc|cầm|lắp)",
    r"sài như nào.*chưa biết", r"xài như nào.*chưa biết", r"độ bền thì chưa",
    r"cần thời gian.*(sử dụng|đánh giá|kiểm chứng)", r"thời gian sau mới biết",
    r"chưa rõ", r"chưa test", r"mua tặng", r"mua hộ", r"mua giùm", r"mua dùm",
    r"kiểm chứng", r"dịch vụ", r"shipper", r"giao hàng", r"đóng gói", r"bảo hành",
    r"nhận xe", r"đặt hôm", r"hẹn thứ", r"nguyên kiện",
]
C3_PATTERNS = [
    r"[a-zA-Z0-9]{14,}", r"(.)\1{4,}",
    r"(zalo|sđt|hotline|liên hệ|inbox|0[35789]\d{8})",
    r"(http|https|www\.)", r"\[tb\]", r"\bvoucher\b",
]
SPEC_HEADER_PAT = re.compile(
    r"(thông số|thành phần|cách dùng|hướng dẫn sử dụng|lưu ý\s*:|tên sản phẩm|"
    r"thương hiệu\s*:|tác dụng\s*:|điểm nổi bật|cơ chế|thời gian giao hàng dự kiến|"
    r"gói hàng bao gồm|quý khách lưu ý|cam kết hàng chính hãng|đặc trưng\s*:)"
)
EVOUCHER_PAT = re.compile(r"(e-?voucher|\bvoucher\b|thẻ nạp|thẻ cào|mã thẻ|nạp tiền điện thoại|gói cước|thẻ viettel|thẻ mobi|thẻ vina)")
TIKI_BOILERPLATE_RE = re.compile(r"giá sản phẩm trên tiki đã bao gồm thuế.*", flags=re.IGNORECASE | re.DOTALL)

DELIVERY_PRAISE_STOPWORDS = {
    "giao", "hàng", "nhanh", "chóng", "đóng", "gói", "cẩn", "thận", "kỹ", "kĩ", "chắc", "chắn",
    "ship", "shipper", "vận", "chuyển", "nhận", "được", "đơn", "hẹn", "đúng", "giờ", "ngày",
    "shop", "tiki", "tư", "vấn", "nhiệt", "tình", "phục", "vụ", "bọc", "kiện", "nguyên",
    "ủng", "hộ", "cảm", "cám", "ơn", "thanks", "thks", "thank", "tốt", "đẹp", "ok", "ổn",
    "rất", "quá", "lắm", "luôn", "nha", "nhé", "ạ", "ơi", "vời", "tuyệt", "sản", "phẩm",
    "chất", "lượng", "giá", "tiền", "đáng", "sao", "cho", "với", "của", "các", "mình", "tôi",
    "này", "khi", "còn", "nhưng", "mới", "chưa", "biết", "thử", "dùng", "xài", "sài", "mua",
}


def norm_nfkc(s):
    return unicodedata.normalize("NFKC", str(s))


def norm_nfc(s):
    return unicodedata.normalize("NFC", str(s))


def max_repeat(s):
    repeats = [len(m.group()) for m in re.finditer(r"(.)\1+", str(s))]
    return max(repeats) if repeats else 1


def char_entropy(s):
    s = str(s).lower()
    if len(s) <= 1:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * np.log2(c / n) for c in counts.values())


def recover_roberta(all_p, xlm_p, deb_p):
    p = np.clip(3.0 * all_p - xlm_p - deb_p, 1e-6, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def fast_macro_f1(y_true, y_pred):
    cm = np.bincount(y_true * 4 + y_pred, minlength=16).reshape(4, 4)
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(denom > 0, 2 * tp / denom, 0.0)))


def add_loo_group_stats(df, group_col, prefix, stat_cols):
    grp = df.groupby(group_col)
    cnt = grp["_char_len"].transform("count")
    out = {f"{prefix}_count": np.log1p(cnt).to_numpy()}
    for col in stat_cols:
        s = grp[col].transform("sum")
        prior = df[col].mean()
        loo_mean = (s - df[col] + 3.0 * prior) / (cnt - 1.0 + 3.0)
        out[f"{prefix}_loo_{col}"] = loo_mean.to_numpy()
    return pd.DataFrame(out, index=df.index)


def compute_oof_target_encoding(train_df, y_labels, eval_dfs, skf):
    priors = np.bincount(y_labels) / len(y_labels)
    te_train = {}
    te_evals = [{} for _ in eval_dfs]

    groups = [
        ("product_name", "prod_te", 5.0),
        ("categories", "cat_te", 10.0),
        ("_plat_cat", "plat_cat_te", 15.0),
    ]
    for group_col, prefix, smoothing in groups:
        for c in range(4):
            te_oof = np.full(len(train_df), priors[c])
            y_bin = (y_labels == c).astype(float)
            for tr_idx, val_idx in skf.split(train_df, y_labels):
                tr_sub = pd.DataFrame({"g": train_df[group_col].iloc[tr_idx], "y": y_bin[tr_idx]})
                stats = tr_sub.groupby("g")["y"].agg(["sum", "count"])
                smoothed = (stats["sum"] + smoothing * priors[c]) / (stats["count"] + smoothing)
                te_oof[val_idx] = train_df[group_col].iloc[val_idx].map(smoothed).fillna(priors[c]).to_numpy()
            te_train[f"{prefix}_{c}"] = te_oof

            full_sub = pd.DataFrame({"g": train_df[group_col], "y": y_bin})
            stats_full = full_sub.groupby("g")["y"].agg(["sum", "count"])
            smoothed_full = (stats_full["sum"] + smoothing * priors[c]) / (stats_full["count"] + smoothing)
            for idx, edf in enumerate(eval_dfs):
                te_evals[idx][f"{prefix}_{c}"] = edf[group_col].map(smoothed_full).fillna(priors[c]).to_numpy()

    return te_train, te_evals


class ResidualMetaMLP(nn.Module):
    """Compact Multi-Sample-Dropout Residual Meta-MLP over standardized Level-1 logits & shifts."""

    def __init__(self, in_dim, hidden=128, num_classes=4):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.res_block = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden),
        )
        self.norm_out = nn.LayerNorm(hidden)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3, 0.4, 0.5)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.norm_out(h + 0.5 * self.res_block(h))
        return sum(self.head(d(h)) for d in self.dropouts) / len(self.dropouts)


def fit_predict_meta_mlp_cv_fold(X_tr_np, y_tr, X_va_np, y_va, X_eval_np_list, cls_weights_np, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_tr_s = np.clip(scaler.fit_transform(np.nan_to_num(X_tr_np, 0.0)), -8.0, 8.0).astype(np.float32)
    X_va_s = np.clip(scaler.transform(np.nan_to_num(X_va_np, 0.0)), -8.0, 8.0).astype(np.float32)
    X_ev_s = [np.clip(scaler.transform(np.nan_to_num(Xe, 0.0)), -8.0, 8.0).astype(np.float32) for Xe in X_eval_np_list]

    model = ResidualMetaMLP(X_tr_s.shape[1], hidden=128, num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.1e-3, weight_decay=2.5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    cw = torch.tensor(cls_weights_np, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.015)

    x_t = torch.from_numpy(X_tr_s).to(device)
    y_t = torch.from_numpy(y_tr.astype(np.int64)).to(device)
    x_v = torch.from_numpy(X_va_s).to(device)
    y_v = torch.from_numpy(y_va.astype(np.int64)).to(device)

    bs = 256
    n = len(X_tr_s)
    best_val_loss = float("inf")
    best_state = None
    rng = np.random.default_rng(seed)

    for _ep in range(30):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            opt.zero_grad(set_to_none=True)
            logits = model(x_t[idx])
            loss = crit(logits, y_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(x_v)
            v_loss = float(crit(va_logits, y_v).item())
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_out = torch.softmax(model(x_v), dim=-1).cpu().numpy()
        ev_outs = [torch.softmax(model(torch.from_numpy(Xe).to(device)), dim=-1).cpu().numpy() for Xe in X_ev_s]
    return va_out, ev_outs


def sync_cache_in(cache_dir: Path | None):
    if cache_dir is None or not cache_dir.exists():
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in cache_dir.glob("*.np*"):
        dst = OUTPUT / p.name
        if not dst.exists() or p.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(p, dst)
            copied.append(p.name)
    if copied:
        print(f"   [Drive Sync] Loaded {len(copied)} files from {cache_dir}")


def backup_to_cache(paths, cache_dir: Path | None):
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        p = Path(p)
        if p.exists():
            shutil.copy2(p, cache_dir / p.name)
    print(f"   [Drive Backup] Saved {len(paths)} artifacts permanently to {cache_dir}")


def load_optional_stream(prefix, n_tr, n_de, n_te):
    p_oof = OUTPUT / f"{prefix}_oof.npy"
    p_dev = OUTPUT / f"{prefix}_dev.npy"
    p_tst = OUTPUT / f"{prefix}_test.npy"
    if p_oof.exists() and p_dev.exists() and p_tst.exists():
        oof = np.load(p_oof)
        dev = np.load(p_dev)
        tst = np.load(p_tst)
        if oof.shape == (n_tr, 4) and dev.shape == (n_de, 4) and tst.shape == (n_te, 4):
            if np.isnan(oof).any() or np.isnan(dev).any() or np.isnan(tst).any():
                print(f"   [WARNING] Stream '{prefix}' contains NaN values! Rejecting stream.")
                return None
            return oof, dev, tst
    return None


def tokenize_words(s):
    return re.findall(r"\w+", s.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default=None, help="Google Drive persistent backup directory")
    parser.add_argument("--no-submit", action="store_true", help="Skip automatic Kaggle CLI submission")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is None and Path("/content/drive/MyDrive").exists():
        cache_dir = Path("/content/drive/MyDrive/DS102_Colab_Cache")

    t_start = time.time()
    print("1. Loading datasets, syncing Google Drive cache, and discovering Level-1 streams...")
    sync_cache_in(cache_dir)

    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")
    sample = pd.read_csv(ROOT / "ds102_samplesubmission.csv", encoding="utf-8-sig")

    labels = train[LABEL].to_numpy()
    dev_labels = dev[LABEL].to_numpy()
    y_full = np.concatenate([labels, dev_labels])
    n_tr, n_de, n_te = len(train), len(dev), len(test)

    # Core Comment-Only & Baseline Context Streams
    xlm_raw_oof = np.load(OUTPUT / "xlm_oof_probabilities.npy")
    xlm_raw_dev = np.load(OUTPUT / "xlm_dev_probabilities.npy")
    xlm_raw_test = np.load(OUTPUT / "xlm_test_probabilities.npy")

    xlm_nfkc_oof = np.load(OUTPUT / "xlm_nfkc_oof.npy")
    xlm_nfkc_dev = np.load(OUTPUT / "xlm_nfkc_dev.npy")
    xlm_nfkc_test = np.load(OUTPUT / "xlm_nfkc_test.npy")

    xcom = (
        0.5 * xlm_raw_oof + 0.5 * xlm_nfkc_oof,
        0.5 * xlm_raw_dev + 0.5 * xlm_nfkc_dev,
        0.5 * xlm_raw_test + 0.5 * xlm_nfkc_test,
    )
    xctx1 = (
        np.load(OUTPUT / "xlm_ctx_oof.npy"),
        np.load(OUTPUT / "xlm_ctx_dev.npy"),
        np.load(OUTPUT / "xlm_ctx_test.npy"),
    )
    pctx = (
        np.load(OUTPUT / "phobert_ctx_oof.npy"),
        np.load(OUTPUT / "phobert_ctx_dev.npy"),
        np.load(OUTPUT / "phobert_ctx_test.npy"),
    )
    deb = (
        np.load(OUTPUT / "deberta_oof_probabilities.npy"),
        np.load(OUTPUT / "deberta_dev_probabilities.npy"),
        np.load(OUTPUT / "deberta_test_probabilities.npy"),
    )
    tfidf = (
        np.load(OUTPUT / "tfidf_oof_probabilities.npy"),
        np.load(OUTPUT / "tfidf_dev_probabilities.npy"),
        np.load(OUTPUT / "tfidf_test_probabilities.npy"),
    )
    all_3 = (
        np.load(OUTPUT / "oof_probabilities.npy"),
        np.load(OUTPUT / "dev_probabilities.npy"),
        np.load(OUTPUT / "test_probabilities.npy"),
    )
    rob = (
        recover_roberta(all_3[0], xlm_raw_oof, deb[0]),
        recover_roberta(all_3[1], xlm_raw_dev, deb[1]),
        recover_roberta(all_3[2], xlm_raw_test, deb[2]),
    )

    # Discover A100 Championship Context Streams
    xctx2 = load_optional_stream("xlm_ctx2", n_tr, n_de, n_te)
    xlge = load_optional_stream("xlm_large_ctx", n_tr, n_de, n_te)
    mdeb = load_optional_stream("mdeberta_ctx", n_tr, n_de, n_te)
    plge = load_optional_stream("phobert_large_ctx", n_tr, n_de, n_te)

    # Averaged XLM-base context stream
    if xctx2 is not None:
        xctx = tuple(0.5 * a + 0.5 * b for a, b in zip(xctx1, xctx2))
    else:
        xctx = xctx1

    active_streams = [
        ("xctx", xctx, True),
        ("pctx", pctx, True),
        ("xcom", xcom, True),
        ("deb", deb, True),
        ("rob", rob, False),
        ("tfidf", tfidf, False),
    ]
    if xctx2 is not None:
        active_streams.append(("xctx2", xctx2, True))
    if xlge is not None:
        active_streams.append(("xlge", xlge, True))
    if mdeb is not None:
        active_streams.append(("mdeb", mdeb, True))
    if plge is not None:
        active_streams.append(("plge", plge, True))

    print("   Active Level-1 Probability Streams (Full 15,896 Raw Macro F1):")
    for s_name, (s_tr, s_de, _), _ in active_streams:
        f_full = fast_macro_f1(y_full, np.vstack([s_tr, s_de]).argmax(1))
        f_tr = fast_macro_f1(labels, s_tr.argmax(1))
        print(f"     - {s_name:6s}: Train OOF={f_tr:.5f} | Full 15,896 OOF={f_full:.5f}")

    # Build weighted Context Consensus & Overall Neural Consensus dynamically based on available A100 models
    ctx_parts = [("xctx", xctx, 1.15 if xctx2 is not None else 1.0), ("pctx", pctx, 0.55)]
    if xlge is not None:
        ctx_parts.append(("xlge", xlge, 1.55))
    if mdeb is not None:
        ctx_parts.append(("mdeb", mdeb, 1.25))
    if plge is not None:
        ctx_parts.append(("plge", plge, 1.15))

    w_ctx_sum = sum(w for _, _, w in ctx_parts)
    ctx_ens = tuple(
        sum(w * tup[split_i] for _, tup, w in ctx_parts) / w_ctx_sum
        for split_i in range(3)
    )
    com_ens = tuple(0.76 * xcom[i] + 0.24 * deb[i] for i in range(3))
    ens_all = tuple(0.74 * ctx_ens[i] + 0.19 * com_ens[i] + 0.03 * rob[i] + 0.04 * tfidf[i] for i in range(3))

    print(
        f"   --> Combined Context Neural Consensus Full 15,896 Raw F1: "
        f"{fast_macro_f1(y_full, np.vstack([ctx_ens[0], ctx_ens[1]]).argmax(1)):.5f}"
    )
    print(
        f"   --> Combined All-L1 Neural Consensus Full 15,896 Raw F1:  "
        f"{fast_macro_f1(y_full, np.vstack([ens_all[0], ens_all[1]]).argmax(1)):.5f}"
    )

    sim_npz = np.load(OUTPUT / "nfkc_sim_features.npz")
    ks_npz = np.load(OUTPUT / "knn_svd_features.npz")

    all_df = pd.concat(
        [train.assign(_split="train"), dev.assign(_split="dev"), test.assign(_split="test")],
        ignore_index=True,
    )

    # Platform detection on all 19,870 rows (zero label leakage, pure listing metadata!)
    raw_descs_all = all_df["description"].fillna("").astype(str).tolist()
    nfkc_descs_all = [norm_nfkc(s) for s in raw_descs_all]
    is_tiki_desc_all = np.array(["giá sản phẩm trên tiki" in s.lower() for s in nfkc_descs_all], dtype=float)
    has_nfd_desc_all = np.array([s != norm_nfc(s) for s in raw_descs_all], dtype=float)
    is_tiki_plat_all = np.maximum(is_tiki_desc_all, has_nfd_desc_all)
    all_df["_is_tiki_desc"] = is_tiki_desc_all
    all_df["_has_nfd_desc"] = has_nfd_desc_all
    all_df["_is_tiki_plat"] = is_tiki_plat_all
    all_df["_plat_cat"] = all_df["_is_tiki_plat"].astype(int).astype(str) + "__" + all_df["category"].fillna("NA").astype(str)

    for k in sim_npz.files:
        all_df[f"_sim_{k}"] = sim_npz[k]

    unsup_names = [
        "max_sim_global", "top5_mean_global", "dup_cnt_050", "dup_cnt_070",
        "max_sim_same_prod", "mean_sim_same_prod",
    ]
    for idx, u_name in enumerate(unsup_names):
        all_df[f"_u_{u_name}"] = ks_npz["unsup_sim"][:, idx]

    n_svd = ks_npz["svd_feats"].shape[1]
    for idx in range(n_svd):
        all_df[f"_svd_{idx}"] = ks_npz["svd_feats"][:, idx]

    # Group Leave-One-Out (LOO) expectations across products & category paths
    loo_streams = [("xctx", xctx), ("pctx", pctx), ("xcom", xcom), ("deb", deb), ("ctxens", ctx_ens)]
    if xlge is not None:
        loo_streams.append(("xlge", xlge))
    if mdeb is not None:
        loo_streams.append(("mdeb", mdeb))
    if plge is not None:
        loo_streams.append(("plge", plge))

    stat_cols = ["_char_len", "rating"]
    all_df["_char_len"] = all_df["comment"].fillna("").astype(str).str.len()
    for s_name, (s_tr, s_de, s_te) in loo_streams:
        s_stack = np.vstack([s_tr, s_de, s_te])
        for c in range(4):
            col_n = f"_{s_name}_{c}"
            all_df[col_n] = s_stack[:, c]
            stat_cols.append(col_n)

    prod_loo_df = add_loo_group_stats(all_df, "product_name", "prod", stat_cols)
    cat_loo_df = add_loo_group_stats(all_df, "categories", "cat_path", stat_cols)
    all_df = pd.concat([all_df, prod_loo_df, cat_loo_df], axis=1)

    all_df["_norm_c"] = all_df["comment"].fillna("").apply(lambda s: " ".join(norm_nfkc(s).lower().split()))
    c_freq = all_df["_norm_c"].value_counts()
    all_df["comment_global_freq"] = all_df["_norm_c"].map(c_freq).astype(float)

    le = LabelEncoder()
    all_df["category_code"] = le.fit_transform(all_df["category"].astype(str))
    le_pc = LabelEncoder()
    all_df["plat_cat_code"] = le_pc.fit_transform(all_df["_plat_cat"])

    # Precompute expensive row-level text & lexical-semantic domain features ONCE for all 19,870 rows!
    print("2. Computing targeted Platform, Shop-Promo, Shopee-Chip & Delivery-vs-Product features...")
    raw_comments_all = all_df["comment"].fillna("").astype(str).tolist()
    nfkc_comments_all = [norm_nfkc(s) for s in raw_comments_all]
    nfkc_prods_all = [norm_nfkc(s) for s in all_df["product_name"].fillna("").astype(str)]
    clean_descs_all = [TIKI_BOILERPLATE_RE.sub(" ", s) for s in nfkc_descs_all]

    pre_txt = {}
    pre_txt["is_tiki_desc"] = is_tiki_desc_all
    pre_txt["has_nfd_desc"] = has_nfd_desc_all
    pre_txt["is_tiki_plat"] = is_tiki_plat_all
    pre_txt["has_nfd_comment"] = np.array([float(s != norm_nfc(s)) for s in raw_comments_all])
    pre_txt["newline_cnt"] = np.array([s.replace("\r\n", "\n").count("\n") for s in raw_comments_all], dtype=float)
    pre_txt["has_spec_header"] = np.array([float(bool(SPEC_HEADER_PAT.search(s.lower()))) for s in nfkc_comments_all])
    pre_txt["has_brand_promo"] = np.array([float(bool(BRAND_PROMO_PAT.search(s.lower()))) for s in nfkc_comments_all])
    pre_txt["is_evoucher"] = np.array([float(bool(EVOUCHER_PAT.search(p.lower()))) for p in nfkc_prods_all])
    pre_txt["hashtag_cnt_comm"] = np.array([len(re.findall(r"#\w+", s)) for s in nfkc_comments_all], dtype=float)
    pre_txt["hashtag_cnt_desc"] = np.array([len(re.findall(r"#\w+", s)) for s in clean_descs_all], dtype=float)
    pre_txt["shopee_chip_cnt"] = np.array(
        [sum(1 for ph in SHOPEE_CHIP_PHRASES if ph in s.lower()) for s in nfkc_comments_all], dtype=float
    )
    pre_txt["shopee_chip_non_tiki"] = pre_txt["shopee_chip_cnt"] * (1.0 - is_tiki_plat_all)

    char_len_all = np.array([len(s) for s in nfkc_comments_all], dtype=float)
    pre_txt["char_len"] = char_len_all
    pre_txt["word_count"] = np.array([len(s.split()) for s in nfkc_comments_all], dtype=float)
    pre_txt["char_entropy"] = np.array([char_entropy(s) for s in nfkc_comments_all], dtype=float)
    pre_txt["max_char_repeat"] = np.array([max_repeat(s) for s in nfkc_comments_all], dtype=float)
    pre_txt["exclamation_count"] = np.array([s.count("!") for s in nfkc_comments_all], dtype=float)
    pre_txt["question_count"] = np.array([s.count("?") for s in nfkc_comments_all], dtype=float)
    pre_txt["uppercase_ratio"] = np.array(
        [sum(1 for ch in s if ch.isupper()) / (len(s) + 1.0) for s in nfkc_comments_all], dtype=float
    )
    pre_txt["digit_count"] = np.array([sum(1 for ch in s if ch.isdigit()) for s in nfkc_comments_all], dtype=float)
    pre_txt["non_ascii_sym_cnt"] = np.array(
        [sum(1 for ch in s if ord(ch) > 0x2600) for s in nfkc_comments_all], dtype=float
    )

    def has_any(t_low, pats):
        return float(any(bool(re.search(p, t_low)) for p in pats))

    pre_txt["has_c1_pat"] = np.array([has_any(c.lower(), C1_PATTERNS) for c in nfkc_comments_all])
    pre_txt["has_c2_pat"] = np.array([has_any(c.lower(), C2_PATTERNS) for c in nfkc_comments_all])
    pre_txt["has_c3_pat"] = np.array([has_any(c.lower(), C3_PATTERNS) for c in nfkc_comments_all])

    # Delivery-only ratio vs product-specific content word overlap (key Class 2 vs Class 0 discriminator!)
    deliv_ratios = np.zeros(len(all_df), dtype=float)
    prod_spec_hits = np.zeros(len(all_df), dtype=float)
    prod_spec_ratios = np.zeros(len(all_df), dtype=float)
    desc_jacc = np.zeros(len(all_df), dtype=float)
    prod_jacc = np.zeros(len(all_df), dtype=float)
    desc_exact = np.zeros(len(all_df), dtype=float)

    for i, (c_s, p_s, d_s) in enumerate(zip(nfkc_comments_all, nfkc_prods_all, clean_descs_all)):
        cw = tokenize_words(c_s)
        pw = set(tokenize_words(p_s))
        dw = set(tokenize_words(d_s))
        if cw:
            d_cnt = sum(1 for w in cw if w in DELIVERY_PRAISE_STOPWORDS)
            deliv_ratios[i] = d_cnt / len(cw)
            content_cw = {w for w in cw if len(w) >= 3 and w not in DELIVERY_PRAISE_STOPWORDS}
            ref_vocab = pw | dw
            hits = len(content_cw & ref_vocab)
            prod_spec_hits[i] = float(hits)
            prod_spec_ratios[i] = hits / (len(content_cw) + 1.0)
            cw_set = set(cw)
            if dw:
                desc_jacc[i] = len(cw_set & dw) / len(cw_set | dw)
            if pw:
                prod_jacc[i] = len(cw_set & pw) / len(cw_set | pw)
        c_clean = " ".join(c_s.lower().split())[:260]
        d_clean = " ".join(d_s.lower().split())
        if len(c_clean) >= 28 and d_clean:
            for j in range(0, max(1, len(c_clean) - 27), 8):
                if c_clean[j : j + 28] in d_clean:
                    desc_exact[i] = 1.0
                    break

    pre_txt["delivery_word_ratio"] = deliv_ratios
    pre_txt["prod_specific_hits"] = prod_spec_hits
    pre_txt["prod_specific_ratio"] = prod_spec_ratios
    pre_txt["delivery_only_signal"] = deliv_ratios * (prod_spec_hits == 0).astype(float)
    pre_txt["desc_jaccard"] = desc_jacc
    pre_txt["prod_jaccard"] = prod_jacc
    pre_txt["desc_exact_slice"] = desc_exact

    for k, arr in pre_txt.items():
        all_df[f"_pre_{k}"] = arr

    def build_features(sub_df, split_idx, te_dict, knn_mat):
        cols = {}
        # 1. Per-stream probabilities, logits, margins, and entropies
        for s_name, s_tup, use_logits in active_streams:
            p_mat = s_tup[split_idx]
            for c in range(4):
                p = np.clip(p_mat[:, c], 1e-6, 1.0 - 1e-6)
                cols[f"{s_name}_p_{c}"] = p
                if use_logits:
                    cols[f"{s_name}_logit_{c}"] = np.log(p / (1.0 - p))
            sp = np.sort(p_mat, axis=1)
            cols[f"{s_name}_max_prob"] = sp[:, -1]
            cols[f"{s_name}_margin"] = sp[:, -1] - sp[:, -2]
            cols[f"{s_name}_entropy"] = -np.sum(p_mat * np.log(np.clip(p_mat, 1e-6, 1.0)), axis=1)

        ens_p = ens_all[split_idx]
        c_ens = ctx_ens[split_idx]
        m_ens = com_ens[split_idx]
        xctx_p = xctx[split_idx]
        pctx_p = pctx[split_idx]
        xcom_p = xcom[split_idx]
        tfidf_p = tfidf[split_idx]

        # Disagreement std across context models
        ctx_stack = np.stack([tup[split_idx] for _, tup, _ in ctx_parts], axis=0)  # (M, N, 4)
        ctx_std = np.std(ctx_stack, axis=0)

        for c in range(4):
            ep = np.clip(ens_p[:, c], 1e-6, 1.0 - 1e-6)
            cep = np.clip(c_ens[:, c], 1e-6, 1.0 - 1e-6)
            cols[f"ens_p_{c}"] = ep
            cols[f"ens_logit_{c}"] = np.log(ep / (1.0 - ep))
            cols[f"ctx_ens_p_{c}"] = cep
            cols[f"ctx_ens_logit_{c}"] = np.log(cep / (1.0 - cep))
            cols[f"com_ens_p_{c}"] = m_ens[:, c]
            cols[f"ctx_std_{c}"] = ctx_std[:, c]
            cols[f"causal_ctx_xlm_{c}"] = xctx_p[:, c] - xcom_p[:, c]
            cols[f"ctx_vs_com_shift_{c}"] = c_ens[:, c] - m_ens[:, c]
            cols[f"diff_xctx_pctx_{c}"] = xctx_p[:, c] - pctx_p[:, c]
            cols[f"diff_ens_tfidf_{c}"] = ens_p[:, c] - tfidf_p[:, c]

            if xlge is not None:
                cols[f"diff_xlge_xctx_{c}"] = xlge[split_idx][:, c] - xctx_p[:, c]
                cols[f"causal_xlge_xcom_{c}"] = xlge[split_idx][:, c] - xcom_p[:, c]
            if mdeb is not None:
                ref_l = xlge[split_idx][:, c] if xlge is not None else xctx_p[:, c]
                cols[f"diff_mdeb_xlm_{c}"] = mdeb[split_idx][:, c] - ref_l
                cols[f"causal_mdeb_deb_{c}"] = mdeb[split_idx][:, c] - deb[split_idx][:, c]
            if plge is not None:
                cols[f"diff_plge_pctx_{c}"] = plge[split_idx][:, c] - pctx_p[:, c]

        boundary_models = [("ens", ens_p), ("ctx_ens", c_ens), ("xctx", xctx_p), ("pctx", pctx_p)]
        if xlge is not None:
            boundary_models.append(("xlge", xlge[split_idx]))
        if mdeb is not None:
            boundary_models.append(("mdeb", mdeb[split_idx]))
        for prefix, mp in boundary_models:
            cols[f"{prefix}_diff_p0_p2"] = mp[:, 0] - mp[:, 2]
            cols[f"{prefix}_diff_p0_p3"] = mp[:, 0] - mp[:, 3]
            cols[f"{prefix}_diff_p1_p3"] = mp[:, 1] - mp[:, 3]
            cols[f"{prefix}_diff_p1_p0"] = mp[:, 1] - mp[:, 0]
            cols[f"{prefix}_diff_p2_p3"] = mp[:, 2] - mp[:, 3]

        preds_stack = np.column_stack([s_tup[split_idx].argmax(axis=1) for _, s_tup, _ in active_streams])
        n_m = float(preds_stack.shape[1])
        for c in range(4):
            cols[f"vote_frac_{c}"] = (preds_stack == c).sum(axis=1) / n_m

        # Group LOO & Target Encoding features
        group_cols = [
            c for c in sub_df.columns
            if c.startswith(("prod_loo_", "cat_path_loo_", "prod_count", "cat_path_count"))
        ]
        for col in group_cols:
            cols[col] = sub_df[col].to_numpy()
        for k, v in te_dict.items():
            cols[k] = v

        for c in range(4):
            cols[f"ctxens_minus_prod_loo_{c}"] = c_ens[:, c] - cols[f"prod_loo__ctxens_{c}"]
            cols[f"ctxens_minus_cat_loo_{c}"] = c_ens[:, c] - cols[f"cat_path_loo__ctxens_{c}"]
            cols[f"xctx_minus_prod_loo_{c}"] = xctx_p[:, c] - cols[f"prod_loo__xctx_{c}"]
            cols[f"pctx_minus_prod_loo_{c}"] = pctx_p[:, c] - cols[f"prod_loo__pctx_{c}"]
            cols[f"xcom_minus_prod_loo_{c}"] = xcom_p[:, c] - cols[f"prod_loo__xcom_{c}"]
            if xlge is not None:
                cols[f"xlge_minus_prod_loo_{c}"] = xlge[split_idx][:, c] - cols[f"prod_loo__xlge_{c}"]

        # Similarity, KNN & Full 16-component SVD features
        for k in ["cos_c_p", "cos_c_d", "cos_c_cat", "ov5_ratio", "win_ov"]:
            cols[k] = sub_df[f"_sim_{k}"].to_numpy()
        for u_name in unsup_names:
            cols[u_name] = sub_df[f"_u_{u_name}"].to_numpy()
        for c in range(4):
            cols[f"knn_max_sim_{c}"] = knn_mat[:, c]
            cols[f"knn_prob_{c}"] = knn_mat[:, 4 + c]
        for idx in range(n_svd):
            cols[f"svd_{idx}"] = sub_df[f"_svd_{idx}"].to_numpy()

        # Numeric & Categorical Metadata
        rating = sub_df["rating"].fillna(5).astype(float).to_numpy()
        cols["rating"] = rating
        cols["rating_diff_prod"] = rating - cols["prod_loo_rating"]
        cols["is_rating_5"] = (rating == 5).astype(float)
        cols["is_rating_low"] = (rating <= 2).astype(float)
        for c in range(4):
            cols[f"rating_x_ens_p{c}"] = rating * ens_p[:, c]

        num_sold = sub_df["num_sold"].fillna(0).astype(float).to_numpy()
        num_rev = sub_df["num_review"].fillna(0).astype(float).to_numpy()
        cols["log_num_sold"] = np.log1p(num_sold)
        cols["log_num_review"] = np.log1p(num_rev)
        cols["is_zero_sold"] = (num_sold == 0).astype(float)
        cols["is_shopee_rounded_sold"] = ((num_sold >= 1000) & (np.mod(num_sold, 100) == 0)).astype(float)
        cols["ratio_rev_sold"] = (num_rev + 1.0) / (num_sold + 1.0)
        cols["category_code"] = sub_df["category_code"].to_numpy()
        cols["plat_cat_code"] = sub_df["plat_cat_code"].to_numpy()
        cols["comment_global_freq"] = sub_df["comment_global_freq"].to_numpy()

        # Precomputed lexical, platform, shop-promo, and delivery-vs-product features
        for k in pre_txt:
            cols[k] = sub_df[f"_pre_{k}"].to_numpy()
        cols["char_len_ratio_prod"] = cols["char_len"] / (cols["prod_loo__char_len"] + 1.0)

        # High-order domain x neural interactions
        cols["desc_copy_signal"] = (
            np.maximum(cols["ov5_ratio"], cols["win_ov"]) * (0.5 + 0.5 * cols["desc_exact_slice"])
            + 0.35 * cols["has_spec_header"]
            + 0.35 * cols["has_brand_promo"]
        )
        cols["ctx_copy_boost"] = c_ens[:, 1] * (
            cols["ov5_ratio"] + cols["win_ov"] + 0.5 * cols["has_spec_header"] + 0.5 * cols["has_brand_promo"]
        )
        cols["shopee_c1_prior_boost"] = c_ens[:, 1] * (1.0 - cols["is_tiki_plat"]) * (1.0 + 0.4 * cols["shopee_chip_cnt"])
        cols["c2_delivery_boost"] = c_ens[:, 2] * (0.5 + cols["delivery_word_ratio"]) * (1.0 / (1.0 + cols["prod_specific_hits"]))

        return pd.DataFrame(cols, index=sub_df.index)

    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_part = all_df.iloc[:n_tr]
    dev_part = all_df.iloc[n_tr : n_tr + n_de]
    test_part = all_df.iloc[n_tr + n_de :]

    te_tr_5, (te_de_5, te_te_5) = compute_oof_target_encoding(train_part, labels, [dev_part, test_part], skf5)
    X_tr_a = build_features(train_part, 0, te_tr_5, ks_npz["knn_a_oof"])
    X_de_a = build_features(dev_part, 1, te_de_5, ks_npz["knn_a_dev"])
    X_te_a = build_features(test_part, 2, te_te_5, ks_npz["knn_a_test"])
    print(f"   Total engineered meta-feature count: {X_tr_a.shape[1]}")

    # Select continuous, non-ID features for the smooth Residual Meta-MLP
    mlp_cols = [
        c for c in X_tr_a.columns
        if c not in ("category_code", "plat_cat_code")
    ]

    balanced_weights = np.array([1.0, 3.6, 1.8, 1.3])

    lgb_d4_params = {
        "objective": "multiclass", "num_class": 4, "metric": "multi_logloss",
        "feature_pre_filter": False,
        "learning_rate": 0.028, "num_leaves": 19, "max_depth": 4, "min_child_samples": 30,
        "feature_fraction": 0.52, "bagging_fraction": 0.80, "bagging_freq": 1,
        "reg_alpha": 0.6, "reg_lambda": 2.2, "verbose": -1, "n_jobs": -1,
    }
    lgb_d5_params = {
        "objective": "multiclass", "num_class": 4, "metric": "multi_logloss",
        "feature_pre_filter": False,
        "learning_rate": 0.024, "num_leaves": 26, "max_depth": 5, "min_child_samples": 34,
        "feature_fraction": 0.46, "bagging_fraction": 0.78, "bagging_freq": 1,
        "reg_alpha": 0.9, "reg_lambda": 3.2, "verbose": -1, "n_jobs": -1,
    }
    lgb_et_params = {
        "objective": "multiclass", "num_class": 4, "metric": "multi_logloss",
        "feature_pre_filter": False,
        "extra_trees": True, "learning_rate": 0.028, "num_leaves": 24, "max_depth": 5,
        "min_child_samples": 28, "feature_fraction": 0.45, "bagging_fraction": 0.82, "bagging_freq": 1,
        "reg_alpha": 0.7, "reg_lambda": 2.8, "verbose": -1, "n_jobs": -1,
    }
    xgb_d4_params = {
        "objective": "multi:softprob", "num_class": 4, "eval_metric": "mlogloss",
        "learning_rate": 0.032, "max_depth": 4, "min_child_weight": 4.0,
        "subsample": 0.80, "colsample_bytree": 0.50,
        "reg_alpha": 0.6, "reg_lambda": 2.5, "tree_method": "hist", "nthread": -1,
    }
    xgb_d5_params = {
        "objective": "multi:softprob", "num_class": 4, "eval_metric": "mlogloss",
        "learning_rate": 0.026, "max_depth": 5, "min_child_weight": 5.5,
        "subsample": 0.78, "colsample_bytree": 0.45,
        "reg_alpha": 0.9, "reg_lambda": 3.5, "tree_method": "hist", "nthread": -1,
    }
    xgb_d3_params = {
        "objective": "multi:softprob", "num_class": 4, "eval_metric": "mlogloss",
        "learning_rate": 0.038, "max_depth": 3, "min_child_weight": 3.5,
        "subsample": 0.82, "colsample_bytree": 0.60,
        "reg_alpha": 0.4, "reg_lambda": 2.0, "tree_method": "hist", "nthread": -1,
    }

    def run_meta_cv(X_train_all, y_all, X_eval_list, skf, seed_offset=0, cat_es=55, full_stage_b=False):
        n_samples = len(X_train_all)
        n_splits = skf.get_n_splits()
        w_all = balanced_weights[y_all]

        keys = ("cat6", "cat5", "lgb", "lgb_et", "xgb", "mlp") if full_stage_b else ("cat6", "cat5", "lgb", "xgb")
        oof_mc = {k: np.zeros((n_samples, 4)) for k in keys}
        eval_mc = [{k: np.zeros((len(Xe), 4)) for k in keys} for Xe in X_eval_list]

        for fold, (t_idx, v_idx) in enumerate(skf.split(X_train_all, y_all), 1):
            seed_f = 42 + seed_offset + fold
            X_tf, y_tf, w_tf = X_train_all.iloc[t_idx], y_all[t_idx], w_all[t_idx]
            X_vf, y_vf, w_vf = X_train_all.iloc[v_idx], y_all[v_idx], w_all[v_idx]

            tr_pool = Pool(X_tf, label=y_tf, weight=w_tf)
            va_pool = Pool(X_vf, label=y_vf, weight=w_vf)

            # 1. CatBoost Depth-6
            mc6 = CatBoostClassifier(
                iterations=1150, learning_rate=0.036, depth=6, l2_leaf_reg=6.0,
                loss_function="MultiClass", random_seed=seed_f, verbose=False,
            )
            mc6.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=cat_es, verbose=False)
            oof_mc["cat6"][v_idx] = mc6.predict_proba(X_vf)
            for idx, Xe in enumerate(X_eval_list):
                eval_mc[idx]["cat6"] += mc6.predict_proba(Xe) / n_splits

            # 2. CatBoost Depth-5 High-Regularization
            mc5 = CatBoostClassifier(
                iterations=1200, learning_rate=0.034, depth=5, l2_leaf_reg=8.5, random_strength=1.5,
                loss_function="MultiClass", random_seed=seed_f + 77, verbose=False,
            )
            mc5.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=cat_es, verbose=False)
            oof_mc["cat5"][v_idx] = mc5.predict_proba(X_vf)
            for idx, Xe in enumerate(X_eval_list):
                eval_mc[idx]["cat5"] += mc5.predict_proba(Xe) / n_splits

            # 3. LightGBM (Bagged d4 + d5) & ExtraTrees
            dtr = lgb.Dataset(
                X_tf, label=y_tf, weight=w_tf, free_raw_data=False, params={"feature_pre_filter": False}
            )
            dva = lgb.Dataset(
                X_vf, label=y_vf, weight=w_vf, reference=dtr, free_raw_data=False, params={"feature_pre_filter": False}
            )
            ml1 = lgb.train(
                dict(lgb_d4_params, random_state=seed_f), dtr, num_boost_round=800,
                valid_sets=[dva], callbacks=[lgb.early_stopping(45, verbose=False)],
            )
            if full_stage_b:
                ml2 = lgb.train(
                    dict(lgb_d5_params, random_state=seed_f + 131), dtr, num_boost_round=850,
                    valid_sets=[dva], callbacks=[lgb.early_stopping(45, verbose=False)],
                )
                oof_mc["lgb"][v_idx] = 0.55 * ml1.predict(X_vf) + 0.45 * ml2.predict(X_vf)
                for idx, Xe in enumerate(X_eval_list):
                    eval_mc[idx]["lgb"] += (0.55 * ml1.predict(Xe) + 0.45 * ml2.predict(Xe)) / n_splits

                m_et = lgb.train(
                    dict(lgb_et_params, random_state=seed_f + 257), dtr, num_boost_round=900,
                    valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                oof_mc["lgb_et"][v_idx] = m_et.predict(X_vf)
                for idx, Xe in enumerate(X_eval_list):
                    eval_mc[idx]["lgb_et"] += m_et.predict(Xe) / n_splits
            else:
                oof_mc["lgb"][v_idx] = ml1.predict(X_vf)
                for idx, Xe in enumerate(X_eval_list):
                    eval_mc[idx]["lgb"] += ml1.predict(Xe) / n_splits

            # 4. XGBoost (3-Profile Bagged d4 + d5 + d3 in Stage B)
            dxtr = xgb.DMatrix(X_tf, label=y_tf, weight=w_tf)
            dxva = xgb.DMatrix(X_vf, label=y_vf, weight=w_vf)
            mx1 = xgb.train(
                dict(xgb_d4_params, random_state=seed_f), dxtr, num_boost_round=800,
                evals=[(dxva, "val")], early_stopping_rounds=45, verbose_eval=False,
            )
            if full_stage_b:
                mx2 = xgb.train(
                    dict(xgb_d5_params, random_state=seed_f + 131), dxtr, num_boost_round=850,
                    evals=[(dxva, "val")], early_stopping_rounds=45, verbose_eval=False,
                )
                mx3 = xgb.train(
                    dict(xgb_d3_params, random_state=seed_f + 271), dxtr, num_boost_round=850,
                    evals=[(dxva, "val")], early_stopping_rounds=45, verbose_eval=False,
                )
                oof_mc["xgb"][v_idx] = 0.45 * mx1.predict(dxva) + 0.35 * mx2.predict(dxva) + 0.20 * mx3.predict(dxva)
                for idx, Xe in enumerate(X_eval_list):
                    dxe = xgb.DMatrix(Xe)
                    eval_mc[idx]["xgb"] += (
                        0.45 * mx1.predict(dxe) + 0.35 * mx2.predict(dxe) + 0.20 * mx3.predict(dxe)
                    ) / n_splits

                # 5. Residual Multi-Sample-Dropout Meta-MLP (smooth non-tree meta-learner)
                va_mlp, ev_mlps = fit_predict_meta_mlp_cv_fold(
                    X_tf[mlp_cols].to_numpy(),
                    y_tf,
                    X_vf[mlp_cols].to_numpy(),
                    y_vf,
                    [Xe[mlp_cols].to_numpy() for Xe in X_eval_list],
                    cls_weights_np=balanced_weights,
                    seed=seed_f + 503,
                )
                oof_mc["mlp"][v_idx] = va_mlp
                for idx in range(len(X_eval_list)):
                    eval_mc[idx]["mlp"] += ev_mlps[idx] / n_splits
            else:
                oof_mc["xgb"][v_idx] = mx1.predict(dxva)
                for idx, Xe in enumerate(X_eval_list):
                    eval_mc[idx]["xgb"] += mx1.predict(xgb.DMatrix(Xe)) / n_splits

        return oof_mc, eval_mc

    print("3. Training Stage A (5-Fold Meta-Ensemble on Train = 14,306 rows)...")
    oof_mc_a, (dev_mc_a, test_mc_a) = run_meta_cv(
        X_tr_a, labels, [X_de_a, X_te_a], skf5, seed_offset=0, cat_es=55, full_stage_b=False
    )
    for k in ("cat6", "cat5", "lgb", "xgb"):
        print(
            f"   [Stage A {k.upper():6s}] Train OOF F1: {fast_macro_f1(labels, oof_mc_a[k].argmax(1)):.5f} "
            f"| Dev Pred F1: {fast_macro_f1(dev_labels, dev_mc_a[k].argmax(1)):.5f}"
        )

    print("4. Training Stage B (10-Fold 6-Family Ensemble on Full Train+Dev = 15,896 rows)...")
    full_part = all_df.iloc[: n_tr + n_de].reset_index(drop=True)
    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=102)
    te_full_10, (te_te_10,) = compute_oof_target_encoding(full_part, y_full, [test_part], skf10)

    # Temporarily rebuild split_idx=0+1 stacked streams for Full 15,896 Stage B
    stacked_streams_backup = active_streams[:]
    orig_ctx_ens, orig_com_ens, orig_ens_all = ctx_ens, com_ens, ens_all
    orig_xctx, orig_pctx, orig_xcom, orig_deb, orig_tfidf = xctx, pctx, xcom, deb, tfidf
    orig_xlge, orig_mdeb, orig_plge = xlge, mdeb, plge

    def stack_tr_de(tup):
        return (np.vstack([tup[0], tup[1]]), tup[1], tup[2])

    active_streams = [(n, stack_tr_de(t), u) for n, t, u in stacked_streams_backup]
    ctx_parts = [(n, stack_tr_de(t), w) for n, t, w in ctx_parts]
    ctx_ens = stack_tr_de(orig_ctx_ens)
    com_ens = stack_tr_de(orig_com_ens)
    ens_all = stack_tr_de(orig_ens_all)
    xctx = stack_tr_de(orig_xctx)
    pctx = stack_tr_de(orig_pctx)
    xcom = stack_tr_de(orig_xcom)
    deb = stack_tr_de(orig_deb)
    tfidf = stack_tr_de(orig_tfidf)
    if orig_xlge is not None:
        xlge = stack_tr_de(orig_xlge)
    if orig_mdeb is not None:
        mdeb = stack_tr_de(orig_mdeb)
    if orig_plge is not None:
        plge = stack_tr_de(orig_plge)

    X_full_b = build_features(all_df.iloc[: n_tr + n_de], 0, te_full_10, ks_npz["knn_b_full"])
    X_te_b = build_features(test_part, 2, te_te_10, ks_npz["knn_b_test"])

    oof_mc_b, (test_mc_b,) = run_meta_cv(
        X_full_b, y_full, [X_te_b], skf10, seed_offset=300, cat_es=85, full_stage_b=True
    )
    for k in ("cat6", "cat5", "lgb", "lgb_et", "xgb", "mlp"):
        print(
            f"   [Stage B {k.upper():6s}] Raw Train-Slice F1: {fast_macro_f1(labels, oof_mc_b[k][:n_tr].argmax(1)):.5f} "
            f"| Full 15,896 OOF F1: {fast_macro_f1(y_full, oof_mc_b[k].argmax(1)):.5f}"
        )

    # Assemble candidate Level-3 component streams on Full 15,896 & Test 3,974
    stage_a_full = 0.35 * np.vstack([oof_mc_a["xgb"], dev_mc_a["xgb"]]) + \
                   0.35 * np.vstack([oof_mc_a["cat6"], dev_mc_a["cat6"]]) + \
                   0.30 * np.vstack([oof_mc_a["lgb"], dev_mc_a["lgb"]])
    stage_a_test = 0.35 * test_mc_a["xgb"] + 0.35 * test_mc_a["cat6"] + 0.30 * test_mc_a["lgb"]

    neural_full = 0.80 * ctx_ens[0] + 0.20 * com_ens[0]
    neural_test = 0.80 * ctx_ens[2] + 0.20 * com_ens[2]

    comp_names = ["b_xgb", "b_lgb", "b_lgb_et", "b_cat6", "b_cat5", "b_mlp", "stage_a", "neural"]
    comp_oofs = [
        oof_mc_b["xgb"], oof_mc_b["lgb"], oof_mc_b["lgb_et"],
        oof_mc_b["cat6"], oof_mc_b["cat5"], oof_mc_b["mlp"],
        stage_a_full, neural_full,
    ]
    comp_tests = [
        test_mc_b["xgb"], test_mc_b["lgb"], test_mc_b["lgb_et"],
        test_mc_b["cat6"], test_mc_b["cat5"], test_mc_b["mlp"],
        stage_a_test, neural_test,
    ]
    prior_weights = [0.30, 0.16, 0.12, 0.14, 0.08, 0.08, 0.06, 0.06]

    # Optional light anchor to Submission 15 (capped <= 0.08, kept ONLY if helpful in CV)
    if (OUTPUT / "submission_15_oof_probs.npy").exists():
        s15_full = np.vstack([
            np.load(OUTPUT / "submission_15_oof_probs.npy"),
            np.load(OUTPUT / "submission_15_dev_probs.npy"),
        ])
        s15_test = np.load(OUTPUT / "submission_15_test_probs.npy")
        comp_names.append("s15_anchor")
        comp_oofs.append(s15_full)
        comp_tests.append(s15_test)
        prior_weights = [0.29, 0.16, 0.11, 0.13, 0.08, 0.08, 0.05, 0.06, 0.04]

    prior_w = np.array(prior_weights, dtype=float)
    prior_w /= prior_w.sum()
    oof_tensor = np.stack(comp_oofs, axis=0)   # (K, 15896, 4)
    test_tensor = np.stack(comp_tests, axis=0)  # (K, 3974, 4)
    n_k = len(comp_names)

    print("\n5. Optimizing Level-3 Ensemble Weights & Multipliers via Strict 5-Fold Nested CV...")
    skf_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    nested_preds = np.zeros(len(y_full), dtype=int)
    fold_alphas = []
    fold_mults = []

    rng_opt = np.random.default_rng(2026)
    # Pre-sample regularized Dirichlet candidate blend vectors around prior_w
    cand_alphas = [prior_w.copy()]
    for conc in (18.0, 35.0, 70.0):
        for _ in range(180):
            a = rng_opt.dirichlet(prior_w * conc + 0.3)
            if comp_names[-1] == "s15_anchor" and a[-1] > 0.08:
                a[-1] = 0.06
                a /= a.sum()
            # Prevent single-model collapse: max weight <= 0.46
            if a.max() <= 0.46:
                cand_alphas.append(a)
    cand_alphas = np.array(cand_alphas)

    for tr_i, va_i in skf_cal.split(np.zeros(len(y_full)), y_full):
        y_sub = y_full[tr_i]
        sub_tensor = oof_tensor[:, tr_i, :]  # (K, n_sub, 4)

        best_alpha = prior_w.copy()
        best_blend_score = -1.0
        for a in cand_alphas:
            p_blend = np.tensordot(a, sub_tensor, axes=(0, 0))
            reg_pen = 0.004 * float(np.sum((a - prior_w) ** 2))
            sc = fast_macro_f1(y_sub, p_blend.argmax(1)) - reg_pen
            if sc > best_blend_score:
                best_blend_score = sc
                best_alpha = a.copy()

        # Coordinate simplex polish around best_alpha
        for step_sz in (0.04, 0.02):
            for k_idx in range(n_k):
                for delta in (-step_sz, step_sz):
                    a_try = best_alpha.copy()
                    a_try[k_idx] = max(0.0, a_try[k_idx] + delta)
                    if a_try.sum() <= 1e-6:
                        continue
                    a_try /= a_try.sum()
                    if comp_names[-1] == "s15_anchor" and a_try[-1] > 0.08:
                        continue
                    if a_try.max() > 0.46:
                        continue
                    p_blend = np.tensordot(a_try, sub_tensor, axes=(0, 0))
                    sc = fast_macro_f1(y_sub, p_blend.argmax(1)) - 0.004 * float(np.sum((a_try - prior_w) ** 2))
                    if sc > best_blend_score:
                        best_blend_score = sc
                        best_alpha = a_try.copy()

        p_sub = np.tensordot(best_alpha, sub_tensor, axes=(0, 0))
        best_s = -1.0
        bw = np.ones(4)
        for w1 in np.linspace(0.96, 1.16, 21):
            for w2 in np.linspace(0.95, 1.05, 11):
                for w3 in np.linspace(0.97, 1.03, 7):
                    w = np.array([1.0, w1, w2, w3])
                    s = fast_macro_f1(y_sub, (p_sub * w).argmax(1)) - 0.0015 * float(np.sum((w - 1.0) ** 2))
                    if s > best_s:
                        best_s = s
                        bw = w.copy()

        p_va = np.tensordot(best_alpha, oof_tensor[:, va_i, :], axes=(0, 0))
        nested_preds[va_i] = (p_va * bw).argmax(1)
        fold_alphas.append(best_alpha)
        fold_mults.append(bw)

    best_alpha = np.mean(fold_alphas, axis=0)
    best_alpha /= best_alpha.sum()
    best_w = np.mean(fold_mults, axis=0)

    full_comb_oof = np.tensordot(best_alpha, oof_tensor, axes=(0, 0))
    final_test = np.tensordot(best_alpha, test_tensor, axes=(0, 0))
    final_oof = full_comb_oof[:n_tr]
    final_dev = full_comb_oof[n_tr:]

    raw_oof_f1 = fast_macro_f1(labels, final_oof.argmax(axis=1))
    raw_full_f1 = fast_macro_f1(y_full, full_comb_oof.argmax(axis=1))
    nested_tr_f1 = fast_macro_f1(labels, nested_preds[:n_tr])
    nested_full_f1 = fast_macro_f1(y_full, nested_preds)

    oof_preds = (final_oof * best_w).argmax(axis=1)
    dev_preds = (final_dev * best_w).argmax(axis=1)
    full_preds = (full_comb_oof * best_w).argmax(axis=1)
    test_preds = (final_test * best_w).argmax(axis=1)

    oof_macro = float(f1_score(labels, oof_preds, average="macro"))
    dev_macro = float(f1_score(dev_labels, dev_preds, average="macro"))
    full_macro = float(f1_score(y_full, full_preds, average="macro"))

    print("\n" + "=" * 78)
    print("Learned Level-3 Blend Weights:")
    for name, wt in zip(comp_names, best_alpha):
        print(f"  - {name:12s}: {wt:.4f}")
    print(f"Nested-CV Averaged Calibration Weights:  {np.round(best_w, 4).tolist()}")
    print("-" * 78)
    print(f"Raw Uncalibrated Train OOF Macro F1:     {raw_oof_f1:.5f}")
    print(f"Raw Uncalibrated Full  OOF Macro F1:     {raw_full_f1:.5f}  (Sub16 was 0.80384)")
    print(f"Strict Nested-CV Train OOF Macro F1:     {nested_tr_f1:.5f}  (Sub16 was 0.80764)")
    print(f"Strict Nested-CV Full  OOF Macro F1:     {nested_full_f1:.5f}  (Sub16 was 0.80279)")
    print(f"Final Train OOF Macro F1 (14,306 rows):  {oof_macro:.5f}  (Sub16 was 0.81053)")
    print(f"Final Full  OOF Macro F1 (15,896 rows):  {full_macro:.5f}  (Sub16 was 0.80558 -> 0.80557 LB!)")
    print(f"Final Dev   OOF Macro F1  (1,590 rows):  {dev_macro:.5f}  (Sub16 was 0.74354)")
    print("=" * 78)

    print("\nDetailed Classification Report on Full Train+Dev OOF (15,896 rows):")
    print(classification_report(y_full, full_preds, digits=4))

    p_s17_oof = OUTPUT / "submission_17_oof_probs.npy"
    p_s17_dev = OUTPUT / "submission_17_dev_probs.npy"
    p_s17_tst = OUTPUT / "submission_17_test_probs.npy"
    p_s17_npz = OUTPUT / "submission_17_components.npz"
    p_s17_json = OUTPUT / "submission_17_metrics.json"
    sub17_out = OUTPUT / "submission_17.csv"
    sub17_root = ROOT / "submission_17.csv"

    np.save(p_s17_oof, final_oof)
    np.save(p_s17_dev, final_dev)
    np.save(p_s17_tst, final_test)
    np.savez_compressed(
        p_s17_npz,
        **{f"a_oof_{k}": oof_mc_a[k] for k in oof_mc_a},
        **{f"a_dev_{k}": dev_mc_a[k] for k in dev_mc_a},
        **{f"a_test_{k}": test_mc_a[k] for k in test_mc_a},
        **{f"b_oof_{k}": oof_mc_b[k] for k in oof_mc_b},
        **{f"b_test_{k}": test_mc_b[k] for k in test_mc_b},
    )

    unique_preds = np.unique(test_preds)
    print(f"Unique test prediction classes: {unique_preds.tolist()}")
    if len(unique_preds) < 4:
        raise ValueError(f"CRITICAL ERROR: test_preds collapsed! Only classes {unique_preds} present.")
    if np.isnan(final_test).any() or np.isnan(final_oof).any():
        raise ValueError("CRITICAL ERROR: probabilities contain NaN values!")

    submission_17 = make_submission(test, sample, test_preds)
    submission_17.to_csv(sub17_out, index=False)
    submission_17.to_csv(sub17_root, index=False)
    print(f"Saved submission to:\n  - {sub17_out}\n  - {sub17_root}")

    test_dist = {int(k): int(v) for k, v in pd.Series(test_preds).value_counts().items()}
    print(f"Test Class Distribution: {test_dist}")
    if (ROOT / "submission_16.csv").exists():
        s16_df = pd.read_csv(ROOT / "submission_16.csv")
        agree = float((s16_df[" Label"] == submission_17[" Label"]).mean()) if " Label" in s16_df.columns else float((s16_df.iloc[:, 1] == submission_17.iloc[:, 1]).mean())
        print(f"Agreement with Submission 16 (0.80557 LB): {agree * 100:.2f}%")

    metrics = {
        "model": "Sub17: Multi-Foundation A100 Context Ensemble + Platform/Promo/Delivery Signals + 6-Family 10-Fold Meta-Stacking (XGB+LGB+ExtraTrees+CatBoost+ResidualMLP)",
        "active_l1_streams": [n for n, _, _ in stacked_streams_backup],
        "features_count": int(X_full_b.shape[1]),
        "blend_weights": {n: round(float(w), 5) for n, w in zip(comp_names, best_alpha)},
        "calibration_multipliers": best_w.tolist(),
        "raw_oof_macro_f1": raw_oof_f1,
        "raw_full_oof_macro_f1": raw_full_f1,
        "strict_nested_cv_oof_macro_f1": nested_tr_f1,
        "strict_nested_cv_full_macro_f1": nested_full_f1,
        "oof_macro_f1": oof_macro,
        "full_15896_oof_macro_f1": full_macro,
        "dev_macro_f1": dev_macro,
        "test_distribution": test_dist,
        "elapsed_minutes": round((time.time() - t_start) / 60.0, 2),
    }
    p_s17_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Backup all generated artifacts to Google Drive BEFORE submitting so nothing can ever be lost!
    backup_to_cache([sub17_root, p_s17_oof, p_s17_dev, p_s17_tst, p_s17_npz, p_s17_json], cache_dir)

    if args.no_submit:
        print("Skipping Kaggle CLI submission (--no-submit specified).")
        return

    local_win_kaggle = ROOT.parent / ".venv" / "Scripts" / "kaggle.exe"
    kaggle_cmd_name = str(local_win_kaggle) if local_win_kaggle.exists() else shutil.which("kaggle")

    if kaggle_cmd_name:
        streams_str = "+".join(n for n, _, _ in stacked_streams_backup if n in ("xlge", "mdeb", "plge", "xctx2", "xctx"))
        msg = f"Sub17: A100 Multi-Backbone ({streams_str}) + 6-Family Meta-Stacking (Full OOF={full_macro:.5f}, Nested={nested_full_f1:.5f})"
        submit_cmd = [
            kaggle_cmd_name, "competitions", "submit",
            "-c", "ds-102-r-11-cnvn",
            "-f", str(sub17_root),
            "-m", msg,
        ]
        print(f"\nAuto-submitting submission_17.csv to Kaggle competition 'ds-102-r-11-cnvn'...")
        res = subprocess.run(submit_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)

        print("Waiting 20 seconds to fetch official Public Leaderboard score...")
        time.sleep(20)
        score_cmd = [kaggle_cmd_name, "competitions", "submissions", "-c", "ds-102-r-11-cnvn"]
        score_res = subprocess.run(score_cmd, capture_output=True, text=True)
        print("\n=== LATEST SUBMISSION STATUS ===")
        print("\n".join(score_res.stdout.splitlines()[:8]))

        lb_cmd = [kaggle_cmd_name, "competitions", "leaderboard", "-c", "ds-102-r-11-cnvn", "-s"]
        lb_res = subprocess.run(lb_cmd, capture_output=True, text=True)
        if lb_res.stdout:
            print("\n=== TOP LEADERBOARD STANDINGS ===")
            print("\n".join(lb_res.stdout.splitlines()[:12]))


if __name__ == "__main__":
    main()
