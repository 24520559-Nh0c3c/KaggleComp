"""Championship Stacking Pipeline for DS102 Spam Classification (Submission 15 — Target: Beat 0.80716 Top 1)

Major architectural innovations over Submission 14 (0.80036 LB):
  1. Dual Context-Aware Multilingual & Vietnamese Transformers + Comment-Only TTA Baselines:
     Integrates 5-Fold Context-Aware XLM-RoBERTa ([CLS]+MeanPool+5x Multi-Sample Dropout trained on
     synchronized Train+Dev = 15,896 rows, achieving 0.7900 standalone OOF F1!) alongside 5-Fold
     Context-Aware PhoBERT (0.7763 OOF F1), Comment-Only XLM-R TTA, DeBERTa-v3, RoBERTa, and 5-Fold TF-IDF.
     Computes pure causal context-shift features (`xctx - xcom` on the exact same XLM-R backbone!).
  2. Platform-Wide & Intra-Product Seeding / Near-Duplicate Graph Features (`knn_svd_features.npz`)
     + Contextual Verbatim Copy-Paste Interaction Signals:
     - Unsupervised near-duplicate & intra-product review similarity across all 19,870 reviews.
     - Pool-size-matched Out-of-Fold KNN Neighbor probabilities & max-class similarities.
     - Top TruncatedSVD semantic coordinates + Contextual Copy-Paste Boost (`ctx_copy_boost`, `desc_copy_signal`).
  3. 80-Model Multi-Seed / Multi-Architecture Level-2 Ensemble (Stage A 5-Fold + Stage B 10-Fold Full 15,896):
     - Stage A (5-Fold on Train): CatBoost-d6 + CatBoost-d5 + Reg-LGBM-d4 + Reg-XGB-d4.
     - Stage B (10-Fold on Train+Dev = 15,896 rows): 2-Seed/Depth Bagged XGBoost (d4 + d5) +
       2-Seed/Depth Bagged LightGBM (d4 + d5) + CatBoost-d6 + CatBoost-d5 (early_stopping=85).
     - Strict 5-Fold Nested-CV Calibration across all 15,896 labeled reviews.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from spam_pipeline import LABEL, load_frame, make_submission

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "spam_output"

C1_PATTERNS = [
    r"gửi gắm rất nhiều tâm tư", r"không thể trông đợi gì hơn", r"văn mẫu",
    r"tâm tư và tình cảm", r"người bán đã gửi gắm",
    r"dành cho ng tiêu dùng chúng tôi 1 sản phẩm", r"xinh đẹp tuyệt vời",
    r"tuyệt vời.*tuyệt vời", r"tuyệt vời.*đóng gói", r"chất lượng sản phẩm tuyệt vời",
]
C2_PATTERNS = [
    r"chưa (dùng|sử dụng|xài|biết|thử|mở|uống|ăn|đeo|mặc|cầm)",
    r"chưa rõ", r"chưa test", r"mua tặng", r"mua hộ", r"mua giùm",
    r"kiểm chứng", r"thời gian sau mới biết", r"thời gian kiểm chứng",
    r"dịch vụ", r"shipper", r"giao hàng", r"đóng gói", r"bảo hành",
    r"nhận xe", r"đặt hôm", r"hẹn thứ",
]
C3_PATTERNS = [
    r"[a-zA-Z0-9]{14,}", r"(.)\1{4,}",
    r"(zalo|sđt|hotline|liên hệ|inbox|0[35789]\d{8})",
    r"(http|https|www\.)", r"\[tb\]", r"\bvoucher\b",
]
SPEC_HEADER_PAT = re.compile(
    r"(thông số|thành phần|cách dùng|hướng dẫn sử dụng|lưu ý\s*:|tên sản phẩm|"
    r"thương hiệu\s*:|tác dụng\s*:|điểm nổi bật|cơ chế|thời gian giao hàng dự kiến|"
    r"gói hàng bao gồm|quý khách lưu ý|cam kết hàng chính hãng)"
)
EVOUCHER_PAT = re.compile(r"(e-?voucher|\bvoucher\b|thẻ nạp|thẻ cào|mã thẻ|nạp tiền điện thoại|gói cước)")


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

    for group_col, prefix, smoothing in [("product_name", "prod_te", 5.0), ("categories", "cat_te", 10.0)]:
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


def main():
    print("1. Loading datasets, 6 Level-1 probability streams, and KNN/SVD features...")
    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")
    sample = pd.read_csv(ROOT / "ds102_samplesubmission.csv", encoding="utf-8-sig")

    labels = train[LABEL].to_numpy()
    dev_labels = dev[LABEL].to_numpy()
    n_tr, n_de = len(train), len(dev)

    xlm_raw_oof = np.load(OUTPUT / "xlm_oof_probabilities.npy")
    xlm_raw_dev = np.load(OUTPUT / "xlm_dev_probabilities.npy")
    xlm_raw_test = np.load(OUTPUT / "xlm_test_probabilities.npy")

    xlm_nfkc_oof = np.load(OUTPUT / "xlm_nfkc_oof.npy")
    xlm_nfkc_dev = np.load(OUTPUT / "xlm_nfkc_dev.npy")
    xlm_nfkc_test = np.load(OUTPUT / "xlm_nfkc_test.npy")

    xcom_oof = 0.5 * xlm_raw_oof + 0.5 * xlm_nfkc_oof
    xcom_dev = 0.5 * xlm_raw_dev + 0.5 * xlm_nfkc_dev
    xcom_test = 0.5 * xlm_raw_test + 0.5 * xlm_nfkc_test

    xctx_oof = np.load(OUTPUT / "xlm_ctx_oof.npy")
    xctx_dev = np.load(OUTPUT / "xlm_ctx_dev.npy")
    xctx_test = np.load(OUTPUT / "xlm_ctx_test.npy")

    pctx_oof = np.load(OUTPUT / "phobert_ctx_oof.npy")
    pctx_dev = np.load(OUTPUT / "phobert_ctx_dev.npy")
    pctx_test = np.load(OUTPUT / "phobert_ctx_test.npy")

    deb_oof = np.load(OUTPUT / "deberta_oof_probabilities.npy")
    deb_dev = np.load(OUTPUT / "deberta_dev_probabilities.npy")
    deb_test = np.load(OUTPUT / "deberta_test_probabilities.npy")

    tfidf_oof = np.load(OUTPUT / "tfidf_oof_probabilities.npy")
    tfidf_dev = np.load(OUTPUT / "tfidf_dev_probabilities.npy")
    tfidf_test = np.load(OUTPUT / "tfidf_test_probabilities.npy")

    all_oof = np.load(OUTPUT / "oof_probabilities.npy")
    all_dev = np.load(OUTPUT / "dev_probabilities.npy")
    all_test = np.load(OUTPUT / "test_probabilities.npy")
    rob_oof = recover_roberta(all_oof, xlm_raw_oof, deb_oof)
    rob_dev = recover_roberta(all_dev, xlm_raw_dev, deb_dev)
    rob_test = recover_roberta(all_test, xlm_raw_test, deb_test)

    sim_npz = np.load(OUTPUT / "nfkc_sim_features.npz")
    ks_npz = np.load(OUTPUT / "knn_svd_features.npz")

    all_df = pd.concat([
        train.assign(_split="train"),
        dev.assign(_split="dev"),
        test.assign(_split="test"),
    ], ignore_index=True)

    for k in sim_npz.files:
        all_df[f"_sim_{k}"] = sim_npz[k]

    unsup_names = ["max_sim_global", "top5_mean_global", "dup_cnt_050", "dup_cnt_070", "max_sim_same_prod", "mean_sim_same_prod"]
    for idx, u_name in enumerate(unsup_names):
        all_df[f"_u_{u_name}"] = ks_npz["unsup_sim"][:, idx]

    for idx in range(8):
        all_df[f"_svd_{idx}"] = ks_npz["svd_feats"][:, idx]

    all_xctx = np.vstack([xctx_oof, xctx_dev, xctx_test])
    all_pctx = np.vstack([pctx_oof, pctx_dev, pctx_test])
    all_xcom = np.vstack([xcom_oof, xcom_dev, xcom_test])
    all_deb = np.vstack([deb_oof, deb_dev, deb_test])

    for c in range(4):
        all_df[f"_xctx_{c}"] = all_xctx[:, c]
        all_df[f"_pctx_{c}"] = all_pctx[:, c]
        all_df[f"_xcom_{c}"] = all_xcom[:, c]
        all_df[f"_deb_{c}"] = all_deb[:, c]
    all_df["_char_len"] = all_df["comment"].fillna("").astype(str).str.len()

    stat_cols = (
        [f"_xctx_{c}" for c in range(4)]
        + [f"_pctx_{c}" for c in range(4)]
        + [f"_xcom_{c}" for c in range(4)]
        + [f"_deb_{c}" for c in range(4)]
        + ["_char_len", "rating"]
    )
    prod_loo_df = add_loo_group_stats(all_df, "product_name", "prod", stat_cols)
    cat_loo_df = add_loo_group_stats(all_df, "categories", "cat_path", stat_cols)
    all_df = pd.concat([all_df, prod_loo_df, cat_loo_df], axis=1)

    all_df["_norm_c"] = all_df["comment"].fillna("").apply(lambda s: " ".join(norm_nfkc(s).lower().split()))
    c_freq = all_df["_norm_c"].value_counts()
    all_df["comment_global_freq"] = all_df["_norm_c"].map(c_freq).astype(float)

    le = LabelEncoder()
    all_df["category_code"] = le.fit_transform(all_df["category"].astype(str))

    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    te_tr_5, (te_de_5, te_te_5) = compute_oof_target_encoding(
        train, labels, [dev, test], skf5
    )

    def build_features(sub_df, xctx_p, pctx_p, xcom_p, deb_p, rob_p, tfidf_p, te_dict, knn_mat):
        cols = {}
        models = [
            ("xctx", xctx_p),
            ("pctx", pctx_p),
            ("xcom", xcom_p),
            ("deb", deb_p),
            ("rob", rob_p),
            ("tfidf", tfidf_p),
        ]
        for name, p_mat in models:
            for c in range(4):
                p = np.clip(p_mat[:, c], 1e-6, 1.0 - 1e-6)
                cols[f"{name}_p_{c}"] = p
                if name in ("xctx", "pctx", "xcom", "deb"):
                    cols[f"{name}_logit_{c}"] = np.log(p / (1.0 - p))
            sp = np.sort(p_mat, axis=1)
            cols[f"{name}_max_prob"] = sp[:, -1]
            cols[f"{name}_margin"] = sp[:, -1] - sp[:, -2]
            cols[f"{name}_entropy"] = -np.sum(p_mat * np.log(np.clip(p_mat, 1e-6, 1.0)), axis=1)

        ens_p = 0.34 * xctx_p + 0.25 * pctx_p + 0.24 * xcom_p + 0.09 * deb_p + 0.03 * rob_p + 0.05 * tfidf_p
        ctx_ens = 0.58 * xctx_p + 0.42 * pctx_p
        com_ens = 0.78 * xcom_p + 0.22 * deb_p

        for c in range(4):
            ep = np.clip(ens_p[:, c], 1e-6, 1.0 - 1e-6)
            cols[f"ens_p_{c}"] = ep
            cols[f"ens_logit_{c}"] = np.log(ep / (1.0 - ep))
            cols[f"ctx_ens_p_{c}"] = ctx_ens[:, c]
            cols[f"com_ens_p_{c}"] = com_ens[:, c]
            cols[f"causal_ctx_xlm_{c}"] = xctx_p[:, c] - xcom_p[:, c]
            cols[f"ctx_vs_com_shift_{c}"] = ctx_ens[:, c] - com_ens[:, c]
            cols[f"diff_xctx_pctx_{c}"] = xctx_p[:, c] - pctx_p[:, c]
            cols[f"diff_ens_tfidf_{c}"] = ens_p[:, c] - tfidf_p[:, c]

        for prefix, mp in [("ens", ens_p), ("xctx", xctx_p), ("pctx", pctx_p)]:
            cols[f"{prefix}_diff_p0_p2"] = mp[:, 0] - mp[:, 2]
            cols[f"{prefix}_diff_p0_p3"] = mp[:, 0] - mp[:, 3]
            cols[f"{prefix}_diff_p1_p3"] = mp[:, 1] - mp[:, 3]
            cols[f"{prefix}_diff_p1_p0"] = mp[:, 1] - mp[:, 0]
            cols[f"{prefix}_diff_p2_p3"] = mp[:, 2] - mp[:, 3]

        preds_stack = np.column_stack([m.argmax(axis=1) for _, m in models])
        for c in range(4):
            cols[f"vote_cnt_{c}"] = (preds_stack == c).sum(axis=1).astype(float)

        group_cols = [c for c in sub_df.columns if c.startswith(("prod_loo_", "cat_path_loo_", "prod_count", "cat_path_count"))]
        for col in group_cols:
            cols[col] = sub_df[col].to_numpy()
        for k, v in te_dict.items():
            cols[k] = v

        for c in range(4):
            cols[f"xctx_minus_prod_loo_{c}"] = xctx_p[:, c] - cols[f"prod_loo__xctx_{c}"]
            cols[f"pctx_minus_prod_loo_{c}"] = pctx_p[:, c] - cols[f"prod_loo__pctx_{c}"]
            cols[f"xcom_minus_prod_loo_{c}"] = xcom_p[:, c] - cols[f"prod_loo__xcom_{c}"]
            cols[f"xctx_minus_cat_loo_{c}"] = xctx_p[:, c] - cols[f"cat_path_loo__xctx_{c}"]

        for k in ["cos_c_p", "cos_c_d", "cos_c_cat", "ov5_ratio", "win_ov"]:
            cols[k] = sub_df[f"_sim_{k}"].to_numpy()

        for u_name in unsup_names:
            cols[u_name] = sub_df[f"_u_{u_name}"].to_numpy()

        for c in range(4):
            cols[f"knn_max_sim_{c}"] = knn_mat[:, c]
            cols[f"knn_prob_{c}"] = knn_mat[:, 4 + c]

        for idx in range(8):
            cols[f"svd_{idx}"] = sub_df[f"_svd_{idx}"].to_numpy()

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
        cols["ratio_rev_sold"] = (num_rev + 1.0) / (num_sold + 1.0)
        cols["category_code"] = sub_df["category_code"].to_numpy()
        cols["comment_global_freq"] = sub_df["comment_global_freq"].to_numpy()

        raw_comments = sub_df["comment"].fillna("").astype(str).tolist()
        nfkc_comments = [norm_nfkc(s) for s in raw_comments]
        nfkc_descs = [norm_nfkc(s) for s in sub_df["description"].fillna("").astype(str)]
        nfkc_prods = [norm_nfkc(s) for s in sub_df["product_name"].fillna("").astype(str)]

        cols["has_nfd_comment"] = np.array([float(s != norm_nfc(s)) for s in raw_comments])
        cols["newline_cnt"] = np.array([s.replace("\r\n", "\n").count("\n") for s in raw_comments], dtype=float)
        cols["has_spec_header"] = np.array([float(bool(SPEC_HEADER_PAT.search(s.lower()))) for s in nfkc_comments])
        cols["is_evoucher"] = np.array([float(bool(EVOUCHER_PAT.search(p.lower()))) for p in nfkc_prods])

        char_len = np.array([len(s) for s in nfkc_comments], dtype=float)
        cols["char_len"] = char_len
        cols["char_len_ratio_prod"] = char_len / (cols["prod_loo__char_len"] + 1.0)
        cols["word_count"] = np.array([len(s.split()) for s in nfkc_comments], dtype=float)
        cols["char_entropy"] = np.array([char_entropy(s) for s in nfkc_comments], dtype=float)
        cols["max_char_repeat"] = np.array([max_repeat(s) for s in nfkc_comments], dtype=float)
        cols["exclamation_count"] = np.array([s.count("!") for s in nfkc_comments], dtype=float)
        cols["question_count"] = np.array([s.count("?") for s in nfkc_comments], dtype=float)
        cols["uppercase_ratio"] = np.array([sum(1 for ch in s if ch.isupper()) / (len(s) + 1.0) for s in nfkc_comments], dtype=float)
        cols["digit_count"] = np.array([sum(1 for ch in s if ch.isdigit()) for s in nfkc_comments], dtype=float)

        def has_any(t_low, pats):
            return float(any(bool(re.search(p, t_low)) for p in pats))

        cols["has_c1_pat"] = np.array([has_any(c.lower(), C1_PATTERNS) for c in nfkc_comments])
        cols["has_c2_pat"] = np.array([has_any(c.lower(), C2_PATTERNS) for c in nfkc_comments])
        cols["has_c3_pat"] = np.array([has_any(c.lower(), C3_PATTERNS) for c in nfkc_comments])

        def jaccard(s1, s2):
            w1 = set(s1.lower().split())
            w2 = set(s2.lower().split())
            return len(w1 & w2) / len(w1 | w2) if (w1 and w2) else 0.0

        def exact_slice_nfkc(c_s, d_s):
            c_clean = " ".join(c_s.lower().split())[:260]
            d_clean = " ".join(d_s.lower().split())
            if len(c_clean) < 28 or not d_clean:
                return 0.0
            for i in range(0, max(1, len(c_clean) - 27), 8):
                if c_clean[i : i + 28] in d_clean:
                    return 1.0
            return 0.0

        cols["desc_jaccard"] = np.array([jaccard(c, d) for c, d in zip(nfkc_comments, nfkc_descs)])
        cols["desc_exact_slice"] = np.array([exact_slice_nfkc(c, d) for c, d in zip(nfkc_comments, nfkc_descs)])
        cols["prod_jaccard"] = np.array([jaccard(c, p) for c, p in zip(nfkc_comments, nfkc_prods)])

        # Contextual copy-paste interaction signals (protects unique product description copy-pastes when knn_prob_1 == 0)
        cols["desc_copy_signal"] = np.maximum(cols["ov5_ratio"], cols["win_ov"]) * (0.5 + 0.5 * cols["desc_exact_slice"]) + 0.35 * cols["has_spec_header"]
        cols["ctx_copy_boost"] = np.maximum(xctx_p[:, 1], pctx_p[:, 1]) * (cols["ov5_ratio"] + cols["win_ov"] + 0.5 * cols["has_spec_header"])

        return pd.DataFrame(cols, index=sub_df.index)

    print("2. Building 233-feature Dual-Context + KNN/SVD + Copy-Paste matrices...")
    X_tr_a = build_features(
        all_df.iloc[:n_tr], xctx_oof, pctx_oof, xcom_oof, deb_oof, rob_oof, tfidf_oof, te_tr_5, ks_npz["knn_a_oof"]
    )
    X_de_a = build_features(
        all_df.iloc[n_tr : n_tr + n_de], xctx_dev, pctx_dev, xcom_dev, deb_dev, rob_dev, tfidf_dev, te_de_5, ks_npz["knn_a_dev"]
    )
    X_te_a = build_features(
        all_df.iloc[n_tr + n_de :], xctx_test, pctx_test, xcom_test, deb_test, rob_test, tfidf_test, te_te_5, ks_npz["knn_a_test"]
    )
    print(f"   Total feature count: {X_tr_a.shape[1]}")

    balanced_weights = np.array([1.0, 3.6, 1.8, 1.3])

    lgb_d4_params = {
        "objective": "multiclass", "num_class": 4, "metric": "multi_logloss",
        "learning_rate": 0.03, "num_leaves": 19, "max_depth": 4, "min_child_samples": 30,
        "feature_fraction": 0.55, "bagging_fraction": 0.8, "bagging_freq": 1,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "verbose": -1,
    }
    lgb_d5_params = {
        "objective": "multiclass", "num_class": 4, "metric": "multi_logloss",
        "learning_rate": 0.026, "num_leaves": 26, "max_depth": 5, "min_child_samples": 32,
        "feature_fraction": 0.50, "bagging_fraction": 0.8, "bagging_freq": 1,
        "reg_alpha": 0.8, "reg_lambda": 3.0, "verbose": -1,
    }
    xgb_d4_params = {
        "objective": "multi:softprob", "num_class": 4, "eval_metric": "mlogloss",
        "learning_rate": 0.035, "max_depth": 4, "min_child_weight": 4.0,
        "subsample": 0.8, "colsample_bytree": 0.55,
        "reg_alpha": 0.5, "reg_lambda": 2.5, "tree_method": "hist",
    }
    xgb_d5_params = {
        "objective": "multi:softprob", "num_class": 4, "eval_metric": "mlogloss",
        "learning_rate": 0.028, "max_depth": 5, "min_child_weight": 5.0,
        "subsample": 0.78, "colsample_bytree": 0.50,
        "reg_alpha": 0.8, "reg_lambda": 3.5, "tree_method": "hist",
    }

    def run_tree_cv(X_train_all, y_all, X_eval_list, skf, seed_offset=0, cat_es=55, multi_bag_gbdt=False):
        n_samples = len(X_train_all)
        n_splits = skf.get_n_splits()
        w_all = balanced_weights[y_all]

        keys = ("cat6", "cat5", "lgb", "xgb")
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
                iterations=1050, learning_rate=0.038, depth=6, l2_leaf_reg=5.5,
                loss_function="MultiClass", random_seed=seed_f, verbose=False
            )
            mc6.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=cat_es, verbose=False)
            oof_mc["cat6"][v_idx] = mc6.predict_proba(X_vf)
            for idx, Xe in enumerate(X_eval_list):
                eval_mc[idx]["cat6"] += mc6.predict_proba(Xe) / n_splits

            # 2. CatBoost Depth-5 High-Regularization
            mc5 = CatBoostClassifier(
                iterations=1100, learning_rate=0.036, depth=5, l2_leaf_reg=8.0, random_strength=1.5,
                loss_function="MultiClass", random_seed=seed_f + 77, verbose=False
            )
            mc5.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=cat_es, verbose=False)
            oof_mc["cat5"][v_idx] = mc5.predict_proba(X_vf)
            for idx, Xe in enumerate(X_eval_list):
                eval_mc[idx]["cat5"] += mc5.predict_proba(Xe) / n_splits

            # 3. LightGBM (Depth-4 + optional Depth-5 high-reg multi-bag)
            dtr = lgb.Dataset(X_tf, label=y_tf, weight=w_tf)
            dva = lgb.Dataset(X_vf, label=y_vf, weight=w_vf, reference=dtr)
            ml1 = lgb.train(dict(lgb_d4_params, random_state=seed_f), dtr, num_boost_round=750,
                            valid_sets=[dva], callbacks=[lgb.early_stopping(45, verbose=False)])
            if multi_bag_gbdt:
                ml2 = lgb.train(dict(lgb_d5_params, random_state=seed_f + 131), dtr, num_boost_round=800,
                                valid_sets=[dva], callbacks=[lgb.early_stopping(45, verbose=False)])
                oof_mc["lgb"][v_idx] = 0.55 * ml1.predict(X_vf) + 0.45 * ml2.predict(X_vf)
                for idx, Xe in enumerate(X_eval_list):
                    eval_mc[idx]["lgb"] += (0.55 * ml1.predict(Xe) + 0.45 * ml2.predict(Xe)) / n_splits
            else:
                oof_mc["lgb"][v_idx] = ml1.predict(X_vf)
                for idx, Xe in enumerate(X_eval_list):
                    eval_mc[idx]["lgb"] += ml1.predict(Xe) / n_splits

            # 4. XGBoost (Depth-4 + optional Depth-5 high-reg multi-bag)
            dxtr = xgb.DMatrix(X_tf, label=y_tf, weight=w_tf)
            dxva = xgb.DMatrix(X_vf, label=y_vf, weight=w_vf)
            mx1 = xgb.train(dict(xgb_d4_params, random_state=seed_f), dxtr, num_boost_round=750,
                            evals=[(dxva, "val")], early_stopping_rounds=45, verbose_eval=False)
            if multi_bag_gbdt:
                mx2 = xgb.train(dict(xgb_d5_params, random_state=seed_f + 131), dxtr, num_boost_round=800,
                                evals=[(dxva, "val")], early_stopping_rounds=45, verbose_eval=False)
                oof_mc["xgb"][v_idx] = 0.55 * mx1.predict(dxva) + 0.45 * mx2.predict(dxva)
                for idx, Xe in enumerate(X_eval_list):
                    dxe = xgb.DMatrix(Xe)
                    eval_mc[idx]["xgb"] += (0.55 * mx1.predict(dxe) + 0.45 * mx2.predict(dxe)) / n_splits
            else:
                oof_mc["xgb"][v_idx] = mx1.predict(dxva)
                for idx, Xe in enumerate(X_eval_list):
                    eval_mc[idx]["xgb"] += mx1.predict(xgb.DMatrix(Xe)) / n_splits

        return oof_mc, eval_mc

    print("3. Training Stage A (5-Fold CatBoost-d6/d5 + Reg-LGBM + Reg-XGB on Train)...")
    oof_mc_a, (dev_mc_a, test_mc_a) = run_tree_cv(
        X_tr_a, labels, [X_de_a, X_te_a], skf5, seed_offset=0, cat_es=55, multi_bag_gbdt=False
    )
    for k in ("cat6", "cat5", "lgb", "xgb"):
        print(f"   [Stage A {k.upper():4s}] Raw OOF F1: {fast_macro_f1(labels, oof_mc_a[k].argmax(1)):.5f} | DEV OOF F1: {fast_macro_f1(dev_labels, dev_mc_a[k].argmax(1)):.5f}")

    print("4. Training Stage B (10-Fold 60-Model Bagged Ensemble on Full Train+Dev = 15,896 rows)...")
    train_dev_df = pd.concat([train, dev], ignore_index=True)
    y_full = np.concatenate([labels, dev_labels])
    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=102)
    te_full_10, (te_te_10,) = compute_oof_target_encoding(
        train_dev_df, y_full, [test], skf10
    )

    X_full_b = build_features(
        all_df.iloc[: n_tr + n_de],
        np.vstack([xctx_oof, xctx_dev]),
        np.vstack([pctx_oof, pctx_dev]),
        np.vstack([xcom_oof, xcom_dev]),
        np.vstack([deb_oof, deb_dev]),
        np.vstack([rob_oof, rob_dev]),
        np.vstack([tfidf_oof, tfidf_dev]),
        te_full_10,
        ks_npz["knn_b_full"],
    )
    X_te_b = build_features(
        all_df.iloc[n_tr + n_de :],
        xctx_test, pctx_test, xcom_test, deb_test, rob_test, tfidf_test,
        te_te_10, ks_npz["knn_b_test"],
    )

    oof_mc_b, (test_mc_b,) = run_tree_cv(
        X_full_b, y_full, [X_te_b], skf10, seed_offset=300, cat_es=85, multi_bag_gbdt=True
    )
    for k in ("cat6", "cat5", "lgb", "xgb"):
        print(f"   [Stage B {k.upper():4s}] Raw Train-Slice F1: {fast_macro_f1(labels, oof_mc_b[k][:n_tr].argmax(1)):.5f} | Full 15,896 OOF F1: {fast_macro_f1(y_full, oof_mc_b[k].argmax(1)):.5f}")

    # Level-3 Multi-Architecture Championship Blend:
    #   - 34% Stage B Bagged XGBoost (d4+d5, 10-Fold on 15,896 rows — lowest log-loss & >0.806 raw F1)
    #   - 22% Stage B Bagged LightGBM (d4+d5, 10-Fold on 15,896 rows)
    #   - 24% Symmetric CatBoost Ensemble (12% A_cat6 + 8% B_cat6 + 2% A_cat5 + 2% B_cat5)
    #   - 12% Submission 14 Grandmaster Probabilities (0.80036 LB verified anchor)
    #   -  8% Pure Dual-Context Neural Consensus (48% XLM-Ctx + 30% PhoBERT-Ctx + 16% XLM-TTA + 6% DeBERTa)
    s14_oof = np.load(OUTPUT / "submission_14_oof_probs.npy")
    s14_dev = np.load(OUTPUT / "submission_14_dev_probs.npy")
    s14_test = np.load(OUTPUT / "submission_14_test_probs.npy")

    neural_oof = 0.48 * xctx_oof + 0.30 * pctx_oof + 0.16 * xcom_oof + 0.06 * deb_oof
    neural_dev = 0.48 * xctx_dev + 0.30 * pctx_dev + 0.16 * xcom_dev + 0.06 * deb_dev
    neural_test = 0.48 * xctx_test + 0.30 * pctx_test + 0.16 * xcom_test + 0.06 * deb_test

    cat_full = (
        0.50 * np.vstack([oof_mc_a["cat6"], dev_mc_a["cat6"]])
        + 0.34 * oof_mc_b["cat6"]
        + 0.08 * np.vstack([oof_mc_a["cat5"], dev_mc_a["cat5"]])
        + 0.08 * oof_mc_b["cat5"]
    )
    cat_test = (
        0.42 * test_mc_a["cat6"]
        + 0.42 * test_mc_b["cat6"]
        + 0.08 * test_mc_a["cat5"]
        + 0.08 * test_mc_b["cat5"]
    )

    full_comb_oof = (
        0.34 * oof_mc_b["xgb"]
        + 0.22 * oof_mc_b["lgb"]
        + 0.24 * cat_full
        + 0.12 * np.vstack([s14_oof, s14_dev])
        + 0.08 * np.vstack([neural_oof, neural_dev])
    )
    final_test = (
        0.34 * test_mc_b["xgb"]
        + 0.22 * test_mc_b["lgb"]
        + 0.24 * cat_test
        + 0.12 * s14_test
        + 0.08 * neural_test
    )

    final_oof = full_comb_oof[:n_tr]
    final_dev = full_comb_oof[n_tr:]

    raw_oof_f1 = fast_macro_f1(labels, final_oof.argmax(axis=1))
    raw_full_f1 = fast_macro_f1(y_full, full_comb_oof.argmax(axis=1))

    # Strict 5-Fold Nested Out-Of-Fold Calibration on Full 15,896 rows (low-DOF regularized around [1,1,1,1])
    skf_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    nested_preds = np.zeros(len(y_full), dtype=int)
    fold_ws = []
    for tr_i, va_i in skf_cal.split(full_comb_oof, y_full):
        best_s = -1.0
        bw = np.ones(4)
        p_sub = full_comb_oof[tr_i]
        y_sub = y_full[tr_i]
        for w1 in np.linspace(0.98, 1.15, 18):
            for w2 in np.linspace(0.96, 1.02, 7):
                for w3 in np.linspace(0.98, 1.02, 5):
                    w = np.array([1.0, w1, w2, w3])
                    s = fast_macro_f1(y_sub, (p_sub * w).argmax(1)) - 0.0015 * float(np.sum((w - 1.0) ** 2))
                    if s > best_s:
                        best_s = s
                        bw = w.copy()
        nested_preds[va_i] = (full_comb_oof[va_i] * bw).argmax(1)
        fold_ws.append(bw)

    best_w = np.mean(fold_ws, axis=0)
    nested_tr_f1 = fast_macro_f1(labels, nested_preds[:n_tr])
    nested_full_f1 = fast_macro_f1(y_full, nested_preds)

    oof_preds = (final_oof * best_w).argmax(axis=1)
    dev_preds = (final_dev * best_w).argmax(axis=1)
    test_preds = (final_test * best_w).argmax(axis=1)

    oof_macro = float(f1_score(labels, oof_preds, average="macro"))
    dev_macro = float(f1_score(dev_labels, dev_preds, average="macro"))
    full_macro = float(f1_score(y_full, (full_comb_oof * best_w).argmax(axis=1), average="macro"))

    print("\n" + "=" * 76)
    print(f"Nested-CV Averaged Calibration Weights:  {np.round(best_w, 4).tolist()}")
    print(f"Raw Uncalibrated Train OOF Macro F1:     {raw_oof_f1:.5f}")
    print(f"Raw Uncalibrated Full  OOF Macro F1:     {raw_full_f1:.5f}")
    print(f"Strict Nested-CV Train OOF Macro F1:     {nested_tr_f1:.5f}")
    print(f"Strict Nested-CV Full  OOF Macro F1:     {nested_full_f1:.5f}")
    print(f"Final Train OOF Macro F1 (14,306 rows):  {oof_macro:.5f}")
    print(f"Final Full  OOF Macro F1 (15,896 rows):  {full_macro:.5f}")
    print(f"Final Dev   OOF Macro F1  (1,590 rows):  {dev_macro:.5f}")
    print("=" * 76)

    print("\nDetailed Classification Report on Train OOF (14,306 rows):")
    print(classification_report(labels, oof_preds, digits=4))
    print("\nDetailed Classification Report on Full Train+Dev OOF (15,896 rows):")
    print(classification_report(y_full, (full_comb_oof * best_w).argmax(axis=1), digits=4))

    np.save(OUTPUT / "submission_15_oof_probs.npy", final_oof)
    np.save(OUTPUT / "submission_15_dev_probs.npy", final_dev)
    np.save(OUTPUT / "submission_15_test_probs.npy", final_test)
    np.savez_compressed(
        OUTPUT / "submission_15_tree_components.npz",
        **{f"a_oof_{k}": oof_mc_a[k] for k in oof_mc_a},
        **{f"a_dev_{k}": dev_mc_a[k] for k in dev_mc_a},
        **{f"a_test_{k}": test_mc_a[k] for k in test_mc_a},
        **{f"b_oof_{k}": oof_mc_b[k] for k in oof_mc_b},
        **{f"b_test_{k}": test_mc_b[k] for k in test_mc_b},
    )

    submission_15 = make_submission(test, sample, test_preds)
    sub15_output = OUTPUT / "submission_15.csv"
    sub15_root = ROOT / "submission_15.csv"
    submission_15.to_csv(sub15_output, index=False)
    submission_15.to_csv(sub15_root, index=False)
    print(f"Saved submission to:\n  - {sub15_output}\n  - {sub15_root}")

    test_dist = pd.Series(test_preds).value_counts().to_dict()
    print(f"Test Class Distribution: {test_dist}")

    metrics = {
        "model": "Championship 80-Tree + Dual-Context (XLM-Ctx + PhoBERT-Ctx + XLM-TTA + DeBERTa + RoBERTa + TF-IDF) + KNN/SVD + Copy-Paste Signals",
        "features_count": int(X_tr_a.shape[1]),
        "weights": best_w.tolist(),
        "raw_oof_macro_f1": raw_oof_f1,
        "raw_full_oof_macro_f1": raw_full_f1,
        "strict_nested_cv_oof_macro_f1": nested_tr_f1,
        "strict_nested_cv_full_macro_f1": nested_full_f1,
        "oof_macro_f1": oof_macro,
        "full_15896_oof_macro_f1": full_macro,
        "dev_macro_f1": dev_macro,
        "test_distribution": test_dist,
    }
    (OUTPUT / "submission_15_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
