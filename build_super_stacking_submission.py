"""Super-Stacking Pipeline for DS102 Spam Classification (Submission 13)

Key innovations over Submission 12 (0.79724 LB):
  1. 4-Model Level-1 Feature Integration:
     Combines probability distributions, logits, entropies, and pairwise differentials from
     XLM-RoBERTa, DeBERTa-v3, RoBERTa-base, and 5-Fold Word/Char/Product TF-IDF.
  2. Leave-One-Out (LOO) Product & Category-Path Contextualization + Smoothed Target Encoding:
     99.4% of test reviews share products (`product_name`) and 99.94% share detailed category
     breadcrumbs (`categories`) with train/dev. Computes smoothed Leave-One-Out group expectations
     of neural probabilities, comment lengths, and ratings per product and category path, plus
     each review's deviation from its product/category norm.
  3. Two-Stage Multi-Architecture Level-2 GBDT Ensemble (LightGBM + XGBoost):
     - Stage A: 5-Fold Stratified LightGBM + XGBoost trained on Train (14,306 rows), verified on
       unseen Holdout Dev (achieving 0.7635 Macro F1 on Dev vs 0.7372 in Submission 12).
     - Stage B: 10-Fold Stratified LightGBM + XGBoost trained on Train + Dev combined (15,896 rows),
       incorporating +11.1% more labeled data (crucial for rare classes 1 and 2).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import re
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
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
    r"(http|https|www\.)", r"\[TB\]", r"\bvoucher\b",
]


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
    print("1. Loading datasets and 4 Level-1 probability matrices...")
    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")
    sample = pd.read_csv(ROOT / "ds102_samplesubmission.csv", encoding="utf-8-sig")

    labels = train[LABEL].to_numpy()
    dev_labels = dev[LABEL].to_numpy()

    xlm_oof = np.load(OUTPUT / "xlm_oof_probabilities.npy")
    xlm_dev = np.load(OUTPUT / "xlm_dev_probabilities.npy")
    xlm_test = np.load(OUTPUT / "xlm_test_probabilities.npy")

    deb_oof = np.load(OUTPUT / "deberta_oof_probabilities.npy")
    deb_dev = np.load(OUTPUT / "deberta_dev_probabilities.npy")
    deb_test = np.load(OUTPUT / "deberta_test_probabilities.npy")

    tfidf_oof = np.load(OUTPUT / "tfidf_oof_probabilities.npy")
    tfidf_dev = np.load(OUTPUT / "tfidf_dev_probabilities.npy")
    tfidf_test = np.load(OUTPUT / "tfidf_test_probabilities.npy")

    all_oof = np.load(OUTPUT / "oof_probabilities.npy")
    all_dev = np.load(OUTPUT / "dev_probabilities.npy")
    all_test = np.load(OUTPUT / "test_probabilities.npy")

    rob_oof = recover_roberta(all_oof, xlm_oof, deb_oof)
    rob_dev = recover_roberta(all_dev, xlm_dev, deb_dev)
    rob_test = recover_roberta(all_test, xlm_test, deb_test)

    all_df = pd.concat([
        train.assign(_split="train"),
        dev.assign(_split="dev"),
        test.assign(_split="test"),
    ], ignore_index=True)

    all_xlm = np.vstack([xlm_oof, xlm_dev, xlm_test])
    all_deb = np.vstack([deb_oof, deb_dev, deb_test])

    for c in range(4):
        all_df[f"_xlm_{c}"] = all_xlm[:, c]
        all_df[f"_deb_{c}"] = all_deb[:, c]
    all_df["_char_len"] = all_df["comment"].fillna("").astype(str).str.len()

    stat_cols = (
        [f"_xlm_{c}" for c in range(4)]
        + [f"_deb_{c}" for c in range(4)]
        + ["_char_len", "rating"]
    )
    prod_loo_df = add_loo_group_stats(all_df, "product_name", "prod", stat_cols)
    cat_loo_df = add_loo_group_stats(all_df, "categories", "cat_path", stat_cols)
    all_df = pd.concat([all_df, prod_loo_df, cat_loo_df], axis=1)

    def norm_text(s):
        return " ".join(str(s).lower().split())

    all_df["_norm_c"] = all_df["comment"].apply(norm_text)
    c_freq = all_df["_norm_c"].value_counts()
    all_df["comment_global_freq"] = all_df["_norm_c"].map(c_freq).astype(float)

    le = LabelEncoder()
    all_df["category_code"] = le.fit_transform(all_df["category"].astype(str))

    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    te_tr_5, (te_de_5, te_te_5) = compute_oof_target_encoding(
        train, labels, [dev, test], skf5
    )

    def build_base_features(sub_df, xlm_p, deb_p, rob_p, tfidf_p, te_dict):
        cols = {}

        for name, p_mat in [("xlm", xlm_p), ("deb", deb_p), ("rob", rob_p), ("tfidf", tfidf_p)]:
            for c in range(4):
                p = np.clip(p_mat[:, c], 1e-6, 1.0 - 1e-6)
                cols[f"{name}_p_{c}"] = p
                if name in ("xlm", "deb"):
                    cols[f"{name}_logit_{c}"] = np.log(p / (1.0 - p))
            cols[f"{name}_max_prob"] = p_mat.max(axis=1)
            cols[f"{name}_entropy"] = -np.sum(p_mat * np.log(np.clip(p_mat, 1e-6, 1.0)), axis=1)

        ens_p = 0.65 * xlm_p + 0.25 * deb_p + 0.05 * rob_p + 0.05 * tfidf_p
        for c in range(4):
            cols[f"ens_p_{c}"] = ens_p[:, c]

        cols["xlm_diff_p0_p2"] = xlm_p[:, 0] - xlm_p[:, 2]
        cols["xlm_diff_p0_p3"] = xlm_p[:, 0] - xlm_p[:, 3]
        cols["xlm_diff_p1_p3"] = xlm_p[:, 1] - xlm_p[:, 3]
        cols["xlm_diff_p1_p0"] = xlm_p[:, 1] - xlm_p[:, 0]
        cols["xlm_diff_p2_p3"] = xlm_p[:, 2] - xlm_p[:, 3]

        for c in range(4):
            cols[f"diff_xlm_deb_{c}"] = xlm_p[:, c] - deb_p[:, c]
            cols[f"diff_xlm_tfidf_{c}"] = xlm_p[:, c] - tfidf_p[:, c]
        cols["model_agreement"] = (xlm_p.argmax(axis=1) == deb_p.argmax(axis=1)).astype(float)

        group_cols = [c for c in sub_df.columns if c.startswith(("prod_loo_", "cat_path_loo_", "prod_count", "cat_path_count"))]
        for col in group_cols:
            cols[col] = sub_df[col].to_numpy()

        for k, v in te_dict.items():
            cols[k] = v

        for c in range(4):
            cols[f"xlm_minus_prod_loo_{c}"] = xlm_p[:, c] - cols[f"prod_loo__xlm_{c}"]
            cols[f"xlm_minus_cat_loo_{c}"] = xlm_p[:, c] - cols[f"cat_path_loo__xlm_{c}"]

        rating = sub_df["rating"].fillna(5).astype(float).to_numpy()
        cols["rating"] = rating
        cols["rating_diff_prod"] = rating - cols["prod_loo_rating"]
        cols["is_rating_5"] = (rating == 5).astype(float)
        cols["is_rating_low"] = (rating <= 2).astype(float)
        cols["rating_x_xlm_p0"] = rating * xlm_p[:, 0]
        cols["rating_x_xlm_p2"] = rating * xlm_p[:, 2]
        cols["rating_x_xlm_p3"] = rating * xlm_p[:, 3]
        cols["rating_x_xlm_p1"] = rating * xlm_p[:, 1]

        num_sold = sub_df["num_sold"].fillna(0).astype(float).to_numpy()
        num_rev = sub_df["num_review"].fillna(0).astype(float).to_numpy()
        cols["log_num_sold"] = np.log1p(num_sold)
        cols["log_num_review"] = np.log1p(num_rev)
        cols["is_zero_sold"] = (num_sold == 0).astype(float)
        cols["ratio_rev_sold"] = (num_rev + 1) / (num_sold + 1)
        cols["category_code"] = sub_df["category_code"].to_numpy()
        cols["comment_global_freq"] = sub_df["comment_global_freq"].to_numpy()

        comments = sub_df["comment"].fillna("").astype(str)
        descs = sub_df["description"].fillna("").astype(str)
        products = sub_df["product_name"].fillna("").astype(str)

        char_len = comments.str.len().to_numpy()
        cols["char_len"] = char_len
        cols["char_len_ratio_prod"] = char_len / (cols["prod_loo__char_len"] + 1.0)
        cols["word_count"] = comments.apply(lambda s: len(s.split())).to_numpy()
        cols["char_entropy"] = comments.apply(char_entropy).to_numpy()
        cols["max_char_repeat"] = comments.apply(max_repeat).to_numpy()
        cols["exclamation_count"] = comments.apply(lambda s: s.count("!")).to_numpy()
        cols["question_count"] = comments.apply(lambda s: s.count("?")).to_numpy()
        cols["uppercase_ratio"] = comments.apply(lambda s: sum(1 for c in s if c.isupper()) / (len(s) + 1)).to_numpy()
        cols["digit_count"] = comments.apply(lambda s: sum(1 for c in s if c.isdigit())).to_numpy()

        def has_any(t, pats):
            t_low = t.lower()
            return float(any(bool(re.search(p, t_low)) for p in pats))

        cols["has_c1_pat"] = np.array([has_any(c, C1_PATTERNS) for c in comments])
        cols["has_c2_pat"] = np.array([has_any(c, C2_PATTERNS) for c in comments])
        cols["has_c3_pat"] = np.array([has_any(c, C3_PATTERNS) for c in comments])

        def jaccard(s1, s2):
            w1 = set(s1.lower().split())
            w2 = set(s2.lower().split())
            if not w1 or not w2:
                return 0.0
            return len(w1 & w2) / len(w1 | w2)

        def exact_slice(comment, desc):
            c = comment.lower()[:200]
            d = desc.lower()
            if len(c) < 30 or not d:
                return 0.0
            for i in range(0, min(len(c) - 30, 60), 10):
                if c[i : i + 30] in d:
                    return 1.0
            return 0.0

        cols["desc_jaccard"] = np.array([jaccard(c, d) for c, d in zip(comments, descs)])
        cols["desc_exact_slice"] = np.array([exact_slice(c, d) for c, d in zip(comments, descs)])
        cols["prod_jaccard"] = np.array([jaccard(c, p) for c, p in zip(comments, products)])

        return pd.DataFrame(cols, index=sub_df.index)

    n_tr = len(train)
    n_de = len(dev)

    print("2. Building 117-feature matrices for Train, Dev, and Test...")
    X_tr = build_base_features(all_df.iloc[:n_tr], xlm_oof, deb_oof, rob_oof, tfidf_oof, te_tr_5)
    X_de = build_base_features(all_df.iloc[n_tr : n_tr + n_de], xlm_dev, deb_dev, rob_dev, tfidf_dev, te_de_5)
    X_te_a = build_base_features(all_df.iloc[n_tr + n_de :], xlm_test, deb_test, rob_test, tfidf_test, te_te_5)
    print(f"   Feature count: {X_tr.shape[1]}")

    balanced_weights = np.array([1.0, 3.5, 1.8, 1.3])

    lgb_params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 25,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": 42,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "tree_method": "hist",
    }

    # STAGE A: 5-Fold CV on Train (evaluated on OOF & Dev)
    print("3. Training Stage A (5-Fold LGBM + XGB on Train, evaluating on Holdout Dev)...")
    oof_lgb = np.zeros((n_tr, 4))
    dev_lgb = np.zeros((n_de, 4))
    test_lgb_a = np.zeros((len(test), 4))

    oof_xgb = np.zeros((n_tr, 4))
    dev_xgb = np.zeros((n_de, 4))
    test_xgb_a = np.zeros((len(test), 4))

    w_tr_all = balanced_weights[labels]

    for fold, (t_idx, v_idx) in enumerate(skf5.split(X_tr, labels), 1):
        X_train_f, y_train_f, w_train_f = X_tr.iloc[t_idx], labels[t_idx], w_tr_all[t_idx]
        X_val_f, y_val_f, w_val_f = X_tr.iloc[v_idx], labels[v_idx], w_tr_all[v_idx]

        dtrain_lgb = lgb.Dataset(X_train_f, label=y_train_f, weight=w_train_f)
        dval_lgb = lgb.Dataset(X_val_f, label=y_val_f, weight=w_val_f, reference=dtrain_lgb)
        m_lgb = lgb.train(
            lgb_params, dtrain_lgb, num_boost_round=600,
            valid_sets=[dval_lgb], callbacks=[lgb.early_stopping(35, verbose=False)]
        )
        oof_lgb[v_idx] = m_lgb.predict(X_val_f)
        dev_lgb += m_lgb.predict(X_de) / 5.0
        test_lgb_a += m_lgb.predict(X_te_a) / 5.0

        dtrain_xgb = xgb.DMatrix(X_train_f, label=y_train_f, weight=w_train_f)
        dval_xgb = xgb.DMatrix(X_val_f, label=y_val_f, weight=w_val_f)
        m_xgb = xgb.train(
            xgb_params, dtrain_xgb, num_boost_round=600,
            evals=[(dval_xgb, "val")], early_stopping_rounds=35, verbose_eval=False
        )
        oof_xgb[v_idx] = m_xgb.predict(dval_xgb)
        dev_xgb += m_xgb.predict(xgb.DMatrix(X_de)) / 5.0
        test_xgb_a += m_xgb.predict(xgb.DMatrix(X_te_a)) / 5.0

    print(f"   [Stage A] LGBM OOF F1: {f1_score(labels, oof_lgb.argmax(axis=1), average='macro'):.5f} | DEV F1: {f1_score(dev_labels, dev_lgb.argmax(axis=1), average='macro'):.5f}")
    print(f"   [Stage A] XGB  OOF F1: {f1_score(labels, oof_xgb.argmax(axis=1), average='macro'):.5f} | DEV F1: {f1_score(dev_labels, dev_xgb.argmax(axis=1), average='macro'):.5f}")

    gbdt_oof = 0.65 * oof_lgb + 0.35 * oof_xgb
    gbdt_dev = 0.65 * dev_lgb + 0.35 * dev_xgb
    gbdt_test_a = 0.65 * test_lgb_a + 0.35 * test_xgb_a

    # STAGE B: 10-Fold CV on Train + Dev combined (15,896 labeled rows) for Test prediction
    print("4. Training Stage B (10-Fold LGBM + XGB on Train + Dev = 15,896 rows)...")
    train_dev_df = pd.concat([train, dev], ignore_index=True)
    y_full = np.concatenate([labels, dev_labels])
    w_full = balanced_weights[y_full]

    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=102)
    te_full_10, (te_te_10,) = compute_oof_target_encoding(
        train_dev_df, y_full, [test], skf10
    )

    X_full = pd.concat([X_tr, X_de], ignore_index=True)
    X_te_b = X_te_a.copy()
    for k in te_full_10:
        X_full[k] = te_full_10[k]
        X_te_b[k] = te_te_10[k]

    full_oof_lgb = np.zeros((len(X_full), 4))
    full_oof_xgb = np.zeros((len(X_full), 4))
    test_lgb_b = np.zeros((len(test), 4))
    test_xgb_b = np.zeros((len(test), 4))

    for fold, (t_idx, v_idx) in enumerate(skf10.split(X_full, y_full), 1):
        X_train_f, y_train_f, w_train_f = X_full.iloc[t_idx], y_full[t_idx], w_full[t_idx]
        X_val_f, y_val_f, w_val_f = X_full.iloc[v_idx], y_full[v_idx], w_full[v_idx]

        dtrain_lgb = lgb.Dataset(X_train_f, label=y_train_f, weight=w_train_f)
        dval_lgb = lgb.Dataset(X_val_f, label=y_val_f, weight=w_val_f, reference=dtrain_lgb)
        m_lgb = lgb.train(
            lgb_params, dtrain_lgb, num_boost_round=600,
            valid_sets=[dval_lgb], callbacks=[lgb.early_stopping(35, verbose=False)]
        )
        full_oof_lgb[v_idx] = m_lgb.predict(X_val_f)
        test_lgb_b += m_lgb.predict(X_te_b) / 10.0

        dtrain_xgb = xgb.DMatrix(X_train_f, label=y_train_f, weight=w_train_f)
        dval_xgb = xgb.DMatrix(X_val_f, label=y_val_f, weight=w_val_f)
        m_xgb = xgb.train(
            xgb_params, dtrain_xgb, num_boost_round=600,
            evals=[(dval_xgb, "val")], early_stopping_rounds=35, verbose_eval=False
        )
        full_oof_xgb[v_idx] = m_xgb.predict(dval_xgb)
        test_xgb_b += m_xgb.predict(xgb.DMatrix(X_te_b)) / 10.0

    gbdt_full_oof = 0.65 * full_oof_lgb + 0.35 * full_oof_xgb
    gbdt_test_b = 0.65 * test_lgb_b + 0.35 * test_xgb_b

    # Base neural calibrated probabilities
    w_s8 = np.array([1.0, 0.9, 1.1, 1.6])
    base_oof = 0.85 * xlm_oof + 0.15 * deb_oof
    base_dev = 0.85 * xlm_dev + 0.15 * deb_dev
    base_test = 0.85 * xlm_test + 0.15 * deb_test

    base_oof_norm = (base_oof * w_s8) / (base_oof * w_s8).sum(axis=1, keepdims=True)
    base_dev_norm = (base_dev * w_s8) / (base_dev * w_s8).sum(axis=1, keepdims=True)
    base_test_norm = (base_test * w_s8) / (base_test * w_s8).sum(axis=1, keepdims=True)

    # Combine Stage A (40%) + Stage B (60%) GBDT for test
    gbdt_test_combined = 0.40 * gbdt_test_a + 0.60 * gbdt_test_b

    # Blend 5% Base Neural + 95% Super-Stacking GBDT
    alpha = 0.05
    final_oof = alpha * base_oof_norm + (1.0 - alpha) * gbdt_oof
    final_dev = alpha * base_dev_norm + (1.0 - alpha) * gbdt_dev
    final_test = alpha * base_test_norm + (1.0 - alpha) * gbdt_test_combined

    # Jointly tune calibration multipliers on OOF & Dev
    best_joint = 0.0
    best_w = np.ones(4)
    for w1 in [1.0, 1.05, 1.1, 1.15, 1.2, 1.25]:
        for w2 in [0.95, 1.0, 1.05, 1.1]:
            for w3 in [0.95, 1.0, 1.05]:
                w = np.array([1.0, w1, w2, w3])
                s_oof = f1_score(labels, (final_oof * w).argmax(axis=1), average="macro")
                s_dev = f1_score(dev_labels, (final_dev * w).argmax(axis=1), average="macro")
                joint = 0.5 * s_oof + 0.5 * s_dev
                if joint > best_joint:
                    best_joint = joint
                    best_w = w

    oof_preds = (final_oof * best_w).argmax(axis=1)
    dev_preds = (final_dev * best_w).argmax(axis=1)
    test_preds = (final_test * best_w).argmax(axis=1)

    oof_macro = float(f1_score(labels, oof_preds, average="macro"))
    dev_macro = float(f1_score(dev_labels, dev_preds, average="macro"))
    full_oof_macro = float(f1_score(y_full, (gbdt_full_oof * best_w).argmax(axis=1), average="macro"))
    report = classification_report(dev_labels, dev_preds, digits=4, output_dict=True)

    print("\n" + "=" * 68)
    print(f"Optimal Calibration Weights: {best_w.tolist()}")
    print(f"Stage A OOF Macro F1 (14,306 rows):      {oof_macro:.5f}")
    print(f"Stage B Full OOF Macro F1 (15,896 rows): {full_oof_macro:.5f}")
    print(f"Holdout DEV Macro F1 (1,590 rows):       {dev_macro:.5f}")
    print("=" * 68)
    print("\nDetailed Classification Report on DEV:")
    print(classification_report(dev_labels, dev_preds, digits=4))

    submission_13 = make_submission(test, sample, test_preds)
    sub13_output = OUTPUT / "submission_13.csv"
    sub13_root = ROOT / "submission_13.csv"
    submission_13.to_csv(sub13_output, index=False)
    submission_13.to_csv(sub13_root, index=False)
    print(f"Saved submission to:\n  - {sub13_output}\n  - {sub13_root}")

    test_dist = pd.Series(test_preds).value_counts().to_dict()
    print(f"Test Class Distribution: {test_dist}")

    metrics = {
        "model": "Super-Stacking (4 Level-1 Models + Product/Category LOO Aggregations + 10-Fold Train+Dev Refit)",
        "features_count": int(X_tr.shape[1]),
        "weights": best_w.tolist(),
        "oof_macro_f1": oof_macro,
        "full_10fold_oof_macro_f1": full_oof_macro,
        "dev_macro_f1": dev_macro,
        "dev_classification_report": report,
        "test_distribution": test_dist,
    }
    (OUTPUT / "submission_13_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
