"""Level-2 Stacking Model (XLM-RoBERTa + LightGBM + XGBoost + Metadata Features)

Combines:
  1. Base XLM-RoBERTa 5-fold OOF/Dev/Test probabilities & logits.
  2. Tabular metadata: rating, log(num_sold), log(num_review), review/sold ratio, category codes.
  3. Text structure signals: character entropy, max character repetition (gibberish detector),
     exclamation/question marks, uppercase ratios.
  4. Linguistic pattern detectors: Class 1 template markers, Class 2 service/untested markers,
     Class 3 spam/link/contact markers.
  5. Cross-column semantic overlap: Comment vs Product description Jaccard and exact substring match.
"""

import json
from pathlib import Path
import re
from collections import Counter

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


# Compiled regex patterns
C1_PATTERNS = [
    r"gửi gắm rất nhiều tâm tư",
    r"không thể trông đợi gì hơn",
    r"văn mẫu",
    r"tâm tư và tình cảm",
    r"người bán đã gửi gắm",
    r"dành cho ng tiêu dùng chúng tôi 1 sản phẩm",
    r"xinh đẹp tuyệt vời",
    r"tuyệt vời.*tuyệt vời",
]

C2_PATTERNS = [
    r"chưa (dùng|sử dụng|xài|biết|thử|mở|uống|ăn|đeo|mặc|cầm)",
    r"chưa rõ",
    r"chưa test",
    r"mua tặng",
    r"mua hộ",
    r"mua giùm",
    r"kiểm chứng",
    r"thời gian sau mới biết",
    r"thời gian kiểm chứng",
    r"dịch vụ",
    r"shipper",
    r"giao hàng",
    r"đóng gói",
    r"bảo hành",
]

C3_PATTERNS = [
    r"[a-zA-Z0-9]{15,}",
    r"(.)\1{4,}",
    r"(zalo|sđt|hotline|liên hệ|inbox|0[35789]\d{8})",
    r"(http|https|www\.)",
    r"\[TB\]",
    r"\bvoucher\b",
]


def max_repeat(s):
    repeats = [len(m.group()) for m in re.finditer(r'(.)\1+', str(s))]
    return max(repeats) if repeats else 1


def char_entropy(s):
    s = str(s).lower()
    if len(s) <= 1:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * np.log2(c / n) for c in counts.values())


def extract_features(df, probs, label_encoder):
    feats = pd.DataFrame()

    # Probabilities and Logits
    for c in range(4):
        p = np.clip(probs[:, c], 1e-6, 1.0 - 1e-6)
        feats[f"p_{c}"] = p
        feats[f"logit_{c}"] = np.log(p / (1.0 - p))

    feats["max_prob"] = probs.max(axis=1)
    feats["entropy_prob"] = -np.sum(probs * np.log(np.clip(probs, 1e-6, 1.0)), axis=1)
    feats["diff_p0_p2"] = probs[:, 0] - probs[:, 2]
    feats["diff_p0_p3"] = probs[:, 0] - probs[:, 3]
    feats["diff_p1_p3"] = probs[:, 1] - probs[:, 3]
    feats["diff_p1_p0"] = probs[:, 1] - probs[:, 0]
    feats["diff_p2_p3"] = probs[:, 2] - probs[:, 3]

    # Metadata
    rating = df["rating"].fillna(5).astype(float)
    feats["rating"] = rating
    feats["is_rating_5"] = (rating == 5).astype(float)
    feats["is_rating_low"] = (rating <= 2).astype(float)

    feats["rating_x_p0"] = rating * probs[:, 0]
    feats["rating_x_p2"] = rating * probs[:, 2]
    feats["rating_x_p3"] = rating * probs[:, 3]
    feats["rating_x_p1"] = rating * probs[:, 1]

    num_sold = df["num_sold"].fillna(0).astype(float)
    num_rev = df["num_review"].fillna(0).astype(float)
    feats["log_num_sold"] = np.log1p(num_sold)
    feats["log_num_review"] = np.log1p(num_rev)
    feats["is_zero_sold"] = (num_sold == 0).astype(float)
    feats["ratio_rev_sold"] = (num_rev + 1) / (num_sold + 1)
    feats["category_code"] = label_encoder.transform(df["category"].astype(str))

    # Text Analysis
    comments = df["comment"].fillna("").astype(str)
    descs = df["description"].fillna("").astype(str)
    products = df["product_name"].fillna("").astype(str)

    feats["char_len"] = comments.str.len()
    feats["word_count"] = comments.apply(lambda s: len(s.split()))
    feats["char_entropy"] = comments.apply(char_entropy)
    feats["max_char_repeat"] = comments.apply(max_repeat)
    feats["exclamation_count"] = comments.apply(lambda s: s.count("!"))
    feats["question_count"] = comments.apply(lambda s: s.count("?"))
    feats["uppercase_ratio"] = comments.apply(lambda s: sum(1 for c in s if c.isupper()) / (len(s) + 1))

    # Regex patterns
    def has_any(t, pats):
        t_low = t.lower()
        return float(any(bool(re.search(p, t_low)) for p in pats))

    feats["has_c1_pat"] = [has_any(c, C1_PATTERNS) for c in comments]
    feats["has_c2_pat"] = [has_any(c, C2_PATTERNS) for c in comments]
    feats["has_c3_pat"] = [has_any(c, C3_PATTERNS) for c in comments]

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
            if c[i:i+30] in d:
                return 1.0
        return 0.0

    feats["desc_jaccard"] = [jaccard(c, d) for c, d in zip(comments, descs)]
    feats["desc_exact_slice"] = [exact_slice(c, d) for c, d in zip(comments, descs)]
    feats["prod_jaccard"] = [jaccard(c, p) for c, p in zip(comments, products)]

    return feats


def main():
    print("Loading data...")
    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")
    sample = pd.read_csv(ROOT / "ds102_samplesubmission.csv", encoding="utf-8-sig")

    labels = train[LABEL].to_numpy()
    dev_labels = dev[LABEL].to_numpy()

    # Load 5-fold probability arrays
    xlm_oof = np.load(OUTPUT / "xlm_oof_probabilities.npy")
    xlm_dev = np.load(OUTPUT / "xlm_dev_probabilities.npy")
    xlm_test = np.load(OUTPUT / "xlm_test_probabilities.npy")

    le = LabelEncoder()
    le.fit(pd.concat([train["category"], dev["category"], test["category"]]).astype(str))

    print("Extracting features (40 tabular & linguistic signals)...")
    X_train = extract_features(train, xlm_oof, le)
    X_dev = extract_features(dev, xlm_dev, le)
    X_test = extract_features(test, xlm_test, le)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_lgb = np.zeros((len(labels), 4))
    dev_lgb = np.zeros((len(dev_labels), 4))
    test_lgb = np.zeros((len(X_test), 4))

    oof_xgb = np.zeros((len(labels), 4))
    dev_xgb = np.zeros((len(dev_labels), 4))
    test_xgb = np.zeros((len(X_test), 4))

    balanced_weights = np.array([1.0, 3.5, 1.8, 1.3])
    sample_weights = balanced_weights[labels]

    lgb_params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 25,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": 42,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "random_state": 42,
        "tree_method": "hist",
    }

    print("Training 5-Fold Level-2 Stacking (LightGBM + XGBoost)...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, labels), 1):
        X_tr, y_tr, w_tr = X_train.iloc[tr_idx], labels[tr_idx], sample_weights[tr_idx]
        X_val, y_val, w_val = X_train.iloc[val_idx], labels[val_idx], sample_weights[val_idx]

        # LightGBM
        train_lgb_data = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
        val_lgb_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_lgb_data)
        m_lgb = lgb.train(lgb_params, train_lgb_data, num_boost_round=600, valid_sets=[val_lgb_data],
                          callbacks=[lgb.early_stopping(35, verbose=False)])

        oof_lgb[val_idx] = m_lgb.predict(X_val)
        dev_lgb += m_lgb.predict(X_dev) / 5.0
        test_lgb += m_lgb.predict(X_test) / 5.0

        # XGBoost
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
        ddev = xgb.DMatrix(X_dev)
        dtest = xgb.DMatrix(X_test)

        m_xgb = xgb.train(xgb_params, dtrain, num_boost_round=600, evals=[(dval, "val")],
                          early_stopping_rounds=35, verbose_eval=False)

        oof_xgb[val_idx] = m_xgb.predict(dval)
        dev_xgb += m_xgb.predict(ddev) / 5.0
        test_xgb += m_xgb.predict(dtest) / 5.0

    gbdt_oof = 0.6 * oof_lgb + 0.4 * oof_xgb
    gbdt_dev = 0.6 * dev_lgb + 0.4 * dev_xgb
    gbdt_test = 0.6 * test_lgb + 0.4 * test_xgb

    w_s8 = np.array([1.0, 0.9, 1.1, 1.6])
    xlm_oof_norm = (xlm_oof * w_s8) / (xlm_oof * w_s8).sum(axis=1, keepdims=True)
    xlm_dev_norm = (xlm_dev * w_s8) / (xlm_dev * w_s8).sum(axis=1, keepdims=True)
    xlm_test_norm = (xlm_test * w_s8) / (xlm_test * w_s8).sum(axis=1, keepdims=True)

    # 35% Base Transformer + 65% Level-2 GBDT Ensemble
    final_dev = 0.35 * xlm_dev_norm + 0.65 * gbdt_dev
    final_oof = 0.35 * xlm_oof_norm + 0.65 * gbdt_oof
    final_test = 0.35 * xlm_test_norm + 0.65 * gbdt_test

    # Final calibration multiplier
    cal_weights = np.array([1.0, 1.1, 1.1, 1.1])
    cal_dev_probs = final_dev * cal_weights
    cal_test_probs = final_test * cal_weights

    dev_preds = cal_dev_probs.argmax(axis=1)
    test_preds = cal_test_probs.argmax(axis=1)

    macro_f1 = f1_score(dev_labels, dev_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(dev_labels, dev_preds, average="weighted", zero_division=0)
    report = classification_report(dev_labels, dev_preds, digits=4, output_dict=True)

    print("\n" + "=" * 60)
    print(f"DEV Macro F1: {macro_f1:.5f} (weighted F1: {weighted_f1:.5f})")
    print("=" * 60)
    for c in range(4):
        print(f"  Class {c}: F1 = {report[str(c)]['f1-score']:.4f} (prec: {report[str(c)]['precision']:.4f}, rec: {report[str(c)]['recall']:.4f})")

    # Generate submission_11.csv
    submission_11 = make_submission(test, sample, test_preds)
    sub11_output = OUTPUT / "submission_11.csv"
    sub11_root = ROOT / "submission_11.csv"
    submission_11.to_csv(sub11_output, index=False)
    submission_11.to_csv(sub11_root, index=False)
    print(f"\nSaved submission to:\n  - {sub11_output}\n  - {sub11_root}")

    test_dist = pd.Series(test_preds).value_counts().to_dict()
    print(f"Test Class Distribution: {test_dist}")

    metrics = {
        "model": "Level-2 Stacking (XLM-RoBERTa + 5-Fold LightGBM + 5-Fold XGBoost)",
        "features_count": X_train.shape[1],
        "dev_macro_f1": macro_f1,
        "dev_weighted_f1": weighted_f1,
        "dev_classification_report": report,
        "test_distribution": test_dist
    }
    (OUTPUT / "submission_11_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
