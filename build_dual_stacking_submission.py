import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(".")
OUTPUT = ROOT / "spam_output"

train = pd.read_csv("ds102_train.csv")
dev = pd.read_csv("ds102_dev.csv")
test = pd.read_csv("ds102_test.csv")
sample = pd.read_csv("ds102_samplesubmission.csv")

labels = train["spam_label"].to_numpy()
dev_labels = dev["spam_label"].to_numpy()

xlm_oof = np.load("spam_output/xlm_oof_probabilities.npy")
xlm_dev = np.load("spam_output/xlm_dev_probabilities.npy")
xlm_test = np.load("spam_output/xlm_test_probabilities.npy")

deb_oof = np.load("spam_output/deberta_oof_probabilities.npy")
deb_dev = np.load("spam_output/deberta_dev_probabilities.npy")
deb_test = np.load("spam_output/deberta_test_probabilities.npy")

le = LabelEncoder()
le.fit(pd.concat([train["category"], dev["category"], test["category"]]).astype(str))

# Patterns
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
    repeats = [len(m.group()) for m in re.finditer(r'(.)\1+', str(s))]
    return max(repeats) if repeats else 1

def char_entropy(s):
    s = str(s).lower()
    if len(s) <= 1:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * np.log2(c / n) for c in counts.values())

def extract_features(df, xlm_p, deb_p):
    feats = pd.DataFrame()

    # XLM features
    for c in range(4):
        p = np.clip(xlm_p[:, c], 1e-6, 1.0 - 1e-6)
        feats[f"xlm_p_{c}"] = p
        feats[f"xlm_logit_{c}"] = np.log(p / (1.0 - p))

    feats["xlm_max_prob"] = xlm_p.max(axis=1)
    feats["xlm_entropy"] = -np.sum(xlm_p * np.log(np.clip(xlm_p, 1e-6, 1.0)), axis=1)
    feats["xlm_diff_p0_p2"] = xlm_p[:, 0] - xlm_p[:, 2]
    feats["xlm_diff_p0_p3"] = xlm_p[:, 0] - xlm_p[:, 3]
    feats["xlm_diff_p1_p3"] = xlm_p[:, 1] - xlm_p[:, 3]
    feats["xlm_diff_p1_p0"] = xlm_p[:, 1] - xlm_p[:, 0]
    feats["xlm_diff_p2_p3"] = xlm_p[:, 2] - xlm_p[:, 3]

    # DeBERTa features
    for c in range(4):
        p = np.clip(deb_p[:, c], 1e-6, 1.0 - 1e-6)
        feats[f"deb_p_{c}"] = p
        feats[f"deb_logit_{c}"] = np.log(p / (1.0 - p))

    feats["deb_max_prob"] = deb_p.max(axis=1)
    feats["deb_entropy"] = -np.sum(deb_p * np.log(np.clip(deb_p, 1e-6, 1.0)), axis=1)

    # Cross-model agreement and differentials
    for c in range(4):
        feats[f"diff_xlm_deb_{c}"] = xlm_p[:, c] - deb_p[:, c]
    feats["model_agreement"] = (xlm_p.argmax(axis=1) == deb_p.argmax(axis=1)).astype(float)

    # Metadata features
    rating = df["rating"].fillna(5).astype(float)
    feats["rating"] = rating
    feats["is_rating_5"] = (rating == 5).astype(float)
    feats["is_rating_low"] = (rating <= 2).astype(float)
    feats["rating_x_xlm_p0"] = rating * xlm_p[:, 0]
    feats["rating_x_xlm_p2"] = rating * xlm_p[:, 2]
    feats["rating_x_xlm_p3"] = rating * xlm_p[:, 3]
    feats["rating_x_xlm_p1"] = rating * xlm_p[:, 1]

    num_sold = df["num_sold"].fillna(0).astype(float)
    num_rev = df["num_review"].fillna(0).astype(float)
    feats["log_num_sold"] = np.log1p(num_sold)
    feats["log_num_review"] = np.log1p(num_rev)
    feats["is_zero_sold"] = (num_sold == 0).astype(float)
    feats["ratio_rev_sold"] = (num_rev + 1) / (num_sold + 1)
    feats["category_code"] = le.transform(df["category"].astype(str))

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

print("1. Extracting 55 features across XLM & DeBERTa...")
X_tr = extract_features(train, xlm_oof, deb_oof)
X_de = extract_features(dev, xlm_dev, deb_dev)
X_te = extract_features(test, xlm_test, deb_test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(labels), 4))
dev_lgb = np.zeros((len(dev_labels), 4))
test_lgb = np.zeros((len(X_te), 4))

oof_xgb = np.zeros((len(labels), 4))
dev_xgb = np.zeros((len(dev_labels), 4))
test_xgb = np.zeros((len(X_te), 4))

balanced_weights = np.array([1.0, 3.5, 1.8, 1.3])
sample_weights = balanced_weights[labels]

lgb_params = {
    "objective": "multiclass",
    "num_class": 4,
    "metric": "multi_logloss",
    "learning_rate": 0.035,
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
    "learning_rate": 0.045,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "random_state": 42,
    "tree_method": "hist",
}

print("2. Training 5-Fold Level-2 Stacking (LightGBM + XGBoost)...")
for fold, (t_idx, v_idx) in enumerate(skf.split(X_tr, labels), 1):
    X_train_f, y_train_f, w_train_f = X_tr.iloc[t_idx], labels[t_idx], sample_weights[t_idx]
    X_val_f, y_val_f, w_val_f = X_tr.iloc[v_idx], labels[v_idx], sample_weights[v_idx]

    # LightGBM
    dtrain_lgb = lgb.Dataset(X_train_f, label=y_train_f, weight=w_train_f)
    dval_lgb = lgb.Dataset(X_val_f, label=y_val_f, weight=w_val_f, reference=dtrain_lgb)
    m_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=600, valid_sets=[dval_lgb],
                      callbacks=[lgb.early_stopping(35, verbose=False)])

    oof_lgb[v_idx] = m_lgb.predict(X_val_f)
    dev_lgb += m_lgb.predict(X_de) / 5.0
    test_lgb += m_lgb.predict(X_te) / 5.0

    # XGBoost
    dtrain_xgb = xgb.DMatrix(X_train_f, label=y_train_f, weight=w_train_f)
    dval_xgb = xgb.DMatrix(X_val_f, label=y_val_f, weight=w_val_f)
    ddev_xgb = xgb.DMatrix(X_de)
    dtest_xgb = xgb.DMatrix(X_te)

    m_xgb = xgb.train(xgb_params, dtrain_xgb, num_boost_round=600, evals=[(dval_xgb, "val")],
                      early_stopping_rounds=35, verbose_eval=False)

    oof_xgb[v_idx] = m_xgb.predict(dval_xgb)
    dev_xgb += m_xgb.predict(ddev_xgb) / 5.0
    test_xgb += m_xgb.predict(dtest_xgb) / 5.0

gbdt_oof = 0.55 * oof_lgb + 0.45 * oof_xgb
gbdt_dev = 0.55 * dev_lgb + 0.45 * dev_xgb
gbdt_test = 0.55 * test_lgb + 0.45 * test_xgb

print(f"\nLevel-2 GBDT Ensemble OOF F1: {f1_score(labels, gbdt_oof.argmax(axis=1), average='macro'):.5f}")
print(f"Level-2 GBDT Ensemble DEV F1: {f1_score(dev_labels, gbdt_dev.argmax(axis=1), average='macro'):.5f}")

# Blend base models (85% XLM + 15% DeBERTa calibrated)
w_s8 = np.array([1.0, 0.9, 1.1, 1.6])
base_oof = 0.85 * xlm_oof + 0.15 * deb_oof
base_dev = 0.85 * xlm_dev + 0.15 * deb_dev
base_test = 0.85 * xlm_test + 0.15 * deb_test

base_oof_norm = (base_oof * w_s8) / (base_oof * w_s8).sum(axis=1, keepdims=True)
base_dev_norm = (base_dev * w_s8) / (base_dev * w_s8).sum(axis=1, keepdims=True)
base_test_norm = (base_test * w_s8) / (base_test * w_s8).sum(axis=1, keepdims=True)

# Grid search for the optimal blend ratio and calibration weights
best_dev = 0
best_cfg = None

for alpha in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35]:
    f_dev = alpha * base_dev_norm + (1.0 - alpha) * gbdt_dev
    f_oof = alpha * base_oof_norm + (1.0 - alpha) * gbdt_oof
    
    for w1 in [0.95, 1.0, 1.05, 1.1, 1.15, 1.2]:
        for w2 in [0.95, 1.0, 1.05, 1.1]:
            for w3 in [0.95, 1.0, 1.05, 1.1]:
                w = np.array([1.0, w1, w2, w3])
                s_dev = f1_score(dev_labels, (f_dev * w).argmax(axis=1), average="macro")
                if s_dev > best_dev:
                    best_dev = s_dev
                    s_oof = f1_score(labels, (f_oof * w).argmax(axis=1), average="macro")
                    best_cfg = (alpha, w, s_dev, s_oof)

alpha_opt, w_opt, dev_score, oof_score = best_cfg
print("\n" + "=" * 65)
print(f"Optimal Configuration: alpha={alpha_opt:.2f}, weights={w_opt.tolist()}")
print(f"DEV Macro F1: {dev_score:.5f} | OOF Macro F1: {oof_score:.5f}")
print("=" * 65)

final_dev = alpha_opt * base_dev_norm + (1.0 - alpha_opt) * gbdt_dev
final_test = alpha_opt * base_test_norm + (1.0 - alpha_opt) * gbdt_test

dev_preds = (final_dev * w_opt).argmax(axis=1)
test_preds = (final_test * w_opt).argmax(axis=1)

print("\nDetailed Classification Report on DEV:")
report = classification_report(dev_labels, dev_preds, digits=4, output_dict=True)
print(classification_report(dev_labels, dev_preds, digits=4))

# Produce submission_12.csv
sample["spam_label"] = test_preds
sub12_output = OUTPUT / "submission_12.csv"
sub12_root = ROOT / "submission_12.csv"
sample.to_csv(sub12_output, index=False)
sample.to_csv(sub12_root, index=False)
print(f"\nSaved submission to:\n  - {sub12_output}\n  - {sub12_root}")

test_dist = pd.Series(test_preds).value_counts().to_dict()
print(f"Test Class Distribution: {test_dist}")

metrics = {
    "model": "Dual-Transformer Level-2 Stacking (XLM + DeBERTa + 5-Fold LGBM + 5-Fold XGBoost)",
    "features_count": X_tr.shape[1],
    "alpha": alpha_opt,
    "weights": w_opt.tolist(),
    "dev_macro_f1": dev_score,
    "oof_macro_f1": oof_score,
    "dev_classification_report": report,
    "test_distribution": test_dist
}
(OUTPUT / "submission_12_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
