"""Train 5-Fold NFKC-Normalized Multi-Field Sparse Classifiers & Compute Semantic Similarities.

Fixes the major NFD/NFKC Unicode mismatch (51.1% of descriptions and ~1.8% of comments use
decomposed NFD or fancy mathematical fonts) and preserves newline tokens (`__nl__`).
Saves:
  - spam_output/nfkc_lr_oof.npy, nfkc_lr_dev.npy, nfkc_lr_test.npy (unweighted calibrated LR)
  - spam_output/nfkc_svc_oof.npy, nfkc_svc_dev.npy, nfkc_svc_test.npy (calibrated LinearSVC)
  - spam_output/nfkc_sim_features.npz (row-wise cosine similarities & n-gram copy-paste metrics)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from spam_pipeline import LABEL, load_frame

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "spam_output"


def norm_nfkc_keep_nl(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n+", " __nl__ ", s)
    return " ".join(s.lower().split())


def norm_nfkc_clean(s):
    s = unicodedata.normalize("NFKC", str(s))
    return " ".join(s.lower().split())


def sliding_ngram_overlap(c_words, d_words, n):
    if len(c_words) < n or len(d_words) < n:
        return 0.0, 0
    d_set = set(tuple(d_words[i : i + n]) for i in range(len(d_words) - n + 1))
    c_grams = [tuple(c_words[i : i + n]) for i in range(len(c_words) - n + 1)]
    hits = sum(1 for g in c_grams if g in d_set)
    return hits / len(c_grams), hits


def max_window_word_jaccard(c_words, d_words, win=30, step=10):
    if not c_words or not d_words:
        return 0.0
    c_set = set(c_words)
    if len(d_words) <= win:
        d_set = set(d_words)
        return len(c_set & d_set) / len(c_set | d_set)
    best = 0.0
    for i in range(0, len(d_words) - win + 1, step):
        w_set = set(d_words[i : i + win])
        inter = len(c_set & w_set)
        if inter > 0:
            score = inter / len(c_set)
            if score > best:
                best = score
    return best


def main():
    print("Loading datasets...")
    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")
    labels = train[LABEL].to_numpy()
    dev_labels = dev[LABEL].to_numpy()

    all_df = pd.concat([train, dev, test], ignore_index=True)
    n_tr, n_de = len(train), len(dev)

    print("Normalizing texts with NFKC & computing row-wise TF-IDF cosine similarities...")
    c_nl = [norm_nfkc_keep_nl(x) for x in all_df["comment"].fillna("")]
    c_clean = [norm_nfkc_clean(x) for x in all_df["comment"].fillna("")]
    p_clean = [norm_nfkc_clean(x) for x in all_df["product_name"].fillna("")]
    d_clean = [norm_nfkc_clean(x) for x in all_df["description"].fillna("")]
    cat_clean = [norm_nfkc_clean(x) for x in all_df["categories"].fillna("")]
    r_tok = [f"__rating_{int(r)}__ __cat_{norm_nfkc_clean(cat)}__" for r, cat in zip(all_df["rating"].fillna(5), all_df["category"].fillna(""))]

    # Unsupervised global TF-IDF space for semantic similarity between comment, product_name, description, and category centroid
    sim_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=120000, sublinear_tf=True)
    sim_vec.fit(c_clean + p_clean + d_clean)

    C_mat = normalize(sim_vec.transform(c_clean))
    P_mat = normalize(sim_vec.transform(p_clean))
    D_mat = normalize(sim_vec.transform(d_clean))

    cos_c_p = np.asarray(C_mat.multiply(P_mat).sum(axis=1)).ravel()
    cos_c_d = np.asarray(C_mat.multiply(D_mat).sum(axis=1)).ravel()

    # Category centroid similarity (using product + comment vocabulary per category)
    cat_codes, uniques = pd.factorize(all_df["category"].astype(str))
    cat_centroids = np.zeros((len(uniques), C_mat.shape[1]), dtype=np.float32)
    CP_mat = normalize(C_mat + P_mat)
    for k in range(len(uniques)):
        mask = cat_codes == k
        centroid = np.asarray(CP_mat[mask].mean(axis=0))
        norm_val = np.linalg.norm(centroid)
        if norm_val > 0:
            cat_centroids[k] = centroid[0] / norm_val
    cat_cent_csr = csr_matrix(cat_centroids)
    row_cat_cent = cat_cent_csr[cat_codes]
    cos_c_cat = np.asarray(C_mat.multiply(row_cat_cent).sum(axis=1)).ravel()

    # N-gram copy-paste & window overlap features
    ov3_ratio = np.zeros(len(all_df), dtype=np.float32)
    ov3_hits = np.zeros(len(all_df), dtype=np.float32)
    ov5_ratio = np.zeros(len(all_df), dtype=np.float32)
    ov5_hits = np.zeros(len(all_df), dtype=np.float32)
    win_ov = np.zeros(len(all_df), dtype=np.float32)

    for i, (cc, dd) in enumerate(zip(c_clean, d_clean)):
        cw = cc.split()
        dw = dd.split()
        ov3_ratio[i], ov3_hits[i] = sliding_ngram_overlap(cw, dw, 3)
        ov5_ratio[i], ov5_hits[i] = sliding_ngram_overlap(cw, dw, 5)
        win_ov[i] = max_window_word_jaccard(cw, dw, win=30, step=10)

    np.savez_compressed(
        OUTPUT / "nfkc_sim_features.npz",
        cos_c_p=cos_c_p,
        cos_c_d=cos_c_d,
        cos_c_cat=cos_c_cat,
        ov3_ratio=ov3_ratio,
        ov3_hits=ov3_hits,
        ov5_ratio=ov5_ratio,
        ov5_hits=ov5_hits,
        win_ov=win_ov,
    )
    print("Saved spam_output/nfkc_sim_features.npz")

    # 5-Fold Sparse Level-1 Models (strictly aligned with seed=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_oof = np.zeros((n_tr, 4), dtype=np.float32)
    lr_dev = np.zeros((n_de, 4), dtype=np.float32)
    lr_test = np.zeros((len(test), 4), dtype=np.float32)

    svc_oof = np.zeros((n_tr, 4), dtype=np.float32)
    svc_dev = np.zeros((n_de, 4), dtype=np.float32)
    svc_test = np.zeros((len(test), 4), dtype=np.float32)

    tr_c, de_c, te_c = c_nl[:n_tr], c_nl[n_tr : n_tr + n_de], c_nl[n_tr + n_de :]
    tr_p, de_p, te_p = p_clean[:n_tr], p_clean[n_tr : n_tr + n_de], p_clean[n_tr + n_de :]
    tr_cat, de_cat, te_cat = cat_clean[:n_tr], cat_clean[n_tr : n_tr + n_de], cat_clean[n_tr + n_de :]
    tr_r, de_r, te_r = r_tok[:n_tr], r_tok[n_tr : n_tr + n_de], r_tok[n_tr + n_de :]

    for fold, (t_idx, v_idx) in enumerate(skf.split(train, labels), 1):
        v_word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=120000)
        v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=160000)
        v_prod = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=35000)
        v_meta = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=15000)

        X_tf = hstack([
            v_word.fit_transform([tr_c[i] for i in t_idx]),
            v_char.fit_transform([tr_c[i] for i in t_idx]),
            v_prod.fit_transform([tr_p[i] for i in t_idx]),
            v_meta.fit_transform([f"{tr_cat[i]} {tr_r[i]}" for i in t_idx]),
        ]).tocsr()

        def transform_split(sc, sp, scat, sr):
            return hstack([
                v_word.transform(sc),
                v_char.transform(sc),
                v_prod.transform(sp),
                v_meta.transform([f"{a} {b}" for a, b in zip(scat, sr)]),
            ]).tocsr()

        X_vf = transform_split([tr_c[i] for i in v_idx], [tr_p[i] for i in v_idx], [tr_cat[i] for i in v_idx], [tr_r[i] for i in v_idx])
        X_de = transform_split(de_c, de_p, de_cat, de_r)
        X_te = transform_split(te_c, te_p, te_cat, te_r)

        # 1. Mildly weighted LogisticRegression (well-calibrated probabilities)
        m_lr = LogisticRegression(C=2.5, max_iter=600, class_weight={0: 1.0, 1: 4.5, 2: 2.0, 3: 1.3}, random_state=42 + fold)
        m_lr.fit(X_tf, labels[t_idx])
        lr_oof[v_idx] = m_lr.predict_proba(X_vf)
        lr_dev += m_lr.predict_proba(X_de) / 5.0
        lr_test += m_lr.predict_proba(X_te) / 5.0

        # 2. Calibrated LinearSVC (margin-based hyperplane classifier)
        base_svc = LinearSVC(C=0.6, class_weight={0: 1.0, 1: 4.0, 2: 1.8, 3: 1.2}, random_state=42 + fold)
        m_svc = CalibratedClassifierCV(base_svc, cv=3, method="sigmoid")
        m_svc.fit(X_tf, labels[t_idx])
        svc_oof[v_idx] = m_svc.predict_proba(X_vf)
        svc_dev += m_svc.predict_proba(X_de) / 5.0
        svc_test += m_svc.predict_proba(X_te) / 5.0

        print(f"Fold {fold} | NFKC-LR F1: {f1_score(labels[v_idx], lr_oof[v_idx].argmax(axis=1), average='macro'):.5f} | NFKC-SVC F1: {f1_score(labels[v_idx], svc_oof[v_idx].argmax(axis=1), average='macro'):.5f}")

    print(f"Overall NFKC-LR  OOF F1: {f1_score(labels, lr_oof.argmax(axis=1), average='macro'):.5f} | DEV F1: {f1_score(dev_labels, lr_dev.argmax(axis=1), average='macro'):.5f}")
    print(f"Overall NFKC-SVC OOF F1: {f1_score(labels, svc_oof.argmax(axis=1), average='macro'):.5f} | DEV F1: {f1_score(dev_labels, svc_dev.argmax(axis=1), average='macro'):.5f}")

    np.save(OUTPUT / "nfkc_lr_oof.npy", lr_oof)
    np.save(OUTPUT / "nfkc_lr_dev.npy", lr_dev)
    np.save(OUTPUT / "nfkc_lr_test.npy", lr_test)

    np.save(OUTPUT / "nfkc_svc_oof.npy", svc_oof)
    np.save(OUTPUT / "nfkc_svc_dev.npy", svc_dev)
    np.save(OUTPUT / "nfkc_svc_test.npy", svc_test)
    print("Saved NFKC LR & SVC probability arrays.")


if __name__ == "__main__":
    main()
