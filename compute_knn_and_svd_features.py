"""Compute Leak-Free Nearest-Neighbor Template/Seeding Features & SVD Semantic Coordinates.

Produces:
  1. Unsupervised global & intra-product review similarity features across all 19,870 reviews:
     max_sim_global, top5_mean_sim_global, dup_cnt_050, dup_cnt_070, max_sim_same_prod, mean_sim_same_prod
  2. Fold-matched Out-of-Fold KNN Neighbor features (identical pool size across OOF, Dev, and Test!):
     - Stage A (5-Fold on Train -> OOF, Dev, Test)
     - Stage B (10-Fold on Train+Dev -> Full OOF, Test)
  3. 16 TruncatedSVD semantic components over NFKC comment + product + category TF-IDF.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

from spam_pipeline import LABEL, load_frame

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "spam_output"


def norm_nfkc_clean(s):
    return " ".join(unicodedata.normalize("NFKC", str(s)).lower().split())


def compute_knn_block(sim_sub, ref_labels, priors, k=15):
    """Given (n_query, n_ref) cosine similarity matrix, return (n_query, 8):
    4 max_sim_per_class + 4 smoothed top-k neighbor class probabilities."""
    n_q = sim_sub.shape[0]
    out = np.zeros((n_q, 8), dtype=np.float32)

    for c in range(4):
        c_cols = np.where(ref_labels == c)[0]
        out[:, c] = sim_sub[:, c_cols].max(axis=1)

    top_idx = np.argpartition(-sim_sub, k, axis=1)[:, :k]
    top_vals = np.take_along_axis(sim_sub, top_idx, axis=1)
    weights = np.maximum(top_vals, 0.0) ** 3
    w_sum = weights.sum(axis=1) + 0.15  # Bayesian prior smoothing
    top_y = ref_labels[top_idx]

    for c in range(4):
        num = ((top_y == c) * weights).sum(axis=1) + 0.15 * priors[c]
        out[:, 4 + c] = num / w_sum
    return out


def main():
    t0 = time.time()
    train = load_frame(ROOT / "ds102_train.csv", labeled=True)
    dev = load_frame(ROOT / "ds102_dev.csv", labeled=True)
    test = load_frame(ROOT / "ds102_test.csv")

    labels = train[LABEL].to_numpy()
    dev_labels = dev[LABEL].to_numpy()
    n_tr, n_de, n_te = len(train), len(dev), len(test)

    all_df = pd.concat([train, dev, test], ignore_index=True)
    c_texts = [norm_nfkc_clean(x) for x in all_df["comment"].fillna("")]
    p_texts = [norm_nfkc_clean(x) for x in all_df["product_name"].fillna("")]
    cat_texts = [norm_nfkc_clean(x) for x in all_df["categories"].fillna("")]

    print("1. Fitting NFKC Word + Char TF-IDF for 19,870 reviews...")
    vec_w = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=100000, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=100000, sublinear_tf=True)
    M_comm = normalize(hstack([vec_w.fit_transform(c_texts), vec_c.fit_transform(c_texts)]).tocsr())

    print("2. Computing 16 TruncatedSVD semantic components...")
    vec_p = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    M_joint = hstack([M_comm, 0.5 * normalize(vec_p.fit_transform([f"{p} {ct}" for p, ct in zip(p_texts, cat_texts)]))]).tocsr()
    svd = TruncatedSVD(n_components=16, random_state=42)
    svd_feats = svd.fit_transform(M_joint).astype(np.float32)

    print("3. Computing unsupervised global & intra-product similarity features...")
    n_all = len(all_df)
    max_sim_global = np.zeros(n_all, dtype=np.float32)
    top5_mean_global = np.zeros(n_all, dtype=np.float32)
    dup_cnt_050 = np.zeros(n_all, dtype=np.float32)
    dup_cnt_070 = np.zeros(n_all, dtype=np.float32)

    chunk = 2000
    for start in range(0, n_all, chunk):
        end = min(start + chunk, n_all)
        S_blk = (M_comm[start:end] @ M_comm.T).toarray()
        for r_i in range(end - start):
            S_blk[r_i, start + r_i] = 0.0
        max_sim_global[start:end] = S_blk.max(axis=1)
        top5 = np.partition(S_blk, -5, axis=1)[:, -5:]
        top5_mean_global[start:end] = top5.mean(axis=1)
        dup_cnt_050[start:end] = np.log1p((S_blk >= 0.50).sum(axis=1))
        dup_cnt_070[start:end] = np.log1p((S_blk >= 0.70).sum(axis=1))

    # Intra-product comment similarity (catches coordinated shop seeding!)
    max_sim_same_prod = np.zeros(n_all, dtype=np.float32)
    mean_sim_same_prod = np.zeros(n_all, dtype=np.float32)
    prod_groups = all_df.groupby(p_texts).indices
    for _, idxs in prod_groups.items():
        if len(idxs) <= 1:
            continue
        sub_m = M_comm[idxs]
        s_p = (sub_m @ sub_m.T).toarray()
        np.fill_diagonal(s_p, 0.0)
        max_sim_same_prod[idxs] = s_p.max(axis=1)
        mean_sim_same_prod[idxs] = s_p.sum(axis=1) / (len(idxs) - 1.0)

    unsup_sim = np.column_stack([
        max_sim_global, top5_mean_global, dup_cnt_050, dup_cnt_070,
        max_sim_same_prod, mean_sim_same_prod,
    ]).astype(np.float32)

    print("4. Computing Fold-Matched Out-of-Fold KNN features (Stage A & Stage B)...")
    n_f = n_tr + n_de
    S_full_vs_full = (M_comm[:n_f] @ M_comm[:n_f].T).toarray()
    np.fill_diagonal(S_full_vs_full, 0.0)
    S_te_vs_full = (M_comm[n_f:] @ M_comm[:n_f].T).toarray()

    # Stage A: 5-Fold on Train (exact pool-size matching for OOF, Dev, and Test!)
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    priors_tr = np.bincount(labels) / len(labels)
    knn_a_oof = np.zeros((n_tr, 8), dtype=np.float32)
    knn_a_dev = np.zeros((n_de, 8), dtype=np.float32)
    knn_a_test = np.zeros((n_te, 8), dtype=np.float32)

    S_tr_tr = S_full_vs_full[:n_tr, :n_tr]
    S_de_tr = S_full_vs_full[n_tr:, :n_tr]
    S_te_tr = S_te_vs_full[:, :n_tr]

    for tr_idx, va_idx in skf5.split(train, labels):
        knn_a_oof[va_idx] = compute_knn_block(S_tr_tr[np.ix_(va_idx, tr_idx)], labels[tr_idx], priors_tr)
        knn_a_dev += compute_knn_block(S_de_tr[:, tr_idx], labels[tr_idx], priors_tr) / 5.0
        knn_a_test += compute_knn_block(S_te_tr[:, tr_idx], labels[tr_idx], priors_tr) / 5.0

    # Stage B: 10-Fold on Train+Dev (exact pool-size matching for Full OOF and Test!)
    y_full = np.concatenate([labels, dev_labels])
    priors_full = np.bincount(y_full) / len(y_full)
    skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=102)
    knn_b_full = np.zeros((n_f, 8), dtype=np.float32)
    knn_b_test = np.zeros((n_te, 8), dtype=np.float32)

    for tr_idx, va_idx in skf10.split(np.arange(n_f), y_full):
        knn_b_full[va_idx] = compute_knn_block(S_full_vs_full[np.ix_(va_idx, tr_idx)], y_full[tr_idx], priors_full)
        knn_b_test += compute_knn_block(S_te_vs_full[:, tr_idx], y_full[tr_idx], priors_full) / 10.0

    np.savez_compressed(
        OUTPUT / "knn_svd_features.npz",
        svd_feats=svd_feats,
        unsup_sim=unsup_sim,
        knn_a_oof=knn_a_oof,
        knn_a_dev=knn_a_dev,
        knn_a_test=knn_a_test,
        knn_b_full=knn_b_full,
        knn_b_test=knn_b_test,
    )
    print(f"Saved spam_output/knn_svd_features.npz in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
