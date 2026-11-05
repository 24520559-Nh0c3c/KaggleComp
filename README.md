# DS102 â€“ PhÃ¢n loáº¡i bÃ¬nh luáº­n spam thÆ°Æ¡ng máº¡i Ä‘iá»‡n tá»­ (Kaggle)

Dá»± Ã¡n giáº£i bÃ i toÃ¡n phÃ¢n loáº¡i bÃ¬nh luáº­n sáº£n pháº©m tiáº¿ng Viá»‡t trÃªn sÃ n thÆ°Æ¡ng máº¡i Ä‘iá»‡n tá»­ thÃ nh **4 lá»›p** (`spam_label` = 0, 1, 2, 3). Metric Ä‘Ã¡nh giÃ¡: **Macro F1**.

| Lá»›p | Ã nghÄ©a |
| :-: | :-- |
| 0 | ÄÃ¡nh giÃ¡ tháº­t |
| 1 | Spam khen lá»‘ / copy nguyÃªn vÄƒn mÃ´ táº£ sáº£n pháº©m |
| 2 | Spam dá»‹ch vá»¥ / chÆ°a dÃ¹ng sáº£n pháº©m |
| 3 | Spam rÃ¡c (kÃ½ tá»± láº·p, quáº£ng cÃ¡o, liÃªn há»‡...) |

## Kiáº¿n trÃºc tá»•ng quan

Giáº£i phÃ¡p lÃ  mÃ´ hÃ¬nh **stacking nhiá»u táº§ng**:

1. **Level-1 (mÃ´ hÃ¬nh ná»n):** cÃ¡c mÃ´ hÃ¬nh transformer 5-fold (XLM-RoBERTa, DeBERTa-v3, RoBERTa, PhoBERT), cÃ¹ng phiÃªn báº£n *context-aware* nháº­n thÃªm `rating`, `categories`, `product_name` vÃ  Ä‘oáº¡n mÃ´ táº£ liÃªn quan; ngoÃ i ra cÃ³ TF-IDF + Logistic Regression/SVC.
2. **Äáº·c trÆ°ng bá»• sung:** chuáº©n hÃ³a Unicode NFKC (xá»­ lÃ½ chá»¯ NFD vÃ  font toÃ¡n há»c dÃ¹ng Ä‘á»ƒ nÃ© lá»c), Ä‘á»™ tÆ°Æ¡ng Ä‘á»“ng bÃ¬nh luáº­nâ€“mÃ´ táº£, phÃ¡t hiá»‡n copy-paste, KNN/SVD trÃªn TF-IDF, thá»‘ng kÃª Leave-One-Out theo sáº£n pháº©m/danh má»¥c, cÃ¡c Ä‘áº·c trÆ°ng hÃ¬nh thá»©c vÄƒn báº£n.
3. **Level-2:** tá»• há»£p CatBoost, LightGBM, XGBoost (Stage A: 5-fold trÃªn Train; Stage B: 10-fold trÃªn Train+Dev), trá»™n vá»›i cÃ¡c submission trÆ°á»›c, rá»“i hiá»‡u chá»‰nh há»‡ sá»‘ theo lá»›p báº±ng nested-CV.

## Cáº¥u trÃºc thÆ° má»¥c

| File | Chá»©c nÄƒng |
| :-- | :-- |
| [spam_pipeline.py](spam_pipeline.py) | Pipeline transformer 5-fold gá»‘c, hÃ m `load_frame`, `make_submission` |
| [train_context_xlm.py](train_context_xlm.py) | Huáº¥n luyá»‡n XLM-RoBERTa cÃ³ ngá»¯ cáº£nh (Train+Dev) |
| [train_context_phobert.py](train_context_phobert.py) | Huáº¥n luyá»‡n PhoBERT cÃ³ ngá»¯ cáº£nh |
| [extract_xlm_nfkc_and_emb.py](extract_xlm_nfkc_and_emb.py) | Suy luáº­n XLM-R trÃªn vÄƒn báº£n NFKC |
| [train_nfkc_sparse_models.py](train_nfkc_sparse_models.py) | TF-IDF/LR/SVC vÃ  Ä‘áº·c trÆ°ng tÆ°Æ¡ng Ä‘á»“ng NFKC |
| [compute_knn_and_svd_features.py](compute_knn_and_svd_features.py) | Äáº·c trÆ°ng KNN, trÃ¹ng láº·p vÃ  SVD |
| [calibrate_xlm.py](calibrate_xlm.py), [evaluate_xlm.py](evaluate_xlm.py), [ensemble_and_calibrate.py](ensemble_and_calibrate.py), [try_tfidf_ensemble.py](try_tfidf_ensemble.py) | Hiá»‡u chá»‰nh, Ä‘Ã¡nh giÃ¡, ensemble cÃ¡c báº£n Ä‘áº§u |
| `build_*_stacking_submission.py` | CÃ¡c phiÃªn báº£n stacking tuáº§n tá»± (submission 11â€“14) |
| [build_top1_submission.py](build_top1_submission.py) | Pipeline cuá»‘i, xuáº¥t `submission_15.csv` |
| `test_*.py` | Kiá»ƒm thá»­ Ä‘Æ¡n vá»‹ |
| `spam_output/` | XÃ¡c suáº¥t OOF/Dev/Test (`.npy`), metrics vÃ  cÃ¡c file submission |
| `ds102_*.csv` | Dá»¯ liá»‡u train/dev/test vÃ  máº«u ná»™p bÃ i |

## Káº¿t quáº£

| Submission | MÃ´ táº£ | Public LB |
| :-- | :-- | :-: |
| 7 | XLM-R 5-fold | 0.76722 |
| 11 | Stacking LightGBM + XGBoost | 0.79574 |
| 13 | Super-stacking + thá»‘ng kÃª sáº£n pháº©m/danh má»¥c | 0.79811 |
| 14 | Context-PhoBERT + NFKC + 3 loáº¡i cÃ¢y | 0.80036 |
| 15 | Dual-context + KNN/SVD + tá»• há»£p cÃ¢y | chÆ°a ná»™p |

Äiá»ƒm Macro F1 OOF trÃªn Train cá»§a `submission_15` lÃ  khoáº£ng 0.809.

## CÃ¡ch cháº¡y

YÃªu cáº§u: Python 3.10+, `torch`, `transformers`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `pandas`, `numpy`. Huáº¥n luyá»‡n transformer cáº§n GPU (Ä‘Ã£ thá»­ trÃªn RTX 3050 Ti 4GB).

```bash
# 1. Huáº¥n luyá»‡n cÃ¡c mÃ´ hÃ¬nh ná»n (cáº§n GPU)
python spam_pipeline.py
python train_context_xlm.py
python train_context_phobert.py
python extract_xlm_nfkc_and_emb.py

# 2. Äáº·c trÆ°ng thÆ°a vÃ  Ä‘á»“ thá»‹ tÆ°Æ¡ng Ä‘á»“ng
python train_nfkc_sparse_models.py
python compute_knn_and_svd_features.py

# 3. Stacking vÃ  xuáº¥t file ná»™p
python build_grandmaster_stacking.py   # submission_14
python build_top1_submission.py        # submission_15
```

CÃ¡c checkpoint `*.pt` khÃ´ng Ä‘Æ°á»£c Ä‘Æ°a lÃªn Git do dung lÆ°á»£ng lá»›n, nÃªn cáº§n cháº¡y láº¡i bÆ°á»›c 1 Ä‘á»ƒ táº¡o láº¡i. RiÃªng bÆ°á»›c 3 cÃ³ thá»ƒ cháº¡y ngay náº¿u Ä‘Ã£ cÃ³ cÃ¡c file `.npy` trong `spam_output/`.

## Giáº¥y phÃ©p

Xem [LICENSE](LICENSE).
