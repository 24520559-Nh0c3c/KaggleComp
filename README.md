# DS102 – Phân loại bình luận spam thương mại điện tử (Kaggle)

Dự án giải bài toán phân loại bình luận sản phẩm tiếng Việt trên sàn thương mại điện tử thành **4 lớp** (`spam_label` = 0, 1, 2, 3). Metric đánh giá: **Macro F1**.

| Lớp | Ý nghĩa |
| :-: | :-- |
| 0 | Đánh giá thật |
| 1 | Spam khen lố / copy nguyên văn mô tả sản phẩm |
| 2 | Spam dịch vụ / chưa dùng sản phẩm |
| 3 | Spam rác (ký tự lặp, quảng cáo, liên hệ...) |

## Kiến trúc tổng quan

Giải pháp là mô hình **stacking nhiều tầng**:

1. **Level-1 (mô hình nền):** các mô hình transformer 5-fold (XLM-RoBERTa, DeBERTa-v3, RoBERTa, PhoBERT), cùng phiên bản *context-aware* nhận thêm `rating`, `categories`, `product_name` và đoạn mô tả liên quan; ngoài ra có TF-IDF + Logistic Regression/SVC.
2. **Đặc trưng bổ sung:** chuẩn hóa Unicode NFKC (xử lý chữ NFD và font toán học dùng để né lọc), độ tương đồng bình luận–mô tả, phát hiện copy-paste, KNN/SVD trên TF-IDF, thống kê Leave-One-Out theo sản phẩm/danh mục, các đặc trưng hình thức văn bản.
3. **Level-2:** tổ hợp CatBoost, LightGBM, XGBoost (Stage A: 5-fold trên Train; Stage B: 10-fold trên Train+Dev), trộn với các submission trước, rồi hiệu chỉnh hệ số theo lớp bằng nested-CV.

## Cấu trúc thư mục

| File | Chức năng |
| :-- | :-- |
| [spam_pipeline.py](spam_pipeline.py) | Pipeline transformer 5-fold gốc, hàm `load_frame`, `make_submission` |
| [train_context_xlm.py](train_context_xlm.py) | Huấn luyện XLM-RoBERTa có ngữ cảnh (Train+Dev) |
| [train_context_phobert.py](train_context_phobert.py) | Huấn luyện PhoBERT có ngữ cảnh |
| [extract_xlm_nfkc_and_emb.py](extract_xlm_nfkc_and_emb.py) | Suy luận XLM-R trên văn bản NFKC |
| [train_nfkc_sparse_models.py](train_nfkc_sparse_models.py) | TF-IDF/LR/SVC và đặc trưng tương đồng NFKC |
| [compute_knn_and_svd_features.py](compute_knn_and_svd_features.py) | Đặc trưng KNN, trùng lặp và SVD |
| [calibrate_xlm.py](calibrate_xlm.py), [evaluate_xlm.py](evaluate_xlm.py), [ensemble_and_calibrate.py](ensemble_and_calibrate.py), [try_tfidf_ensemble.py](try_tfidf_ensemble.py) | Hiệu chỉnh, đánh giá, ensemble các bản đầu |
| `build_*_stacking_submission.py` | Các phiên bản stacking tuần tự (submission 11–14) |
| [build_top1_submission.py](build_top1_submission.py) | Pipeline cuối, xuất `submission_15.csv` |
| `test_*.py` | Kiểm thử đơn vị |
| `spam_output/` | Xác suất OOF/Dev/Test (`.npy`), metrics và các file submission |
| `ds102_*.csv` | Dữ liệu train/dev/test và mẫu nộp bài |

## Kết quả

| Submission | Mô tả | Public LB |
| :-- | :-- | :-: |
| 7 | XLM-R 5-fold | 0.76722 |
| 11 | Stacking LightGBM + XGBoost | 0.79574 |
| 13 | Super-stacking + thống kê sản phẩm/danh mục | 0.79811 |
| 14 | Context-PhoBERT + NFKC + 3 loại cây | 0.80036 |
| 15 | Dual-context + KNN/SVD + tổ hợp cây | chưa nộp |

Điểm Macro F1 OOF trên Train của `submission_15` là khoảng 0.809.

## Cách chạy

Yêu cầu: Python 3.10+, `torch`, `transformers`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `pandas`, `numpy`. Huấn luyện transformer cần GPU (đã thử trên RTX 3050 Ti 4GB).

```bash
# 1. Huấn luyện các mô hình nền (cần GPU)
python spam_pipeline.py
python train_context_xlm.py
python train_context_phobert.py
python extract_xlm_nfkc_and_emb.py

# 2. Đặc trưng thưa và đồ thị tương đồng
python train_nfkc_sparse_models.py
python compute_knn_and_svd_features.py

# 3. Stacking và xuất file nộp
python build_grandmaster_stacking.py   # submission_14
python build_top1_submission.py        # submission_15
```

Các checkpoint `*.pt` không được đưa lên Git do dung lượng lớn, nên cần chạy lại bước 1 để tạo lại. Riêng bước 3 có thể chạy ngay nếu đã có các file `.npy` trong `spam_output/`.

## Giấy phép

Xem [LICENSE](LICENSE).
