# DS102 – Phân loại bình luận spam thương mại điện tử (Kaggle)

Dự án giải bài toán phân loại bình luận sản phẩm tiếng Việt trên sàn thương mại điện tử thành **4 lớp** (`spam_label` = 0, 1, 2, 3). Metric đánh giá: **Macro F1**.

| Lớp | Ý nghĩa |
| :-: | :-- |
| 0 | Đánh giá thật |
| 1 | Spam khen lố / quảng cáo shop / copy nguyên văn mô tả sản phẩm |
| 2 | Spam dịch vụ / giao hàng / chưa dùng sản phẩm |
| 3 | Spam rác (ký tự lặp, kiếm xu, nội dung lạc đề, liên hệ...) |

## Kiến trúc tổng quan

Giải pháp là hệ thống **Multi-Foundation Context Stacking nhiều tầng**:

1. **Level-1 (10 luồng mô hình nền):**
   - **Tổ hợp A100 Heavyweight Context-Aware Transformers (`Train+Dev = 15,896` mẫu, `BF16`):**
     - `xlm_large_ctx`: 5-Fold `xlm-roberta-large` (24 tầng, 1024d, `MAX_LEN=384`, `[CLS]+Mean+Max` + Layerwise LR Decay)
     - `mdeberta_ctx`: 5-Fold `microsoft/mdeberta-v3-base` (Disentangled Relative Attention đa ngôn ngữ, `MAX_LEN=384`)
     - `phobert_large_ctx`: 5-Fold `vinai/phobert-large` (24 tầng, 1024d Tiếng Việt chuyên sâu, `MAX_LEN=254`)
     - `xlm_ctx` & `xlm_ctx2`: 2 biến thể 5-Fold `xlm-roberta-base` có ngữ cảnh (Comment-First & Product-First)
     - `phobert_ctx`: 5-Fold `vinai/phobert-base` có ngữ cảnh
   - **Các luồng đối chứng chỉ bình luận & thưa:** `xlm-roberta` Raw+NFKC TTA, `deberta-v3`, `roberta`, và 5-Fold Word+Char `TF-IDF`.
2. **Đặc trưng miền chuyên sâu:** chuẩn hóa Unicode NFKC, nhận diện sàn thương mại điện tử (`Tiki` vs `Shopee`), phát hiện mẫu quảng cáo đại lý / cụm gợi ý nhận xu Shopee / đánh giá thuần giao hàng so với từ khóa đặc thù sản phẩm, đồ thị láng giềng `KNN` + 16 chiều `TruncatedSVD`, và thống kê `Leave-One-Out` theo sản phẩm & danh mục.
3. **Level-2 & Level-3 (`build_submission_17.py`):** tổ hợp 6 họ mô hình trên toàn bộ `15,896` mẫu (`Bagged XGBoost`, `Bagged LightGBM`, `LightGBM ExtraTrees`, `CatBoost Depth-6`, `CatBoost Depth-5`, và `PyTorch Residual Meta-MLP`), tối ưu trọng số tổ hợp Simplex và hệ số nhân theo lớp bằng **Strict 5-Fold Nested CV**.

## Cấu trúc thư mục

| File | Chức năng |
| :-- | :-- |
| [run_on_colab.ipynb](run_on_colab.ipynb) | Notebook chạy tự động từ đầu đến cuối trên Google Colab A100 (tích hợp sao lưu Google Drive + nộp Kaggle CLI) |
| [train_a100_championship_models.py](train_a100_championship_models.py) | Huấn luyện 4 mô hình nền hạng nặng trên A100 (`BF16`, lưu dự phòng từng Fold lên Google Drive) |
| [build_submission_17.py](build_submission_17.py) | Siêu tổ hợp Stacking 6 họ mô hình + đặc trưng sàn `Tiki/Shopee` + tự động nộp `submission_17.csv` |
| [train_context_xlm.py](train_context_xlm.py) | Huấn luyện XLM-RoBERTa-base có ngữ cảnh (Variant A & B) |
| [train_context_phobert.py](train_context_phobert.py) | Huấn luyện PhoBERT-base có ngữ cảnh |
| [compute_knn_and_svd_features.py](compute_knn_and_svd_features.py) | Đặc trưng đồ thị tương đồng KNN, trùng lặp và 16 chiều SVD |
| [build_submission_16.py](build_submission_16.py) | Pipeline xuất `submission_16.csv` (`0.80557` Public LB) |
| [build_top1_submission.py](build_top1_submission.py) | Pipeline xuất `submission_15.csv` (`0.80393` Public LB) |
| [spam_pipeline.py](spam_pipeline.py) | Pipeline transformer 5-fold gốc, hàm `load_frame`, `make_submission` |
| `spam_output/` | Xác suất OOF/Dev/Test (`.npy`), ma trận đặc trưng (`.npz`), metrics và các file submission |

## Kết quả tiến triển trên Public Leaderboard

| Submission | Mô tả | Public LB |
| :-- | :-- | :-: |
| 7 | XLM-R 5-fold cơ sở | 0.76722 |
| 11 | Stacking LightGBM + XGBoost | 0.79574 |
| 13 | Super-stacking + thống kê sản phẩm/danh mục | 0.79811 |
| 14 | Context-PhoBERT + NFKC + 3 loại cây | 0.80036 |
| 15 | Dual-context + KNN/SVD + tổ hợp 80 cây | 0.80393 |
| 16 | Dual-Run Context-XLM x2 + 80 cây + S15 anchor | **0.80557** |
| 17 | A100 Multi-Foundation (`XLM-Large + mDeBERTa-v3 + PhoBERT-Large + XLM-Ctx2`) + 6-Family Meta-Stacking | *Chạy trên Colab A100* |

## Cách chạy trên Google Colab (Khuyên dùng)

Mở trực tiếp [run_on_colab.ipynb](run_on_colab.ipynb) trên Google Colab (chọn **A100 GPU**), bấm **Runtime -> Run all**:
1. Tự động gắn kết **Google Drive** (`/content/drive/MyDrive/DS102_Colab_Cache`) để sao lưu mọi Fold và file `.npy` theo thời gian thực (không bao giờ mất dữ liệu khi tắt runtime).
2. Huấn luyện 4 kiến trúc Transformer ngữ cảnh bằng `BF16` và chạy `build_submission_17.py` để tự động nộp `submission_17.csv` lên Kaggle qua CLI.

## Giấy phép

Xem [LICENSE](LICENSE).
