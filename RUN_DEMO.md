# Hướng dẫn chạy demo trên Windows

## 1. Tạo môi trường

```powershell
cd C:\Users\ROG\idp-invoice-extraction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn activate:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 1b. Bật OCR GPU thật (PaddleOCR, khuyến nghị cho đồ án)

```powershell
python scripts\check_gpu_env.py          # xác minh Python/Paddle/CUDA/GPU
pip uninstall -y opencv-python-headless  # chỉ còn 1 package cung cấp cv2
pip install -r requirements-gpu.txt      # paddlepaddle-gpu 3.4.0 (cu129) + paddleocr 3.7.0
```

Rồi đặt trong `.env`:

```dotenv
IDP_USE_PADDLE=true
IDP_PADDLE_DEVICE=auto
IDP_PADDLE_LANG=vi
IDP_PADDLE_OCR_VERSION=PP-OCRv6
IDP_PADDLE_MODEL_SIZE=medium
IDP_OCR_STRICT=true
```

- `IDP_PADDLE_DEVICE=auto` ưu tiên CUDA và tự fallback về CPU khi GPU lỗi; `gpu` bắt buộc GPU (fail-fast); `cpu` bỏ qua GPU.
- `IDP_OCR_STRICT=true` tắt sidecar/mock trong real mode: engine lỗi → API trả `503` thay vì bịa kết quả.
- Model tự tải về `~\.paddlex\official_models` ở lần chạy đầu (det+rec ~135MB). Từ lần sau chạy offline được.
- Benchmark: `python scripts\benchmark_ocr.py data\uploads\*.webp --repeats 5`

## 2. Khởi động

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000/`. Không cần chạy frontend riêng vì FastAPI phục vụ giao diện tĩnh. Lần extract đầu tiên sau khi mở server có thể mất 1-2s (GPU warm-up); model đã tải trước thì ổn định ở ~250ms OCR/ảnh.

## 3. Real mode với PaddleOCR GPU (mặc định khi `IDP_USE_PADDLE=true`)

Upload ảnh hóa đơn trong UI và nhấn **Rút trích dữ liệu**. Engine hiển thị ngay trên form (`OCR: PaddleOCR · GPU (CUDA) · vi`). Chỉ số ở chip metrics gồm thời gian OCR/thiết bị. Khu vực "Văn bản OCR demo" tự khóa trong real mode vì sidecar không còn được đọc.

Dòng thời gian xử lý trên hóa đơn mẫu (1131×1600, RTX 5060 Laptop 8GB):

| Giai đoạn | GPU (PP-OCRv6 medium) | CPU fallback |
|---|---|---|
| Preprocess | ~300 ms | ~320 ms |
| OCR | **~250 ms** | ~12.000 ms |
| KIE + validation | ~25 ms | ~25 ms |
| **Tổng cold** | ~1.6-1.9 s | ~13 s |

VRAM chiếm dụng: ~590 MB (`IDP_PADDLE_GPU_MEMORY_FRACTION=0.5` giới hạn pool của Paddle). Kết quả trên hóa đơn mẫu: 39 token, confidence trung bình 0.971, tổng 12.000.000đ khớp và status VALIDATED.

## 4. Sidecar text offline (demo không model)

Đặt `receipt.txt` cùng thư mục với `receipt.jpg`, rồi upload `receipt.jpg`. Sidecar được đọc khi PaddleOCR không khả dụng. Mỗi dòng nên là một dòng hóa đơn; KIE nhận diện mẫu nhãn tiếng Việt và dòng item dạng:

```text
NƯỚC ÉP CAM
Địa chỉ: 12 Lê Lợi, Đà Nẵng
Số hóa đơn: HD-001
Ngày: 22/10/2026 14:30
Nước cam 2 30.000 60.000
Bánh ngọt 1 25.000 25.000
Thuế VAT: 8.500
Giảm giá: 0
Tổng cộng: 93.500
```

Confidence của sidecar cố ý là `0.55`, vì vậy trạng thái mặc định là `NEEDS_REVIEW`; đây là hành vi đúng để minh họa human-in-the-loop.

## 5. Các bước preprocessing

`ImagePreprocessor` chạy theo thứ tự:

1. Giải mã bytes/path bằng Pillow, áp dụng EXIF orientation và bỏ alpha.
2. Giới hạn kích thước dài nhất (`IDP_MAX_IMAGE_DIMENSION`) để tránh ảnh quá lớn.
3. Canny + morphology + contour scoring để tìm tứ giác tờ hóa đơn; warp 4 điểm khi đủ tin cậy.
4. Deskew bằng foreground projection/minAreaRect, chỉ xoay khi góc hợp lệ.
5. Khử nền sáng không đều bằng background morphology và phép chia illumination.
6. Fast non-local means denoise.
7. CLAHE local contrast enhancement.
8. Unsharp mask tùy chọn.
9. Adaptive threshold hoặc Otsu tạo ảnh nhị phân sạch cho các model cần input binary.

Endpoint preview trả về `original`, `enhanced`, `binary` dạng data URL cùng metrics: kích thước, góc deskew, confidence phát hiện trang, variance Laplacian, mean/std độ sáng và warning. Các ảnh enhanced/binary được lưu trong `data/processed` khi chạy extract để bounding boxes khớp với ảnh đang review.

### Tinh chỉnh qua `.env`

Copy `.env.example` thành `.env` và chỉnh:

```dotenv
IDP_PIPELINE_MODE=heuristic
IDP_USE_PADDLE=false
IDP_USE_VIETOCR=false
IDP_MAX_UPLOAD_BYTES=12000000
IDP_MAX_IMAGE_DIMENSION=3000
IDP_MAX_DECODED_PIXELS=40000000
IDP_REVIEW_CONFIDENCE_THRESHOLD=0.85
```

## 6. Hybrid — OCR + KIE + VLM cục bộ dự phòng (P4)

Luồng đầy đủ: **Upload → Preprocessing → PaddleOCR GPU → Spatial KIE → Validation → Quality gate → VLM fallback (chỉ khi cần) → Reconciliation → Human review**.

### Cài đặt (1 lần) — venv riêng cho VLM

Windows không cho phép Paddle (cuDNN 9.9) và torch (cuDNN 9.10) chung một process — CUDA runtime DLL trùng tên sẽ làm bên import sau gãy. Vì vậy VLM chạy trong **venv + process riêng**:

```powershell
py -3.12 -m venv C:\Users\khoas\idp-invoice-extraction-main\.venv-vlm
C:\Users\khoas\idp-invoice-extraction-main\.venv-vlm\Scripts\Activate.ps1
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # Blackwell sm_120
pip install -r requirements-vlm.txt                                          # transformers + bitsandbytes
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct                         # ~7 GB vào HF cache (dùng chung)
deactivate
```

Rồi trong `.env` (venv chính **không** cài torch): `IDP_VLM_PROVIDER=local` + `IDP_VLM_WORKER_PYTHON=C:\Users\khoas\idp-invoice-extraction-main\.venv-vlm\Scripts\python.exe`. Kiểm tra nhanh: chip "VLM dự phòng: sẵn sàng" trên UI hoặc `GET /api/ocr/status` → `vlm_available: true`. Adapter **không bao giờ tải model giữa request** — model phải có sẵn trong cache.

### Cách chạy

- UI: chọn **Hybrid · VLM dự phòng** trong dropdown Pipeline rồi nhấn Rút trích.
- API: `POST /api/extract` với `mode=hybrid` (mode `ocr_kie` là bí danh của `heuristic`).

### Quality gate — khi nào VLM chạy?

VLM **chỉ** được gọi khi OCR+KIE chưa đủ tin: thiếu trường quan trọng (`seller/total_cost/timestamp`, cấu hình qua `IDP_HYBRID_CRITICAL_FIELDS`), kie_conf thấp, validation tổng tiền lỗi, không có dòng hàng, hoặc chỉ 1 dòng hàng khớp tổng (nghi gộp dòng). KIE sạch → **không tốn** thời gian VLM.

### Reconciliation (KIE-first)

- KIE đủ tin (kie_conf ≥ `IDP_HYBRID_MIN_KIE_CONFIDENCE`=0.75): **giữ KIE**, bất đồng VLM chỉ ghi nhận (resolved).
- KIE yếu + VLM khác + VLM có bằng chứng (evidence trích từ ảnh): dùng VLM, đánh dấu **xung đột cần soát** → `NEEDS_REVIEW`.
- Trường KIE thiếu: điền từ VLM (source `vlm`); VLM không có bằng chứng → bỏ.
- Trường VLM **không bịa confidence** — provenance (nguồn + evidence) là tín hiệu cho reviewer.
- Sau merge, validation chạy lại trên hóa đơn cuối.

Response thêm: `vlm_triggered`, `fallback_reasons`, `sources` (kie/vlm/kie+vlm), `conflicts`; `field_details` có thêm `source` + `evidence`; `timings.vlm_ms`. UI hiển thị các mục này ở khối provenance dưới form.

### VRAM trên RTX 5060 8 GB

Paddle ~0.6 GB + Qwen 3B 4-bit (NF4 double-quant) ~2.5–3.5 GB + activation. OCR và VLM **chạy tuần tự** (không đồng thời). `IDP_VLM_MAX_PIXELS=1003520` (≈1 MP) giữ số visual token vừa phải; đặt `IDP_VLM_KEEP_LOADED=false` nếu cần nhả VRAM sau mỗi lần gọi. Lần gọi VLM đầu tiên sau khi khởi động sẽ load model (~10–30 s); các lần sau nhanh hơn nhiều.

### Đánh giá Hybrid

```powershell
python scripts\evaluate_pipeline.py evaluation\fixtures\invoice_ground_truth.jsonl --mode hybrid
```

In thêm VLM fallback rate, VRAM peak/ảnh và ghi vào predictions JSONL. Baseline tham chiếu của OCR+KIE là **158/176 (89.8%)** trên 11 fixture KIE — không phải kết quả Hybrid.

## 7. VLM structured output

OpenAI:

```dotenv
IDP_PIPELINE_MODE=vlm
IDP_VLM_PROVIDER=openai
IDP_OPENAI_API_KEY=...
IDP_VLM_MODEL=gpt-4o-mini
```

Gemini:

```dotenv
IDP_PIPELINE_MODE=vlm
IDP_VLM_PROVIDER=gemini
IDP_GEMINI_API_KEY=...
IDP_VLM_MODEL=gemini-2.0-flash
```

Provider chỉ được gọi khi chọn `vlm`; key không bao giờ commit vào git. Nếu provider chưa cài/configured, API trả lỗi rõ ràng thay vì giả vờ có kết quả.

## 8. Chuyển database

SQLite là mặc định. MySQL:

```dotenv
IDP_DATABASE_URL=mysql+pymysql://user:password@localhost:3306/idp_invoice
```

PostgreSQL cần cài driver tương ứng, ví dụ `psycopg[binary]`, rồi đặt URL SQLAlchemy.

## 9. Đánh giá

```powershell
python evaluation\evaluate_metrics.py evaluation\fixtures\sample_predictions.jsonl
```

Chạy evaluation đầu-cuối (pipeline thật + ground truth):

```powershell
python scripts\evaluate_pipeline.py evaluation\fixtures\invoice_ground_truth.jsonl
```

CER/WER được tính trên text đã chuẩn hóa Unicode; Entity F1 so sánh key-value sau canonicalization số/ngày, gồm cả micro và macro report.

## 10. Fine-tuning

Mở `notebooks/train_finetune_guide.ipynb` để xem format dataset MC-OCR, split train/validation, crop sau detection, augmentation và skeleton fine-tune VietOCR/LayoutLMv3. Notebook là hướng dẫn, không tự tải model khi import.
