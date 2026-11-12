# Vietnamese Invoice IDP

Hệ thống demo **Intelligent Document Processing (IDP)** rút trích dữ liệu hóa đơn tiếng Việt từ ảnh. Có hai chế độ: demo offline nhanh (preprocessing + KIE + validation không cần model) và **real mode với PaddleOCR chạy GPU NVIDIA** cho nhận diện thật.

## Điểm nổi bật

- **Image preprocessing đầy đủ**: đọc EXIF và xoay đúng chiều, chuẩn hóa màu/alpha, giới hạn kích thước, phát hiện tờ giấy bằng contour tứ giác, perspective warp, deskew, khử bóng/ánh sáng không đều, denoise, CLAHE, unsharp mask, adaptive/Otsu threshold và đo chất lượng ảnh.
- **OCR thật trên GPU**: PaddleOCR 3.7 + PaddlePaddle 3.4 (CUDA 12.9) cho tiếng Việt lẫn tiếng Anh, bật/tắt linh hoạt bằng `.env`. Tự chọn GPU/CPU, giới hạn VRAM (batch size, memory fraction), tự fallback CPU khi GPU lỗi, và tắt hoàn toàn sidecar/mock ở real mode.
- **Hybrid OCR + VLM cục bộ (P4)**: quality gate sau KIE — chỉ gọi Qwen2.5-VL-3B (4-bit, chạy GPU máy này, không cần API key) khi thiếu trường quan trọng, confidence thấp hoặc validation lỗi; đối soát KIE-first giữ nguyên giá trị tin cậy, điền trường thiếu bằng VLM có bằng chứng, xung đột chưa rõ → `NEEDS_REVIEW`; UI hiển thị nguồn từng trường (KIE/VLM), lý do fallback và xung đột.
- **Strategy Pattern**: chuyển giữa pipeline 2-stage OCR + KIE, Hybrid và VLM structured JSON bằng biến môi trường hoặc query `mode`.
- **Human-in-the-loop**: giao diện split-screen có canvas bounding boxes, preview các bước xử lý, chỉnh sửa header/line items và lưu lại hóa đơn.
- **Validation**: chuẩn hóa số tiền/ngày giờ tiếng Việt (cả dạng "Ngày 16 tháng 6 năm 2025"), kiểm tra `Sum(items) + tax - discount = total`, cảnh báo confidence thấp và trạng thái `NEEDS_REVIEW`.
- **Backend**: FastAPI, SQLAlchemy, SQLite mặc định; có thể chuyển sang MySQL/PostgreSQL bằng `IDP_DATABASE_URL`.
- **Evaluation**: CER, WER, Entity Precision/Recall/F1 trên JSONL, kèm script chạy pipeline thật trên ảnh và so với ground truth.

> Chế độ sidecar text/mock chỉ phục vụ demo offline. Khi báo cáo đồ án, hãy dùng model và tập test có ground truth để đo accuracy thực tế — xem mục Đánh giá bên dưới.

## Cấu trúc

```text
idp-invoice-extraction/
├── backend/app/
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models/
│   ├── schemas/
│   ├── api/routes.py
│   └── idp_engine/
├── backend/tests/
├── evaluation/evaluate_metrics.py
├── frontend/{index.html,app.js,styles.css}
├── notebooks/train_finetune_guide.ipynb
├── data/uploads/       # ảnh upload, tạo tự động
├── data/processed/     # ảnh sau preprocessing, tạo tự động
├── requirements.txt
└── RUN_DEMO.md
```

## Chạy nhanh trên Windows

```powershell
cd C:\Users\ROG\idp-invoice-extraction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000/>. API docs ở <http://127.0.0.1:8000/docs>.

Nếu chạy từ thư mục `backend`, dùng `python -m uvicorn app.main:app --reload`; README này ưu tiên chạy từ root để các đường dẫn mặc định nhất quán.

### Bật OCR GPU (PaddleOCR + CUDA)

```powershell
python scripts\check_gpu_env.py              # xác minh Python/Paddle/CUDA/NVIDIA driver
pip uninstall -y opencv-python-headless      # paddleocr cần opencv-contrib-python
pip install -r requirements-gpu.txt          # paddlepaddle-gpu 3.4.0 (cu129) + paddleocr 3.7.0
copy .env.example .env                       # rồi đặt IDP_USE_PADDLE=true trong .env
```

### Bật Hybrid VLM cục bộ (tùy chọn, P4 — venv riêng)

```powershell
py -3.12 -m venv C:\Users\khoas\idp-invoice-extraction-main\.venv-vlm   # venv riêng: paddle và torch không chung process được
C:\Users\khoas\idp-invoice-extraction-main\.venv-vlm\Scripts\Activate.ps1
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # Blackwell sm_120
pip install -r requirements-vlm.txt          # transformers + accelerate + bitsandbytes (4-bit)
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct   # ~7 GB, chạy 1 lần, cache dùng chung
# trong .env: IDP_VLM_PROVIDER=local + IDP_VLM_WORKER_PYTHON=<đường dẫn python của .venv-vlm>
```

Chi tiết cấu hình, VRAM và benchmark xem [RUN_DEMO.md](RUN_DEMO.md).

## Demo không cần OCR model

Tạo một ảnh bất kỳ tên `invoice.jpg` và file `invoice.txt` cạnh nó:

```text
CỬA HÀNG MINH AN
Địa chỉ: 123 Nguyễn Huệ, Quận 1, TP.HCM
Số hóa đơn: HD-2026-001
Ngày: 22/10/2026 14:30
Cà phê sữa 2 30.000 60.000
Bánh mì 1 25.000 25.000
Thuế VAT: 8.500
Tổng cộng: 93.500
```

Upload ảnh trong UI. Khi PaddleOCR chưa cài hoặc không load được model, detector sẽ đọc sidecar `.txt`, nhưng vẫn chạy toàn bộ KIE, chuẩn hóa và validation.

## OCR/VLM tùy chọn

- OCR thật: cài theo mục "Bật OCR GPU" ở trên rồi đặt `IDP_USE_PADDLE=true`. Bỏ cài hoặc đặt `false` sẽ về chế độ sidecar demo. Khi PaddleOCR bật, sidecar/mock bị tắt (`IDP_OCR_STRICT=true` là mặc định): engine lỗi sẽ trả lỗi rõ ràng thay vì bịa kết quả.
- VietOCR: cài thêm `vietocr` và `torch` (GPU: build cu128), bật `IDP_USE_VIETOCR=true`; dùng PaddleOCR để detect và VietOCR để nhận diện crop.
- VLM: đặt `IDP_PIPELINE_MODE=vlm`, `IDP_VLM_PROVIDER=openai` hoặc `gemini`, cùng API key. Adapter yêu cầu structured JSON theo schema trong `backend/app/schemas/invoice.py`. Provider `local` (Qwen2.5-VL, cài theo mục Hybrid) cũng dùng được cho chế độ này.
- Hybrid: đặt `IDP_PIPELINE_MODE=hybrid` (hoặc chọn trong UI) — OCR+KIE vẫn chạy trước, VLM cục bộ chỉ làm fallback theo quality gate; xem RUN_DEMO.md mục Hybrid.

Các dependency model nặng không nằm trong requirements mặc định để demo cài đặt nhanh và ổn định trên Windows.

## API chính

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/ocr/status` | Trạng thái engine OCR (backend, thiết bị GPU/CPU) + sẵn sàng của VLM fallback |
| `POST` | `/api/preprocess/preview` | Xem các ảnh trung gian preprocessing |
| `POST` | `/api/extract` | Upload và rút trích hóa đơn (`mode=heuristic/ocr_kie/hybrid/vlm`) |
| `GET/POST` | `/api/invoices` | Liệt kê/tạo hóa đơn |
| `GET/PUT/DELETE` | `/api/invoices/{id}` | Xem/sửa/xóa hóa đơn |
| `GET` | `/api/stats` | KPI và thống kê chi tiêu |
| `GET` | `/api/export?format=csv` | Xuất CSV hoặc XLSX |

## Scripts

| Script | Mục đích |
|---|---|
| `scripts/check_gpu_env.py` | Kiểm tra chuỗi Python → Paddle → CUDA → GPU trước khi cài |
| `scripts/benchmark_ocr.py` | Đo thời gian OCR trên ảnh thật, xuất JSON vào `evaluation/benchmarks/` |
| `scripts/evaluate_pipeline.py` | Chạy pipeline thật trên ground truth, tính CER/WER/Entity F1 |

## Evaluation

Mỗi dòng JSONL có dạng:

```json
{"reference":{"text":"...","entities":{"TOTAL_COST":93500}},"prediction":{"text":"...","entities":{"TOTAL_COST":93500}}}
```

Chạy:

```powershell
python evaluation\evaluate_metrics.py path\to\predictions.jsonl
```

## Kiểm thử

```powershell
pytest -q
```

Chi tiết cấu hình, preprocessing và cách lấy model xem [RUN_DEMO.md](RUN_DEMO.md).
