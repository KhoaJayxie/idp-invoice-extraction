# PROGRESS — Vietnamese Invoice IDP V2

> Nhật ký tiến độ cho đồ án. Cập nhật lần cuối: **08/11/2026**.
> Mục đích: lần sau mở lại dự án là nắm ngay được đã làm gì, sửa ở đâu, số liệu ra sao và việc gì còn lại.

## 0. Cập nhật 08/11/2026 — Spatial KIE viết lại hoàn toàn (mục chính của đợt này)

**Kết quả headline: entity accuracy trên bộ fixture OCR thật tăng từ 48.2% (27/56 trường) → 100% (56/56). 41/41 test pass (17 test KIE mới).**

### `backend/app/idp_engine/kie_extractor.py` — viết lại từ regex-line sang spatial KIE
- **Nhóm tokens thành dòng theo hình học** (`_build_lines`): y-overlap ≥45% gộp dòng; **tách segment theo khoảng trống x** (≥ max(0.13×page_width, 110px)) → hóa đơn 2 cột (buyer/bank, phone/account) không còn lẫn nhau.
- **Bảng hàng hóa theo cột** (`_detect_table`/`_parse_items`): header suy ra x-range từng cột (desc/qty/price/amount/disc) từ keyword; gán số cho cột; **toán học phân biệt** qty×price≈amount (tolerance max(0.05, 2%)); đơn giá wrap dòng dưới được gập lại (`merged` block numeric-only); còn thiếu cột qty (`SÓ GIÒ` hỏng) → số nguyên 1-3 chữ số đứng cuối mô tả được hiểu là SL. Item-code ≥1e8 (barcode/phone) bị chặn.
- **Trường bên (Seller/Buyer/Address/Tax ID/Invoice ID/Date/Currency)**: label mờ hữu hạn (rapidfuzz cửa sổ 1-4 từ, ≥80, diff độ dài ≤1 nếu ngắn) + 3 phương thức (inline/right/below theo x-overlap ≥0.35); nhãn kề bên là label khác thì bị **từ chối làm value** (`_classify_line` guard — sửa lỗi buyer='THANH TOÁN CHO'); run address neo theo **segment chứa value** (không phải segment rộng nhất) + **chặn ở header bảng** + lọc theo cột của party → address 2 cột sạch.
- **Seller**: label → header inference (business hints + caps + ngoặc, dòng đầu tiên mạnh thắng; giữ dòng tiếp không chứa số).
- **Money field**: loại dòng subtotal/tax/discount lẫn nhau (`exclude_labels`), loại tiền mặt/thối/changes (`_MONEY_LINE_EXCLUDE`), **ưu tiên modifier Grand/Rounded**, chặn tại "GST summary"; subtotal suy ra từ tổng dòng hàng khi thiếu. Tiền tệ biết theo dòng (RM/USD dot-decimal vs VND dot-thousands); match dấu AM/PM; ngày dài VN.
- **Timestamp 2 nút thắt đã sửa**: `_value_after_label` cắt ở dấu `:` đầu tiên (lấy nhầm `08:32:00`); nay tái dựng từ cả segment và gộp `Time:` khi không thấy ngày; gom thời gian từ segment kề/line kề. Thịt lỗi nữa: dòng `Date...` bị coi là **summary-start** vì 'cash' trong 'Cashier' — đã bỏ 'cashier' khỏi `_DATE_LINE`; `_first_summary_line` **bỏ qua dòng nhãn tax-id mà token trùng `_TAXID_VALUE`** (`GST No: 789417-W`, `MST: 031...`) nhưng giữ dòng tiền thật (`Thuế VAT: 8.500`).
- **Per-field confidence tách bạch**: `kie_conf = ocr_conf × method weight` (inline 1.0, right 0.95, below 0.85, computed 0.8, pattern_scan 0.7, business/address inference 0.45–0.55). Không đủ bằng chứng → `None` (không bịa). `field_details` kèm `value/ocr_conf/kie_conf/method/box_ids` cho mọi trường + items.
- `extract(tokens)` giữ wrapper 3-tuple; `extract_with_details` trả 4-tuple (data, boxes, raw_text, field_details).
- **Cho phép label chung**: Document No / Receipt No / Invoice No / Cash Bill No; nhãn EN+VN cho mọi trường.
- **Regression demo của user (ảnh thật)**: BOOK TA — trước seller='tan woon yann', id=None, items=0 → nay đủ 11 trường + 1 item; ZHENG HUI — trước seller='h02038', tax=0, items=0, total thiếu GST → nay đủ: seller/buyer/tax 000800689824/id CS00082258/address/timestamp **2018-02-09 08:32:00**/subtotal 411.5/tax 24.69/total 436.20 + **6 items**.

### Subway khác
- `postprocessor.py` — `currency_hint()` (VND/MYR/USD/EUR/GBP/SGD), `normalize_money(value, decimal_dot)` ép đọc dot theo tiền tệ; gập AM/PM 12h trong `normalize_datetime`; `validate_invoice` thêm `rounding_gap` (+0.01 RM) đề phòng Rounding Adjustment — giữ tolerance toán.
- `schemas/invoice.py` — thêm `buyer`, `buyer_address`, `tax_id`, `buyer_tax_id`, `currency` vào InvoiceData/InvoiceUpdate; `field_details` vào ExtractionResponse.
- `models/invoice.py` + `database.py` — migration 5 cột mới (nullable, additive).
- `api/routes.py` — `_to_out` map 5 trường mới.
- `pipeline.py` — gọi `extract_with_details`, trả `field_details`, normalize 5 trường mới. (OCR/FE/API behavior giữ nguyên.)
- **Frontend**: 5 input mới (MST người bán, khách hàng, địa chỉ/MST khách, tiền tệ) ở form Thông tin chung; `fillForm` + `saveInvoice` đã map.
- `scripts/evaluate_pipeline.py` — ENTITY_KEYS mở rộng thêm BUYER, BUYER_ADDRESS, TAX_ID, BUYER_TAX_ID, CURRENCY.
- **Mới**: `scripts/compare_kie.py` (so trước/sau), `evaluation/fixtures/kie_ground_truth.json` (GT chép tay cho 4 fixture), `evaluation/fixtures/after_kie_results.json`, `backend/tests/test_kie_extractor.py` (17 test: grouping/segment 2 cột, table columns + multi-line + wrapped price, desc-qty fallback, tax-id patterns, AM/PM, same-seg time fold, RM-vs-VND, Total-vs-GST, null-when-insufficient, field_details shape, regression 3 ảnh demo bằng tokens OCR thật).

### Bảng so trước/sau (4 fixture × 14 trường = 56 điểm mô tả trường)
| Field | bookta | zhenghui | vn_bookstore_1/2 |
|---|---|---|---|
| seller | ❌→✅ | ❌→✅ | ❌→✅ |
| buyer | ✅ | ❌→✅ | ❌→✅ |
| address | ❌→✅ | ❌→✅ | ✅ |
| buyer_address | ✅ | ✅ | ❌→✅ |
| tax_id | ❌→✅ | ❌→✅ | ✅ (null đúng) |
| invoice_id | ❌→✅ | ❌→✅ | ❌→✅ |
| currency | ❌→✅ | ❌→✅ | ❌→✅ |
| subtotal | ❌→✅ (computed) | ✅ | ✅ |
| tax | ✅ (0) | ❌→✅ 24.69 | ✅ |
| total_cost | ✅ | ❌→✅ 436.20 | ✅ |
| timestamp | ✅ | ❌→✅ 08:32:00 | ✅ |
| items | ❌→✅ (1) | ❌→✅ (6) | ❌→✅ (3) |

- **BEFORE 48.2% (27/56) → AFTER 100% (56/56)**. Script tái lập: `python scripts/compare_kie.py`.

### Hạn chế còn lại của KIE (trung thực để báo cáo)
1. Tên hàng VN vẫn mang **dấu hỏng do OCR** ("Tiu thuyt", "Đưng ABC") — đã dùng GT theo chữ in trên hóa đơn; độ chính xác 100% này là **KIE-level** (bind label→value đúng), không phải OCR-level (dấu/WRF theo evaluation/fixtures/invoice_ground_truth.jsonl vẫn CER 0.346).
2. Bundle 2 cột dạng bank/phone đã tách tốt bằng segment, nhưng header 2 cột cực ngắn (gap <110px) vẫn có thể gộp sai — không thường gặp với kích thước ảnh hiện tại.
3. GT hiện chỉ có **4 hóa đơn VN/EN** (2 ảnh demo SROIE + 1 hóa đơn sách VN). Số 100% chưa có giá trị thống kê rộng — cần bổ sung tập ảnh hóa đơn thật như đã dòng sẵn ở mục 6.
4. Bookta subtotal = **computed** (từ dòng hàng), vì hóa đơn chỉ in `TOTAL 9.00` — đúng vai trò của validation bên postprocessor khi đã có dòng hàng.

---

## 1. Trạng thái hiện tại (tóm tắt một câu)

Hệ thống chạy được **OCR thật bằng PaddleOCR trên GPU** (RTX 5060 Laptop 8GB, CUDA 12.9, PP-OCRv6 medium, tiếng Việt + tiếng Anh), end-to-end extract → KIE → validation → lưu DB hoạt động, **41/41 test pass**, UI đã sửa đủ 4 lỗi ưu tiên (bbox sai vị trí, UI overflow, ảnh quá nhỏ, lỗi API/data), **KIE spatial 100% entity accuracy trên 4 fixture OCR thật**.

## 2. Môi trường đã xác minh

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop 8GB | Blackwell, compute capability **12.0 (sm_120)** |
| Driver | 617.42, CUDA UMD 13.4 | Đủ cho CUDA 12.9 runtime |
| Python | 3.12.10 (venv `.venv`) | |
| paddlepaddle-gpu | **3.4.0** (cu129, CUDA 12.9 + cuDNN 9.9) | Wheel Windows cp312 duy nhất mới nhất; cu128 index chỉ có wheel Linux |
| paddleocr | **3.7.0** (paddlex 3.7.2) | |
| OpenCV | opencv-contrib-python 4.10.0.84 | Đã gỡ opencv-python-headless (xung đột 2 package cung cấp cv2) |
| numpy | 2.3.5 | paddlex yêu cầu <2.4, tự downgrade từ 2.5.4 |
| Model cache | `~\.paddlex\official_models\` | PP-OCRv6_medium_det/rec (134MB) + textline orientation; tải 1 lần, chạy offline từ lần sau |

Chuỗi kiểm tra môi trường: `python scripts\check_gpu_env.py` → tất cả PASS (đã chạy thật, GPU matmul OK).

## 3. Đã sửa / đã làm gì (chi tiết theo file)

### OCR thật + kiến trúc engine
- `backend/app/idp_engine/ocr_detector.py` — **viết lại adapter**:
  - PaddleOCR 3.x-first (signature introspection + passthrough kwargs `device/enable_mkldnn/cpu_threads`), vẫn còn nhánh tương thích 2.x.
  - Chọn thiết bị `auto|gpu|cpu`: `auto` probe GPU (matmul thật) rồi fallback CPU có cảnh báo; `gpu` bắt buộc thì fail-fast; **CPU tắt MKLDNN** (Paddle 3.4 bị bug OneDNN `ConvertPirAttribute2RuntimeAttribute` trên model PIR).
  - Pin model theo version: PP-OCRv6 dùng `tiny|small|medium`, PP-OCRv5 dùng `mobile|server`.
  - **Strict real mode**: `use_paddle=true` + `strict` → không đọc sidecar/mock nữa; engine lỗi raise `OCRUnavailableError` → API trả **503** kèm lý do. Lỗi suy luận cũng được ghi vào `backend_warning` thay vì nuốt im lặng.
  - Parse kết quả 3.x (`rec_texts/rec_scores/dt_polys`) + 2.x legacy; placeholder box dùng đúng kích thước ảnh (trước đây hardcode 800px).
  - Lock `_predict_lock` quanh predict (an toàn khi FastAPI threadpool gọi đồng thời).
- `backend/app/idp_engine/pipeline.py` — truyền settings mới vào detector, đo `timings` (preprocess/OCR/KIE/total ms), trả `engine` info, confidence tổng đổi **min → mean** (một token xấu không kéo chết cả hóa đơn).
- `backend/app/config.py` — settings mới: `ocr_strict`, `paddle_device`, `paddle_lang`, `paddle_ocr_version`, `paddle_model_size`, `paddle_rec_batch_size`, `paddle_det_limit_side_len`, `paddle_gpu_memory_fraction` (đọc từ `.env` với prefix `IDP_`).
- `backend/app/api/routes.py` — endpoint mới **`GET /api/ocr/status`**; `/api/extract` map `OCRUnavailableError → 503`; bắt thêm `OSError` + `Exception` chung (500 + dọn file upload) cho cả extract lẫn preview.
- `.env` (máy này) đã bật real mode: `IDP_USE_PADDLE=true`, device auto, PP-OCRv6 medium, strict. `.env.example` thêm đầy đủ khóa mới (mặc định `false` để an toàn khi clone).

### KIE (rút trích trường) — xử lý output OCR thật
- `backend/app/idp_engine/kie_extractor.py`:
  - **Tái dựng dòng** `_merge_into_lines()`: PP-OCR trả box theo cột (tên hàng | SL | đơn giá | thành tiền là 4 box riêng) — gộp theo chồng lấn trục y (≥45%) thành dòng hoàn chỉnh rồi mới chạy regex; overlay vẫn giữ box từng token.
  - **Fuzzy label matching** (rapidfuzz ≥80): OCR hay rơi chữ cái trên font nhỏ ("Tổng cộng" → "Tng cng", "Thuế" → "Thu") — so sánh cửa sổ 1-4 từ liền kề trên chuỗi đã bỏ dấu; đã tinh chỉnh ngưỡng để loại false positive "Huệ," vs "Thuế".
  - Item: bỏ ký tự tiền tệ dính vào số ("3.000.000đ", OCR hay đọc thành "d"); chấp nhận dòng 3-cột (tên + SL + thành tiền, đơn giá = amount/qty phải gần nguyên); chặn số điện thoại/tài khoản bị nhận nhầm item (qty ≤ 999, tiền ≥ 500đ, giá trị nguyên).
  - Nhãn "Tổng phụ" cho subtotal.
- `backend/app/idp_engine/postprocessor.py` — parse ngày dạng dài "Ngày 16 tháng 6 năm 2025" (kể cả không dấu); validation **ưu tiên tổng dòng hàng** khi "Tạm tính" trên hóa đơn bị OCR đọc sai (cảnh báo rõ ràng).
- `backend/app/idp_engine/preprocessor.py` — deskew bỏ qua góc ±90° (ảnh thẳng đứng được nhận là "không nghiêng", hết warning giả); tối ưu `_correct_illumination` tính background ở bản thu nhỏ 720px rồi upscale (700ms → ~290ms cho cả pipeline preprocess, chất lượng tương đương).

### Frontend (4 lỗi ưu tiên)
- `frontend/app.js`:
  - **BBox sai vị trí — đã sửa tận gốc**: thêm `imageContentRect()` tính đúng rect *nội dung hiển thị* của ảnh (mô hình contain/letterbox của `object-fit`) + `sourceDimensions()` lấy processed_width/height từ metadata; canvas phủ đúng vùng ảnh (không phải cả stage); hit-test tooltip dùng chung phép biến đổi → vẽ và bắt trỏ chuột không bao giờ lệch nhau.
  - `ResizeObserver` trên stage + ảnh: đổi cỡ cửa sổ/đổi variant/ảnh load chậm đều vẽ lại; gán `onload` trước khi set `src` và vẽ ngay nếu ảnh đã load (tránh mất overlay do cache).
  - `loadInvoice`: nếu thiếu ảnh aligned thì **ẩn box** thay vì vẽ sai vị trí.
  - **Lỗi lệch múi giờ khi lưu**: gửi chuỗi datetime-local nguyên vẹn thay vì `toISOString()` (trước đây 14:30 giờ VN bị lưu thành 07:30 UTC).
  - Chip trạng thái engine (`OCR: PaddleOCR · GPU (CUDA) · vi`) qua `/api/ocr/status`; khóa ô "Văn bản OCR demo" khi strict real mode; metrics hiển thị thêm OCR ms / tổng ms / thiết bị.
- `frontend/styles.css` — **ảnh receipt lớn hơn**: `max-height: min(72vh, 920px)` (trước 64vh/700px), stage/drop-zone min-height 520px, cột trái 1.15fr; **chống overflow**: `flex-wrap` cho image-toolbar/filter-row, `overflow:hidden` cho chart-track, media query 1050/820/560px cập nhật; footer căn giữa cùng 1440px với topbar; style chip engine.
- Đã **verify bằng browser thật** (Playwright + Chrome headless): upload ảnh thật → Rút trích → canvas trùng pixel với ảnh (left 257=257, width 488.6=488.6), 39 box đè đúng từng chữ trên ảnh, không tràn ngang (scrollWidth=clientWidth=1600), form điền đúng tổng 12.000.000, status "✓ Đã xác nhận", chip "PaddleOCR · GPU (CUDA)".

### API/data handling
- `backend/app/schemas/invoice.py` — bỏ field `aligned_image_url` **khai báo 2 lần** trong `ExtractionResponse`; thêm `timings`, `engine`; `InvoiceUpdate` thêm `field_confidence`.
- `backend/app/api/routes.py` — PUT invoice cập nhật được `field_confidence` trong raw_json (trước đây bị kẹt giá trị cũ).

### Test
- `backend/tests/test_postprocessing.py` — +9 test mới: gộp dòng theo cột, fuzzy label ("Tng cng" → total), tiền tệ dính số (đ và d), chặn phone-as-item, item 3-cột + tên wrap ngoặc, sidecar box phân bố full chiều cao, strict raise `OCRUnavailableError`, ngày dạng dài, validation ưu tiên tổng dòng hàng.
- `backend/tests/test_api.py` — +2 test: extract trả 503 khi engine chết ở strict mode; bất biến "mọi box nằm trong processed_width/height".
- `backend/tests/conftest.py` — ghim `use_paddle=false` trong test session (suite phải deterministic kể cả khi máy dev bật real mode trong `.env`).

### Scripts / docs / evaluation
- `scripts/check_gpu_env.py` (mới) — kiểm tra Python → Paddle → CUDA → GPU → PaddleOCR → model cache, exit code 0 = sẵn sàng GPU.
- `scripts/benchmark_ocr.py` (mới) — đo cold/warm preprocess + OCR trên ảnh thật, xuất JSON vào `evaluation/benchmarks/`.
- `scripts/evaluate_pipeline.py` (mới) — chạy pipeline thật trên ground truth JSONL → tính CER/WER/Entity F1, lưu predictions.
- `requirements-gpu.txt` (mới) — stack GPU pin phiên, đúng thứ tự cài.
- `evaluation/fixtures/invoice_ground_truth.jsonl` (mới) — ground truth cho 2 ảnh hóa đơn mẫu.
- `README.md`, `RUN_DEMO.md` — cập nhật hướng dẫn cài GPU, bảng benchmark, endpoint mới, scripts.

## 4. Số liệu benchmark (đo thật 07/11/2026)

Ảnh: hóa đơn mẫu 1131×1600 (webp). PP-OCRv6 medium, det_limit 960, rec_batch 16, VRAM fraction 0.5.

| Metric | GPU (RTX 5060 Laptop) | CPU fallback |
|---|---|---|
| Preprocess | ~290-350 ms | ~325 ms |
| **OCR (warm, median)** | **~250-300 ms** | ~11.900-12.000 ms |
| KIE + validation | ~25 ms | ~25 ms |
| Tổng (cold, gồm warm-up) | 0.65-1.9 s | ~14.6 s |
| Tokens / avg confidence | 39 / 0.971 | 39 / 0.971 (giống hệt) |
| VRAM chiếm dụng | **~587 MB** reserved / 220 MB allocated | — |

→ GPU nhanh hơn CPU **~40-48 lần** trên cùng model. Cold start đầu tiên sau khi cài (tải model + compile kernel) ~15s; các lần sau trong phiên đã ở mức bảng trên.

**Evaluation đầu-cuối** (2 ảnh, ground truth 5 trường): CER 0.346 (0.302 nếu bỏ dấu), WER 0.605, **Entity F1 0.60** (P=R=0.60) — khớp hoàn hảo TOTAL_COST/SUBTOTAL/TIMESTAMP; SELLER sai do model mất dấu tiếng Việt trên font đậm nghệ thuật ("NHÀ XUÃT BÁN HUYÊN THOAI"); TAX=0 không xuất trong prediction (hệ thống bỏ giá trị 0). Trạng thái hai ảnh: **VALIDATED, confidence 0.971, tổng 12.000.000đ khớp công thức**.

## 5. Hạn chế đã biết (nếu thầy/cô hỏi)

1. **Dấu tiếng Việt**: PP-OCRv6 medium đọc tốt phần lớn chữ nhưng vẫn lỗi dấu trên font đậm/nghệ thuật và font nhỏ (giữ nguyên âm đôi khi sai "Ấ"→"Ã", nhỏ thường mất dấu hẳn). KIE đã chống chịu bằng fuzzy matching; muốn chất lượng text cao hơn có thể thử `IDP_PADDLE_MODEL_SIZE` lớn hơn khi có model lớn hơn, hoặc fine-tune (notebook có skeleton).
2. **Hóa đơn chụp lệch/cong**: hóa đơn mẫu là scan phẳng. Preprocessing có perspective/deskew nhưng chưa test rộng trên ảnh chụp nghiêng thật — nên bổ sung tập ảnh chụp điện thoại khi làm demo.
3. ~~**2 cột trái/phải trên hóa đơn bị gộp thành 1 dòng**~~ — **đã xử lý 08/11**: segment tách theo khoảng trống x (≥110px) + address run neo theo cột của party; chỉ còn rủi ro khi 2 cột cách nhau <110px (hiếm ở độ phân giải demo).
4. **VietOCR/torch chưa cài** (tùy chọn, tốn ~2.5GB); hướng dẫn trong `requirements-gpu.txt` nếu muốn bật.

## 6. Việc tiếp theo gợi ý (làm tiếp khi mở dự án)

- [ ] Thu thập 10-20 ảnh hóa đơn chụp thật (điện thoại, nhiều góc/ánh sáng) → mở rộng `invoice_ground_truth.jsonl` → chạy lại `scripts/evaluate_pipeline.py` cho số liệu báo cáo thuyết phục hơn.
- [ ] (Tùy chọn) Cài torch cu128 + vietocr để so sánh nhận diện dòng tiếng Việt với PP-OCRv6 rec.
- [ ] Fine-tune theo `notebooks/train_finetune_guide.ipynb` nếu muốn tăng CER trên domain hóa đơn VN.
- [ ] Nếu gặp lỗi cài lại từ đầu: làm đúng thứ tự trong `requirements-gpu.txt` (uninstall opencv-headless trước, cài paddle-gpu trước paddleocr).
- [ ] DB demo (`data/idp_invoice.db`) đang trống — demo UI lưu thử 1-2 hóa đơn để có dữ liệu dashboard.
