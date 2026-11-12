(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    file: null,
    localUrl: null,
    response: null,
    previewMetadata: null,
    previewSources: { original: null, enhanced: null, binary: null },
    currentVariant: "original",
    selectedBox: -1,
    invoiceId: null,
    boxes: [],
    items: [],
    stats: null,
  };

  const money = (value) => new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 2 }).format(Number(value || 0));
  const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
  const formatDate = (value) => {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
  };
  const showToast = (message, error = false) => {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.toggle("error", error);
    toast.hidden = false;
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => { toast.hidden = true; }, 3600);
  };

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    let body = null;
    try { body = await response.json(); } catch (_) { /* non-JSON errors are handled below */ }
    if (!response.ok) {
      const detail = body && (body.detail || body.message);
      throw new Error(detail || `API error ${response.status}`);
    }
    return body;
  }

  function setApiStatus(online, label) {
    $("api-dot").classList.toggle("online", online);
    $("api-label").textContent = label;
  }

  async function checkHealth() {
    try {
      const health = await api("/health");
      setApiStatus(true, `API sẵn sàng · ${health.pipeline_mode}`);
    } catch (_) {
      setApiStatus(false, "Không kết nối được API");
    }
    try {
      const status = await api("/api/ocr/status");
      renderOcrStatus(status);
    } catch (_) { /* status chip stays neutral */ }
  }

  function renderOcrStatus(status) {
    const chip = $("ocr-engine");
    if (!chip) return;
    const engine = status.engine || {};
    if (status.use_paddle && engine.backend === "paddleocr") {
      const device = engine.device === "gpu" ? "GPU (CUDA)" : "CPU";
      chip.textContent = `OCR: PaddleOCR · ${device} · ${engine.lang || "vi"}`;
      chip.className = "ocr-engine-chip online";
    } else if (status.use_paddle) {
      chip.textContent = "OCR: đang khởi tạo model…";
      chip.className = "ocr-engine-chip pending";
    } else {
      chip.textContent = status.strict ? "OCR: sidecar demo (real mode tắt)" : "OCR: sidecar demo";
      chip.className = "ocr-engine-chip offline";
    }
    if (status.warning) chip.title = status.warning;
    renderVlmStatus(status);
    // Real mode never reads sidecar text — say so instead of accepting input
    // that would silently be ignored.
    const sidecar = document.querySelector(".sidecar-settings");
    if (sidecar) {
      if (status.use_paddle && status.strict) {
        sidecar.setAttribute("disabled", "disabled");
        sidecar.setAttribute("title", "Real mode (IDP_USE_PADDLE=true) không dùng văn bản demo sidecar.");
      } else {
        sidecar.removeAttribute("disabled");
        sidecar.removeAttribute("title");
      }
    }
  }

  function renderVlmStatus(status) {
    const chip = $("vlm-status");
    if (!chip) return;
    chip.hidden = false;
    if (status.vlm_available) {
      const model = (status.vlm_provider === "local" ? (status.vlm_local_model || "").split("/").pop() : status.vlm_provider) || "VLM";
      chip.textContent = `VLM dự phòng: sẵn sàng · ${model}`;
      chip.title = "Hybrid mode sẽ gọi VLM này khi OCR+KIE thiếu dữ liệu hoặc validation lỗi.";
    } else {
      chip.textContent = "VLM dự phòng: chưa cấu hình";
      chip.title = status.vlm_unavailable_reason || "Đặt IDP_VLM_PROVIDER=local và tải model để bật.";
    }
  }

  function setupTheme() {
    const saved = localStorage.getItem("idp-theme");
    if (saved === "dark" || saved === "light") document.documentElement.dataset.theme = saved;
    $("theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("idp-theme", next);
    });
  }

  function setupFileInputs() {
    $("file-input").addEventListener("change", (event) => acceptFile(event.target.files[0]));
    $("drop-input").addEventListener("change", (event) => acceptFile(event.target.files[0]));
    const zone = $("drop-zone");
    zone.addEventListener("click", (event) => {
      if (event.target.closest("button, input, a")) return;
      $("drop-input").click();
    });
    zone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("drop-input").click(); }
    });
    ["dragenter", "dragover"].forEach((eventName) => zone.addEventListener(eventName, (event) => {
      event.preventDefault(); zone.classList.add("dragging");
    }));
    ["dragleave", "drop"].forEach((eventName) => zone.addEventListener(eventName, (event) => {
      event.preventDefault(); zone.classList.remove("dragging");
    }));
    zone.addEventListener("drop", (event) => acceptFile(event.dataTransfer.files[0]));
    $("preview-button").addEventListener("click", previewImage);
    document.querySelectorAll(".segment").forEach((button) => button.addEventListener("click", () => selectVariant(button.dataset.variant)));
    $("image-stage").addEventListener("mousemove", showBoxAtPointer);
    $("image-stage").addEventListener("mouseleave", () => { $("box-tooltip").hidden = true; });
    window.addEventListener("resize", drawBoxes);
    // Redraw whenever the stage or image box changes size (panel reflow,
    // details toggles, font loading) so the overlay never drifts.
    if (window.ResizeObserver) {
      state.resizeObserver = new ResizeObserver(() => drawBoxes());
      state.resizeObserver.observe($("image-stage"));
      state.resizeObserver.observe($("receipt-image"));
    }
  }

  function acceptFile(file) {
    if (!file) return;
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"];
    if (!allowed.includes(file.type)) { showToast("Vui lòng chọn file ảnh JPG, PNG, WEBP, BMP hoặc TIFF.", true); return; }
    if (file.size > 12 * 1024 * 1024) { showToast("Ảnh vượt giới hạn 12 MB.", true); return; }
    state.file = file;
    resetForm();
    state.response = null;
    state.previewMetadata = null;
    state.boxes = [];
    state.selectedBox = -1;
    if (state.localUrl) URL.revokeObjectURL(state.localUrl);
    state.localUrl = URL.createObjectURL(file);
    state.previewSources = { original: state.localUrl, enhanced: null, binary: null };
    $("empty-upload").hidden = true;
    $("image-stage").hidden = false;
    const image = $("receipt-image");
    image.onload = drawBoxes;
    image.src = state.localUrl;
    $("review-status").textContent = "Sẵn sàng rút trích";
    $("review-status").className = "status-pill status-neutral";
    $("box-count").textContent = "0 vùng";
    $("image-metrics").replaceChildren(textNode("Ảnh đã chọn · nhấn “Rút trích” để bắt đầu."));
    renderOCRList();
    showToast(`Đã chọn ${file.name}`);
  }

  function textNode(value) { return document.createTextNode(value); }

  function preprocessOptions() {
    const result = {};
    document.querySelectorAll(".preprocess-settings input[type=checkbox]").forEach((input) => { result[input.name] = input.checked; });
    return result;
  }

  async function previewImage() {
    if (!state.file) { showToast("Hãy chọn ảnh trước.", true); return; }
    const button = $("preview-button");
    button.disabled = true;
    button.textContent = "Đang xử lý…";
    try {
      const form = new FormData();
      form.append("file", state.file);
      form.append("options", JSON.stringify(preprocessOptions()));
      const result = await api("/api/preprocess/preview", { method: "POST", body: form });
      state.previewSources.original = result.original;
      state.previewSources.enhanced = result.enhanced;
      state.previewSources.binary = result.binary;
      state.previewMetadata = result.metadata;
      updateMetrics(result.metadata);
      selectVariant(state.currentVariant);
      showToast("Đã tạo preview preprocessing.");
    } catch (error) { showToast(error.message, true); }
    finally { button.disabled = false; button.textContent = "Xem xử lý ảnh"; }
  }

  async function extractInvoice() {
    if (!state.file) { showToast("Hãy chọn ảnh trước.", true); return; }
    const button = $("extract-button");
    if (button) button.disabled = true;
    const form = new FormData();
    form.append("file", state.file);
    form.append("mode", $("pipeline-mode").value);
    form.append("preprocess_options", JSON.stringify(preprocessOptions()));
    const demoText = $("ocr-text");
    if (demoText && demoText.value.trim()) form.append("ocr_text", demoText.value.trim());
    try {
      const result = await api("/api/extract", { method: "POST", body: form });
      state.response = result;
      state.previewMetadata = result.preprocess;
      state.boxes = result.boxes || [];
      state.invoiceId = null;
      state.previewSources = {
        // aligned is the geometric image used for OCR, so boxes remain exact.
        original: result.aligned_image_url ? withApiBase(result.aligned_image_url) : state.localUrl,
        enhanced: result.processed_image_url ? withApiBase(result.processed_image_url) : null,
        binary: result.binary_image_url ? withApiBase(result.binary_image_url) : null,
      };
      fillForm(result.invoice, result.status);
      updateConfidence(result.confidence);
      updateMetrics(result.preprocess);
      renderWarnings(result.warnings || []);
      renderProvenance(result);
      renderOCRList();
      selectVariant("original");
      updateReviewStatus(result.status, result.confidence);
      showToast("Đã hoàn tất rút trích. Hãy kiểm tra các trường trước khi lưu.");
    } catch (error) { showToast(error.message, true); }
    finally { if (button) button.disabled = false; }
  }

  function withApiBase(path) { return path.startsWith("http") ? path : path; }

  function selectVariant(variant) {
    if (!state.previewSources[variant]) {
      if (variant !== "original") showToast("Chưa có ảnh variant này; nhấn Xem xử lý ảnh.", true);
      return;
    }
    state.currentVariant = variant;
    const image = $("receipt-image");
    image.onload = drawBoxes;
    image.onerror = () => { /* keep the previous frame visible */ };
    image.src = state.previewSources[variant];
    document.querySelectorAll(".segment").forEach((button) => {
      const active = button.dataset.variant === variant;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (image.complete && image.naturalWidth) drawBoxes();
  }

  // Geometry helper: rect of the *displayed image content* inside the stage.
  // The <img> can be letterboxed by object-fit when we stretch it, so the
  // element rect is not always the content rect; boxes must map onto the
  // content, otherwise they drift off the text.
  function imageContentRect() {
    const image = $("receipt-image");
    const stageRect = $("image-stage").getBoundingClientRect();
    if (!image.complete || !image.naturalWidth) return null;
    const rect = image.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return null;
    const containScale = Math.min(rect.width / image.naturalWidth, rect.height / image.naturalHeight);
    const contentWidth = image.naturalWidth * containScale;
    const contentHeight = image.naturalHeight * containScale;
    return {
      stageRect,
      left: rect.left + (rect.width - contentWidth) / 2 - stageRect.left,
      top: rect.top + (rect.height - contentHeight) / 2 - stageRect.top,
      width: contentWidth,
      height: contentHeight,
    };
  }

  function sourceDimensions() {
    // Boxes are stored in processed-image pixels; the displayed aligned/enhanced/
    // binary artifacts share those dimensions exactly.
    const image = $("receipt-image");
    const metadata = state.previewMetadata || (state.response && state.response.preprocess);
    return {
      width: (metadata && metadata.processed_width) || image.naturalWidth,
      height: (metadata && metadata.processed_height) || image.naturalHeight,
    };
  }

  function drawBoxes() {
    const canvas = $("box-canvas");
    const frame = imageContentRect();
    if (!frame) { canvas.width = canvas.width; return; }
    canvas.style.left = `${frame.left}px`;
    canvas.style.top = `${frame.top}px`;
    canvas.style.width = `${frame.width}px`;
    canvas.style.height = `${frame.height}px`;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(frame.width * dpr));
    canvas.height = Math.max(1, Math.round(frame.height * dpr));
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, frame.width, frame.height);
    const source = sourceDimensions();
    const sx = frame.width / source.width;
    const sy = frame.height / source.height;
    const palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
    state.boxes.forEach((item, index) => {
      if (!Array.isArray(item.box) || item.box.length < 4) return;
      const x = Number(item.box[0]) * sx;
      const y = Number(item.box[1]) * sy;
      const width = Math.max(2, (Number(item.box[2]) - Number(item.box[0])) * sx);
      const height = Math.max(2, (Number(item.box[3]) - Number(item.box[1])) * sy);
      const color = palette[index % palette.length];
      context.globalAlpha = index === state.selectedBox ? 0.28 : 0.12;
      context.fillStyle = color;
      context.fillRect(x, y, width, height);
      context.globalAlpha = index === state.selectedBox ? 1 : .78;
      context.strokeStyle = color;
      context.lineWidth = index === state.selectedBox ? 2.5 : 1.5;
      context.strokeRect(x, y, width, height);
    });
    context.globalAlpha = 1;
  }

  function boxIndexAtEvent(event) {
    const frame = imageContentRect();
    if (!frame) return -1;
    const source = sourceDimensions();
    const x = (event.clientX - frame.left - frame.stageRect.left) * source.width / frame.width;
    const y = (event.clientY - frame.top - frame.stageRect.top) * source.height / frame.height;
    for (let index = state.boxes.length - 1; index >= 0; index -= 1) {
      const box = state.boxes[index].box;
      if (x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3]) return index;
    }
    return -1;
  }

  function showBoxAtPointer(event) {
    const index = boxIndexAtEvent(event);
    if (index < 0) { $("box-tooltip").hidden = true; return; }
    state.selectedBox = index;
    drawBoxes();
    const item = state.boxes[index];
    const tooltip = $("box-tooltip");
    tooltip.textContent = `${item.label || "OCR"} · ${item.text || "(trống)"} · ${percent(item.confidence)}`;
    const stageRect = $("image-stage").getBoundingClientRect();
    const left = Math.max(8, Math.min(event.clientX - stageRect.left + 12, stageRect.width - 220));
    const top = Math.max(8, event.clientY - stageRect.top - 42);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.hidden = false;
    document.querySelectorAll(".ocr-token").forEach((node, nodeIndex) => node.classList.toggle("active", nodeIndex === index));
  }

  function renderOCRList() {
    const list = $("ocr-list");
    list.replaceChildren();
    $("box-count").textContent = `${state.boxes.length} vùng`;
    if (!state.boxes.length) { list.append(textNode("Các từ nhận diện sẽ hiện ở đây.")); list.className = "ocr-list empty-state"; return; }
    list.className = "ocr-list";
    state.boxes.forEach((item, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "ocr-token";
      button.addEventListener("click", () => { state.selectedBox = index; drawBoxes(); button.scrollIntoView({ block: "nearest" }); });
      const text = document.createElement("span"); text.className = "ocr-token-text"; text.textContent = item.text || "(trống)";
      const label = document.createElement("span"); label.className = "ocr-token-label"; label.textContent = item.label || "OCR";
      const score = document.createElement("span"); score.className = "ocr-token-score"; score.textContent = percent(item.confidence);
      button.append(text, label, score); list.append(button);
    });
  }

  function updateMetrics(metadata) {
    const target = $("image-metrics");
    target.replaceChildren();
    if (!metadata) { target.append(textNode("Chưa có thông tin xử lý ảnh.")); return; }
    const timings = state.response && state.response.timings;
    const metrics = [
      `Kích thước ${metadata.processed_width}×${metadata.processed_height}`,
      `Biên trang ${metadata.document_detected ? percent(metadata.document_confidence) : "không thấy"}`,
      `Deskew ${metadata.deskew_applied ? `${Number(metadata.deskew_angle).toFixed(1)}°` : "—"}`,
      `Độ nét ${Number(metadata.blur_score || 0).toFixed(0)}`,
      `Tương phản ${Number(metadata.contrast_std || 0).toFixed(1)}`,
    ];
    if (timings && timings.ocr_ms !== undefined) {
      metrics.push(`OCR ${Number(timings.ocr_ms).toFixed(0)} ms`, `Tổng ${Number(timings.total_ms).toFixed(0)} ms`);
      if (state.response.engine && state.response.engine.device) metrics.push(`Thiết bị ${state.response.engine.device === "gpu" ? "GPU" : "CPU"}`);
    }
    metrics.forEach((value) => { const chip = document.createElement("span"); chip.className = "metric-chip"; chip.textContent = value; target.append(chip); });
    (metadata.warnings || []).slice(0, 3).forEach((value) => { const chip = document.createElement("span"); chip.className = "metric-chip metric-warning"; chip.textContent = `⚠ ${value}`; target.append(chip); });
  }

  function updateConfidence(value) {
    const normalized = Math.max(0, Math.min(1, Number(value || 0)));
    $("confidence-bar").style.width = `${normalized * 100}%`;
    $("confidence-bar").style.background = normalized >= .85 ? "var(--status-good)" : normalized >= .6 ? "var(--status-warning)" : "var(--status-critical)";
    $("confidence-value").textContent = percent(normalized);
  }

  function updateReviewStatus(status, confidence) {
    const element = $("review-status");
    element.textContent = status === "VALIDATED" ? "✓ Đã xác nhận" : "! Cần kiểm tra";
    element.className = `status-pill ${status === "VALIDATED" ? "status-valid" : "status-review"}`;
    updateConfidence(confidence);
  }

  function fillForm(invoice, status) {
    ["seller", "invoice_id", "address", "tax_id", "buyer", "buyer_address", "buyer_tax_id", "currency", "subtotal", "tax", "discount", "total_cost"].forEach((key) => {
      $(key).value = invoice[key] === null || invoice[key] === undefined ? "" : invoice[key];
    });
    $("timestamp").value = invoice.timestamp ? String(invoice.timestamp).slice(0, 16) : "";
    $("status").value = status || "NEEDS_REVIEW";
    state.items = (invoice.items || []).map((item) => ({ ...item }));
    renderItems();
    updateMathCheck();
  }

  function addItem(item = {}) {
    state.items.push({ name: item.name || "", quantity: Number(item.quantity || 1), unit_price: Number(item.unit_price || 0), amount: Number(item.amount || 0), confidence: Number(item.confidence || 0) });
    renderItems();
  }

  function renderItems() {
    const body = $("items-body"); body.replaceChildren();
    $("items-empty").hidden = state.items.length > 0;
    state.items.forEach((item, index) => {
      const row = document.createElement("tr");
      const fields = [
        ["name", "text", item.name, "item-name"],
        ["quantity", "number", item.quantity, ""],
        ["unit_price", "number", item.unit_price, ""],
        ["amount", "number", item.amount, ""],
      ];
      fields.forEach(([key, type, value, extra]) => {
        const cell = document.createElement("td");
        const input = document.createElement("input"); input.className = `item-input ${extra}`; input.type = type; input.value = value ?? ""; input.step = "any"; input.setAttribute("aria-label", `${key} dòng ${index + 1}`);
        input.addEventListener("input", () => { state.items[index][key] = type === "number" ? Number(input.value || 0) : input.value; updateMathCheck(); });
        cell.append(input); row.append(cell);
      });
      const removeCell = document.createElement("td"); const remove = document.createElement("button"); remove.className = "remove-item"; remove.type = "button"; remove.textContent = "×"; remove.setAttribute("aria-label", `Xóa dòng ${index + 1}`); remove.addEventListener("click", () => { state.items.splice(index, 1); renderItems(); updateMathCheck(); }); removeCell.append(remove); row.append(removeCell); body.append(row);
    });
  }

  function updateMathCheck() {
    const lineSum = state.items.reduce((sum, item) => sum + (Number(item.amount) || (Number(item.quantity) || 0) * (Number(item.unit_price) || 0)), 0);
    const subtotal = state.items.length ? lineSum : Number($("subtotal").value || 0);
    const tax = Number($("tax").value || 0);
    const discount = Number($("discount").value || 0);
    const total = Number($("total_cost").value || 0);
    const expected = subtotal + tax - discount;
    const delta = total - expected;
    const target = $("math-check");
    target.className = `math-check ${total > 0 && Math.abs(delta) <= Math.max(1, Math.abs(total) * .01) ? "pass" : "fail"}`;
    target.textContent = `Hàng hóa ${money(lineSum)} + thuế ${money(tax)} − giảm ${money(discount)} = ${money(expected)} · tổng nhập ${money(total)} · lệch ${money(delta)}`;
  }

  function renderWarnings(warnings) {
    const target = $("warnings"); target.replaceChildren();
    warnings.forEach((message) => { const item = document.createElement("div"); item.className = "warning-item"; const mark = document.createElement("span"); mark.className = "warning-mark"; mark.textContent = "⚠"; const text = document.createElement("span"); text.textContent = message; item.append(mark, text); target.append(item); });
  }

  // Hybrid provenance: VLM activation, per-field sources, KIE↔VLM conflicts.
  function renderProvenance(result) {
    const target = $("provenance");
    if (!target) return;
    target.replaceChildren();
    const sources = result.sources || {};
    const conflicts = result.conflicts || [];
    const reasons = result.fallback_reasons || [];
    const chips = [];
    if (result.vlm_triggered) {
      chips.push(`VLM dự phòng: ĐÃ CHẠY${reasons.length ? ` — ${reasons.join("; ")}` : ""}`);
    } else if (reasons.length) {
      chips.push(`VLM dự phòng: không chạy (${reasons.join("; ")})`);
    }
    const vlmFields = Object.entries(sources).filter(([, source]) => source === "vlm").map(([field]) => field);
    const agreedFields = Object.entries(sources).filter(([, source]) => source === "kie+vlm").map(([field]) => field);
    if (vlmFields.length) chips.push(`Trường lấy từ VLM: ${vlmFields.join(", ")}`);
    if (agreedFields.length) chips.push(`KIE và VLM khớp nhau: ${agreedFields.join(", ")}`);
    conflicts.forEach((conflict) => {
      const state = conflict.resolved ? "đã xử lý" : "CẦN SOÁT";
      chips.push(`Bất đồng ${conflict.field} (${state}): KIE “${conflict.kie_value}” vs VLM “${conflict.vlm_value}” → giữ ${String(conflict.chosen_source).toUpperCase()}`);
    });
    if (!chips.length) return;
    chips.forEach((message) => {
      const item = document.createElement("div");
      item.className = "warning-item provenance-item";
      const mark = document.createElement("span");
      mark.className = "warning-mark";
      mark.textContent = result.vlm_triggered ? "⚡" : "🔗";
      const text = document.createElement("span");
      text.textContent = message;
      item.append(mark, text);
      target.append(item);
    });
  }

  async function saveInvoice(event) {
    event.preventDefault();
    const payload = {
      invoice_id: $("invoice_id").value.trim() || null,
      seller: $("seller").value.trim() || null,
      address: $("address").value.trim() || null,
      tax_id: $("tax_id").value.trim() || null,
      buyer: $("buyer").value.trim() || null,
      buyer_address: $("buyer_address").value.trim() || null,
      buyer_tax_id: $("buyer_tax_id").value.trim() || null,
      currency: $("currency").value.trim() || null,
      timestamp: $("timestamp").value || null,
      subtotal: Number($("subtotal").value || 0), tax: Number($("tax").value || 0), discount: Number($("discount").value || 0), total_cost: Number($("total_cost").value || 0),
      items: state.items.map((item) => ({ name: String(item.name || ""), quantity: Number(item.quantity || 0), unit_price: Number(item.unit_price || 0), amount: Number(item.amount || 0), confidence: Number(item.confidence || 0), bbox: item.bbox || null })),
      status: $("status").value,
      confidence: state.response ? Number(state.response.confidence || 0) : 0,
      strategy: state.response ? state.response.strategy : "manual-review",
      source_image: state.response ? state.response.source_image : null,
      aligned_image_url: state.response ? state.response.aligned_image_url : null,
      processed_image_url: state.response ? state.response.processed_image_url : null,
      binary_image_url: state.response ? state.response.binary_image_url : null,
      original_filename: state.response ? state.response.original_filename : null,
      raw_json: state.response ? {
        raw_text: state.response.raw_text || "",
        boxes: state.response.boxes || state.boxes || [],
        field_confidence: (state.response.invoice && state.response.invoice.field_confidence) || {},
        field_details: state.response.field_details || {},
        sources: state.response.sources || {},
        conflicts: state.response.conflicts || [],
        fallback_reasons: state.response.fallback_reasons || [],
        vlm_triggered: Boolean(state.response.vlm_triggered),
      } : null,
      preprocessing_json: state.response ? state.response.preprocess : null,
    };
    const button = $("save-button"); button.disabled = true;
    try {
      const result = await api(state.invoiceId ? `/api/invoices/${state.invoiceId}` : "/api/invoices", { method: state.invoiceId ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      state.invoiceId = result.id;
      showToast(`Đã lưu hóa đơn #${result.id}.`);
      await Promise.all([loadHistory(), loadDashboard()]);
    } catch (error) { showToast(error.message, true); }
    finally { button.disabled = false; }
  }

  function resetForm() {
    $("invoice-form").reset(); state.items = []; state.invoiceId = null; renderItems(); updateConfidence(0); renderWarnings([]); renderProvenance({}); updateMathCheck(); $("review-status").textContent = "Chưa có dữ liệu"; $("review-status").className = "status-pill status-neutral";
  }

  async function loadHistory() {
    try {
      const rows = await api("/api/invoices?limit=100");
      const target = $("history-list"); target.replaceChildren(); $("history-count").textContent = `(${rows.length})`;
      if (!rows.length) { target.append(textNode("Chưa có hóa đơn đã lưu.")); return; }
      rows.forEach((row) => {
        const button = document.createElement("button"); button.type = "button"; button.className = "history-row";
        const title = document.createElement("strong"); title.textContent = row.invoice_id || row.seller || `Hóa đơn #${row.id}`;
        const total = document.createElement("span"); total.textContent = money(row.total_cost);
        const status = document.createElement("span"); status.className = "history-status"; status.textContent = `${row.status === "VALIDATED" ? "✓ Đã xác nhận" : "! Cần kiểm tra"} · ${formatDate(row.created_at)}`;
        button.append(title, total, status); button.addEventListener("click", () => loadInvoice(row.id)); target.append(button);
      });
    } catch (error) { showToast(error.message, true); }
  }

  async function loadInvoice(id) {
    try {
      const row = await api(`/api/invoices/${id}`);
      state.invoiceId = row.id;
      state.boxes = (row.raw_json && row.raw_json.boxes) || [];
      state.previewMetadata = row.preprocessing_json || null;
      state.response = {
        confidence: row.confidence,
        strategy: row.strategy,
        raw_text: (row.raw_json || {}).raw_text || "",
        boxes: state.boxes,
        invoice: { ...row, field_confidence: row.field_confidence || {} },
        preprocess: row.preprocessing_json,
        source_image: row.source_image,
        original_filename: row.original_filename,
        aligned_image_url: row.aligned_image_url,
        processed_image_url: row.processed_image_url,
        binary_image_url: row.binary_image_url,
      };
      state.previewSources = {
        original: row.aligned_image_url || row.source_image_url || null,
        enhanced: row.processed_image_url || null,
        binary: row.binary_image_url || null,
      };
      fillForm(row, row.status);
      updateReviewStatus(row.status, row.confidence);
      renderWarnings([]);
      renderProvenance({
        vlm_triggered: (row.raw_json || {}).vlm_triggered,
        fallback_reasons: (row.raw_json || {}).fallback_reasons || [],
        sources: (row.raw_json || {}).sources || {},
        conflicts: (row.raw_json || {}).conflicts || [],
      });
      updateMetrics(row.preprocessing_json);
      // Boxes are stored in aligned-image coordinates; without the aligned
      // artifact they cannot be mapped onto the raw upload, so hide them
      // instead of drawing them at wrong positions.
      if (!row.aligned_image_url) {
        state.boxes = [];
        state.response.boxes = [];
      }
      renderOCRList();
      if (state.previewSources.original) {
        $("empty-upload").hidden = true;
        $("image-stage").hidden = false;
        selectVariant("original");
      } else {
        $("image-stage").hidden = true;
        $("empty-upload").hidden = false;
      }
      showToast(`Đang xem hóa đơn #${id}.`);
    } catch (error) { showToast(error.message, true); }
  }

  function renderDashboard(stats) {
    state.stats = stats;
    $("kpi-count").textContent = money(stats.invoice_count);
    $("kpi-spend").textContent = money(stats.total_spend);
    $("kpi-tax").textContent = money(stats.total_tax);
    $("kpi-review").textContent = money(stats.needs_review);
    $("kpi-validated").textContent = `${money(stats.validated)} hóa đơn hợp lệ`;
    $("dashboard-updated").textContent = `Cập nhật ${new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })}`;
    renderMonthly(stats.by_month || []); renderSellers(stats.by_seller || []);
  }

  function renderMonthly(rows) {
    const chart = $("monthly-chart"); const table = $("monthly-table"); chart.replaceChildren(); table.replaceChildren();
    if (!rows.length) { chart.append(textNode("Chưa có dữ liệu trong khoảng đã chọn.")); chart.className = "monthly-chart chart-empty"; const empty = document.createElement("tr"); const cell = document.createElement("td"); cell.colSpan = 3; cell.className = "empty-cell"; cell.textContent = "Chưa có dữ liệu"; empty.append(cell); table.append(empty); return; }
    chart.className = "monthly-chart";
    const visibleRows = rows.slice(-8);
    const maximum = Math.max(...visibleRows.map((row) => Number(row.total || 0)), 1);
    visibleRows.forEach((row, index) => {
      const wrapper = document.createElement("div"); wrapper.className = "chart-row";
      const period = document.createElement("span"); period.className = "chart-period"; period.textContent = row.month;
      const track = document.createElement("div"); track.className = "chart-track"; track.setAttribute("tabindex", "0"); track.setAttribute("aria-label", `${row.month}: ${money(row.total)} VND, ${row.count} hóa đơn`); track.title = `${row.month}: ${money(row.total)} VND`;
      const bar = document.createElement("span"); bar.className = "chart-bar"; bar.style.setProperty("--bar-width", `${Math.max(1.5, Number(row.total || 0) / maximum * 72)}%`); bar.setAttribute("aria-hidden", "true"); track.append(bar);
      if (index === visibleRows.length - 1) { const value = document.createElement("span"); value.className = "chart-value-end"; value.textContent = money(row.total); track.append(value); }
      wrapper.append(period, track); chart.append(wrapper);
      const tr = document.createElement("tr"); [row.month, row.count, money(row.total)].forEach((valueText) => { const td = document.createElement("td"); td.textContent = String(valueText); tr.append(td); }); table.append(tr);
    });
  }

  function renderSellers(rows) {
    const target = $("seller-list"); target.replaceChildren();
    if (!rows.length) { const empty = document.createElement("p"); empty.className = "empty-state"; empty.textContent = "Chưa có hóa đơn đã lưu."; target.append(empty); return; }
    const maximum = Math.max(...rows.map((row) => Number(row.total || 0)), 1);
    rows.slice(0, 6).forEach((row) => {
      const wrapper = document.createElement("div"); wrapper.className = "seller-row";
      const name = document.createElement("strong"); name.textContent = row.seller;
      const total = document.createElement("span"); total.textContent = money(row.total);
      const track = document.createElement("div"); track.className = "seller-track"; const fill = document.createElement("span"); fill.className = "seller-fill"; fill.style.setProperty("--fill", `${Number(row.total || 0) / maximum * 100}%`); track.append(fill); wrapper.append(name, total, track); target.append(wrapper);
    });
  }

  async function loadDashboard() {
    try { renderDashboard(await api(`/api/stats?days=${encodeURIComponent($("date-range").value)}`)); }
    catch (error) { showToast(error.message, true); }
  }

  function setupForm() {
    $("extract-button")?.addEventListener("click", extractInvoice);
    $("invoice-form").addEventListener("submit", saveInvoice);
    $("add-item").addEventListener("click", () => addItem());
    $("reset-button").addEventListener("click", resetForm);
    ["subtotal", "tax", "discount", "total_cost"].forEach((id) => $(id).addEventListener("input", updateMathCheck));
    $("date-range").addEventListener("change", loadDashboard);
    $("refresh-dashboard").addEventListener("click", loadDashboard);
    $("pipeline-mode").addEventListener("change", () => {
      const mode = $("pipeline-mode").value;
      if (mode === "vlm") showToast("VLM cần cấu hình API key trong .env (hoặc provider local).");
      if (mode === "hybrid") showToast("Hybrid: chạy OCR+KIE trước; VLM cục bộ chỉ chạy khi thiếu dữ liệu/lỗi validation.");
    });
  }

  // The action is placed beside the file controls only after the DOM exists.
  function addExtractButton() {
    const toolbar = document.querySelector(".image-toolbar");
    if (!toolbar || $("extract-button")) return;
    const button = document.createElement("button"); button.id = "extract-button"; button.type = "button"; button.className = "button button-primary button-small"; button.textContent = "Rút trích dữ liệu"; button.addEventListener("click", extractInvoice); toolbar.append(button);
  }

  setupTheme(); setupFileInputs(); setupForm(); addExtractButton(); checkHealth(); loadHistory(); loadDashboard(); renderItems(); updateMathCheck();
})();
