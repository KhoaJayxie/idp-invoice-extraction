"""Local VLM adapter: Qwen2.5-VL via transformers + bitsandbytes 4-bit.

Runs entirely on this machine (Windows 11, Python 3.12, RTX 5060 Laptop 8 GB)
so the Hybrid pipeline can fall back to a vision-language model without any
API key. Design guards:

- **Out-of-process**: torch/transformers run inside `vlm_worker.py` in a
  separate python process. On Windows, PaddlePaddle (cuDNN 9.9) and torch
  (cuDNN 9.10) cannot share one process — the CUDA runtime DLL names collide
  and whichever imports second fails. The worker keeps the API process
  paddle-only; the model stays resident in the child between extractions.
- **No surprise downloads**: the HF snapshot must already be cached
  (`huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct`); the adapter
  raises instead of pulling gigabytes mid-request.
- **VRAM-bounded**: 4-bit NF4 weights (~2.5 GB) + `max_pixels` cap on visual
  tokens; `keep_loaded=false` unloads (terminates) the worker after a call.
- **Honest output**: the model returns `null` for anything it cannot read;
  parsing validates against the project's `InvoiceData` schema and anything
  unparsable raises instead of guessing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..schemas.invoice import InvoiceData

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Fields the hybrid pipeline may accept from the VLM (mirrors InvoiceData).
_PROMPT_FIELDS = (
    "invoice_id", "invoice_series", "invoice_form", "seller", "buyer",
    "address", "buyer_address", "tax_id", "buyer_tax_id", "currency",
    "timestamp", "subtotal", "tax", "discount", "total_cost",
)

_SYSTEM_PROMPT = (
    "Bạn là hệ thống IDP trích xuất dữ liệu từ ảnh hóa đơn tiếng Việt. "
    "CHỈ đọc từ ảnh, tuyệt đối không bịa dữ liệu. Trả về MỘT khối JSON duy nhất "
    "(không markdown, không giải thích) với đúng các key sau:\n"
    + ", ".join(_PROMPT_FIELDS)
    + ", items\n"
    "Quy ước:\n"
    "- Chuỗi không đọc được → null. Số tiền là số (không chuỗi, không dấu phân cách).\n"
    "- timestamp dạng ISO 8601 'YYYY-MM-DDTHH:MM:SS' khi ảnh có ngày (và giờ nếu có).\n"
    "- items là mảng object {name, quantity, unit_price, amount}; thiếu dòng → mảng rỗng.\n"
    "- Với hóa đơn VND, dấu '.' trong số là phân cách nghìn (30.000 = 30000); "
    "với RM/USD thì '.' là thập phân.\n"
    "- Thêm key 'evidence': object ánh xạ mỗi trường CÓ giá trị sang một trích dẫn "
    "ngắn (tối đa 12 từ) đọc trực tiếp từ ảnh làm bằng chứng."
)

# Pure-stdlib import check that runs in a THROWAWAY process: importing torch
# inside the API process would poison paddle's CUDA DLLs (see module docstring).
_DEPS_PROBE = (
    "import json,sys\n"
    "info={'ok': True, 'cuda': False, 'error': None}\n"
    "try:\n"
    "    import torch\n"
    "    info['cuda'] = bool(torch.cuda.is_available())\n"
    "except Exception as exc:\n"
    "    info['ok'] = False; info['error'] = f'torch: {type(exc).__name__}: {exc}'\n"
    "if info['ok']:\n"
    "    try:\n"
    "        import transformers\n"
    "    except Exception as exc:\n"
    "        info['ok'] = False; info['error'] = f'transformers: {type(exc).__name__}: {exc}'\n"
    "print(json.dumps(info))\n"
)


class LocalVLMError(RuntimeError):
    """Raised when the local VLM cannot load or generate a valid result."""


@lru_cache(maxsize=4)
def _probe_deps_cached(python_exe: str) -> tuple[bool, str]:
    """Run the torch/transformers probe in a subprocess and cache the answer."""
    try:
        completed = subprocess.run(
            [python_exe, "-c", _DEPS_PROBE],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"probe thất bại: {exc}"
    try:
        info = json.loads((completed.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, (completed.stderr or "probe không trả về kết quả")[-300:]
    if info.get("ok"):
        if not info.get("cuda"):
            return True, "torch có CUDA=false — VLM sẽ chạy CPU (rất chậm)"
        return True, ""
    return False, str(info.get("error") or "dependency probe failed")


class LocalVLM:
    """Parent-side handle for the out-of-process Qwen2.5-VL worker."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_pixels: int = 1_003_520,
        max_new_tokens: int = 768,
        keep_loaded: bool = True,
        python_exe: str | None = None,
    ):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.max_pixels = max(196_608, int(max_pixels or 1_003_520))  # floor 250*28*28
        self.max_new_tokens = max(128, int(max_new_tokens or 768))
        self.keep_loaded = bool(keep_loaded)
        self._python = python_exe or sys.executable
        self._proc: subprocess.Popen | None = None
        self._stderr_path: Path | None = None
        self._stderr_handle = None
        self._lock = threading.Lock()
        self._request_seq = 0
        self.last_device: str | None = None
        self.last_ms: float | None = None
        self.last_visual_tokens: int | None = None
        self.last_vram_peak_mb: float | None = None

    # ------------------------------------------------------------ availability

    def deps_available(self) -> tuple[bool, str]:
        """Cheap check (cached per worker python): torch+transformers usable?"""
        return _probe_deps_cached(self._python)

    def model_cached(self) -> bool:
        """True when the HF snapshot is already on disk (no network needed)."""
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
            snapshot_download(repo_id=self.model_id, local_files_only=True)
            return True
        except Exception:
            return False

    def describe(self) -> dict[str, Any]:
        deps_ok, deps_note = self.deps_available()
        return {
            "provider": "local",
            "model": self.model_id,
            "mode": "subprocess (torch tách khỏi paddle)",
            "device": self.last_device,
            "loaded": self._proc is not None and self._proc.poll() is None,
            "available": deps_ok and self.model_cached(),
            "note": deps_note or None,
            "max_pixels": self.max_pixels,
            "last_ms": self.last_ms,
            "last_visual_tokens": self.last_visual_tokens,
            "last_vram_peak_mb": self.last_vram_peak_mb,
        }

    # ------------------------------------------------------------------ worker

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._stderr_path = Path(__file__).resolve().parent / "_vlm_worker_stderr.log"
        self._stderr_handle = self._stderr_path.open("w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [
                self._python, "-X", "utf8", "-m",
                "backend.app.idp_engine.vlm_worker",
                self.model_id, str(self.max_pixels), str(self.max_new_tokens),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            cwd=str(Path(__file__).resolve().parents[3]),
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        return self._proc

    def _request(self, payload: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
        proc = self._ensure_worker()
        self._request_seq += 1
        payload["id"] = self._request_seq
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self._kill_worker()
            raise LocalVLMError(f"VLM worker không nhận lệnh ({exc}); worker có thể đã chết.") from exc
        while True:
            line = proc.stdout.readline()
            if not line:
                tail = self._stderr_tail()
                self._kill_worker()
                raise LocalVLMError(f"VLM worker chết bất thường. stderr: {tail or '(trống)'}")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue  # stray non-JSON line from a chatty library
            if response.get("id") != payload["id"]:
                continue  # stale reply from a previous request
            if not response.get("ok"):
                raise LocalVLMError(str(response.get("error") or "VLM worker trả lỗi không rõ."))
            return response

    def _stderr_tail(self, limit: int = 400) -> str:
        try:
            if self._stderr_path is not None and self._stderr_path.is_file():
                return self._stderr_path.read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            pass
        return ""

    def _kill_worker(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
            except OSError:
                pass
            self._proc = None
        if getattr(self, "_stderr_handle", None) is not None:
            try:
                self._stderr_handle.close()
            except OSError:
                pass
            self._stderr_handle = None

    # ---------------------------------------------------------------- extract

    def extract(self, image_path: str | Path) -> tuple[InvoiceData, list[dict[str, Any]], str]:
        """Return (InvoiceData, boxes=[], raw_text=raw JSON) — VLMExtractor shape."""
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise LocalVLMError(f"Không tìm thấy ảnh cho VLM: {path}")
        with self._lock:
            response = self._request({"op": "extract", "image": str(path)})
            if not self.keep_loaded:
                try:
                    self._request({"op": "unload"})
                except LocalVLMError:
                    self._kill_worker()
        self.last_device = response.get("device")
        self.last_ms = response.get("ms")
        self.last_visual_tokens = response.get("visual_tokens")
        self.last_vram_peak_mb = response.get("vram_peak_mb")
        payload = response.get("data")
        if not isinstance(payload, dict):
            raise LocalVLMError("VLM worker trả dữ liệu không đúng cấu trúc.")
        data = InvoiceData.model_validate(payload)
        return data, [], json.dumps(payload, ensure_ascii=False)

    def unload(self) -> None:
        """Free the child's VRAM (worker stays alive unless unresponsive)."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            try:
                self._request({"op": "unload"}, timeout=120.0)
            except (LocalVLMError, OSError):
                self._kill_worker()

    def shutdown(self) -> None:
        with self._lock:
            self._kill_worker()

    # ------------------------------------------------------------------ parse

    _MONEY_KEYS = frozenset({"subtotal", "tax", "discount", "total_cost", "quantity", "unit_price", "amount"})
    _STRING_FIELDS = frozenset({
        "invoice_id", "invoice_series", "invoice_form", "seller", "buyer",
        "address", "buyer_address", "tax_id", "buyer_tax_id", "currency",
    })

    @classmethod
    def _coerce_value(cls, key: str, value: Any) -> Any:
        """Tolerate realistic VLM output shapes without fabricating data.

        Models love '12.000.000đ' strings, nulls for money fields and the
        occasional {'name': ..., 'address': ...} object where a plain string
        belongs. Money strings go through the project's normalize_money
        (dot-thousands vs dot-decimal aware); null money becomes 0 (schema
        default, semantically 'not extracted'); dict/list values for string
        fields collapse to their text content.
        """
        from .postprocessor import normalize_money

        if key in cls._MONEY_KEYS:
            if value is None:
                return 0.0
            if isinstance(value, bool):
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            return normalize_money(value)
        if key in cls._STRING_FIELDS:
            if isinstance(value, dict):
                for candidate in ("name", "company", "value", "text"):
                    inner = value.get(candidate)
                    if isinstance(inner, str) and inner.strip():
                        return inner.strip()
                joined = ", ".join(str(item) for item in value.values() if item)
                return joined or None
            if isinstance(value, list):
                joined = ", ".join(str(item) for item in value if item)
                return joined or None
        return value

    @classmethod
    def _coerce_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        coerced: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "items" and isinstance(value, list):
                items: list[Any] = []
                for item in value:
                    if isinstance(item, dict):
                        items.append({k: cls._coerce_value(k, v) for k, v in item.items()})
                    else:
                        items.append(item)
                coerced[key] = items
            else:
                coerced[key] = cls._coerce_value(key, value)
        return coerced

    @staticmethod
    def parse_output(raw: str) -> tuple[InvoiceData, dict[str, str]]:
        """Parse the model's JSON; raises on garbage instead of fabricating."""
        cleaned = (raw or "").strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"Local VLM không trả về JSON: {raw[:200]!r}")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Local VLM JSON lỗi cú pháp: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Local VLM JSON phải là object.")
        evidence_raw = payload.pop("evidence", {}) or {}
        evidence = {
            str(key): str(value)[:200]
            for key, value in evidence_raw.items()
            if isinstance(key, str) and value
        } if isinstance(evidence_raw, dict) else {}
        data = InvoiceData.model_validate(LocalVLM._coerce_payload(payload))
        return data, evidence
