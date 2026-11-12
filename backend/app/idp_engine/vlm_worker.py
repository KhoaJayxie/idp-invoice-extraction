"""Out-of-process VLM worker (P4 hybrid).

Runs torch + transformers + Qwen2.5-VL in a SEPARATE python process. Reason:
on Windows one process can hold only one copy of the CUDA runtime DLLs by
name (cudnn64_9.dll, cublas64_12.dll) — PaddlePaddle 3.4 (cu129, cuDNN 9.9)
and torch 2.8 (cu128, cuDNN 9.10) ship incompatible builds, so loading both
in one process breaks whichever imports second. This worker keeps the API
process paddle-only and the VLM torch-only.

Protocol: one JSON request per stdin line → one JSON response per stdout line.

    {"id": 1, "op": "extract", "image": "C:/path/aligned.png"}
    {"id": 1, "ok": true, "data": {...}, "evidence": {...}, "device": "cuda", "ms": 1234}

Ops: "extract" (loads the model on first use and keeps it resident),
"unload" (frees VRAM), "ping" (reports load state without loading).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the project package importable regardless of the parent's cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.idp_engine.local_vlm import (  # noqa: E402
    _SYSTEM_PROMPT,
    LocalVLM,
    LocalVLMError,
)


def _build_inputs(processor, image, max_pixels: int):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Đọc hóa đơn và trả về đúng JSON như yêu cầu."},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        return processor(images=[image], text=[prompt], return_tensors="pt", max_pixels=max_pixels)
    except TypeError:
        return processor(images=[image], text=[prompt], return_tensors="pt")


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-VL-3B-Instruct"
    max_pixels = int(sys.argv[2]) if len(sys.argv) > 2 else 1_003_520
    max_new_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 768

    os_env_quiet = {
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    import os  # noqa: PLC0415

    for key, value in os_env_quiet.items():
        os.environ.setdefault(key, value)

    state = {"model": None, "processor": None, "torch": None, "device": "cpu"}

    def respond(payload: dict) -> None:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def load() -> None:
        if state["model"] is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415

            if torch.cuda.is_available():
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, quantization_config=quant, device_map={"": 0}
                )
                device = "cuda"
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.float32, device_map={"": "cpu"}
                )
                device = "cpu"
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            # Partial load must not poison the next attempt: the parent would
            # otherwise see "model loaded" with a missing processor.
            state.update(model=None, processor=None, torch=None, device="cpu")
            raise
        state["torch"] = torch
        state["model"] = model
        state["processor"] = processor
        state["device"] = device

    def unload() -> None:
        import gc  # noqa: PLC0415

        state["model"] = None
        state["processor"] = None
        if state["torch"] is not None:
            gc.collect()
            if state["torch"].cuda.is_available():
                state["torch"].cuda.empty_cache()
        state["device"] = "cpu"

    def extract(image_path: str) -> dict:
        path = Path(image_path)
        if not path.is_file():
            raise LocalVLMError(f"Không tìm thấy ảnh cho VLM: {path}")
        load()
        torch = state["torch"]
        processor = state["processor"]
        model = state["model"]
        from PIL import Image  # noqa: PLC0415

        image = Image.open(path).convert("RGB")
        inputs = _build_inputs(processor, image, max_pixels)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        started = time.perf_counter()
        vram_peak_mb = None
        if state["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1]:]
        raw = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        if state["device"] == "cuda":
            vram_peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        data, evidence = LocalVLM.parse_output(raw)
        payload = data.model_dump(mode="json")
        payload["evidence"] = evidence
        return {
            "data": payload,
            "raw": json.dumps(payload, ensure_ascii=False),
            "device": state["device"],
            "ms": elapsed_ms,
            "visual_tokens": int(inputs["input_ids"].shape[1]),
            "vram_peak_mb": vram_peak_mb,
        }

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        request_id = request.get("id")
        op = request.get("op")
        try:
            if op == "ping":
                respond({"id": request_id, "ok": True, "loaded": state["model"] is not None,
                         "device": state["device"]})
            elif op == "unload":
                unload()
                respond({"id": request_id, "ok": True, "loaded": False})
            elif op == "extract":
                result = extract(str(request.get("image", "")))
                respond({"id": request_id, "ok": True, **result})
            else:
                respond({"id": request_id, "ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:  # any worker failure must reach the parent
            respond({"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
