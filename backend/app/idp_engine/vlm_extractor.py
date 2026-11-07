from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from ..schemas.invoice import InvoiceData


class VLMExtractor:
    """Structured-output adapters for OpenAI Vision and Gemini.

    The provider boundary is deliberately small so a local Qwen2.5-VL adapter or
    a mocked client can be injected in tests without changing the pipeline.
    """

    def __init__(
        self,
        provider: str = "mock",
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
    ):
        self.provider = (provider or "mock").lower()
        self.api_key = api_key
        self.model = model
        self.client = client

    def extract(self, image_path: str | Path) -> tuple[InvoiceData, list[dict], str]:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"Không tìm thấy ảnh cho VLM: {path}")
        if self.provider == "mock":
            return InvoiceData(), [], ""
        if not self.api_key:
            raise RuntimeError(f"VLM provider {self.provider!r} cần API key.")
        if self.provider == "openai":
            payload = self._extract_openai(path)
        elif self.provider in {"gemini", "google"}:
            payload = self._extract_gemini(path)
        else:
            raise RuntimeError(
                f"VLM provider {self.provider!r} chưa được hỗ trợ; dùng mock, openai hoặc gemini."
            )
        data = self.parse_structured(payload)
        return data, [], json.dumps(data.model_dump(mode="json"), ensure_ascii=False)

    def _extract_openai(self, path: Path) -> str | dict[str, Any]:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Cài package openai để dùng VLM_PROVIDER=openai.") from exc
        client = self.client or OpenAI(api_key=self.api_key)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = _mime_type(path)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "vietnamese_invoice",
                    "strict": True,
                    "schema": _openai_schema(),
                },
            },
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Đọc hóa đơn trong ảnh và trả về đúng JSON schema."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ],
        )
        return response.choices[0].message.content or "{}"

    def _extract_gemini(self, path: Path) -> str | dict[str, Any]:
        # REST keeps the base install light and avoids coupling to one rapidly
        # changing Google SDK namespace.
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Cài httpx để dùng VLM_PROVIDER=gemini.") from exc
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        response = httpx.post(
            endpoint,
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": self._system_prompt()}, {"inline_data": {"mime_type": _mime_type(path), "data": encoded}}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                    "responseSchema": _gemini_schema(),
                },
            },
            timeout=90,
        )
        response.raise_for_status()
        body = response.json()
        try:
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini không trả về structured JSON hợp lệ.") from exc

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Bạn là hệ thống IDP hóa đơn tiếng Việt. Trích xuất seller, address, invoice_id, "
            "timestamp, subtotal, tax, discount, total_cost và items. Số tiền là số, ngày giờ "
            "ISO 8601 nếu đọc được. Không đoán dữ liệu không có; dùng null hoặc 0. Chỉ trả JSON."
        )

    @staticmethod
    def parse_structured(payload: str | dict[str, Any]) -> InvoiceData:
        if isinstance(payload, str):
            cleaned = payload.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise ValueError("VLM output không phải JSON hợp lệ.") from exc
        return InvoiceData.model_validate(payload)


def _openai_schema() -> dict[str, Any]:
    """A flat strict schema compatible with OpenAI structured outputs."""
    number = {"type": "number"}
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    nullable_box = {
        "anyOf": [
            {"type": "array", "items": {"type": "number"}},
            {"type": "null"},
        ]
    }
    item_properties = {
        "name": {"type": "string"},
        "quantity": number,
        "unit_price": number,
        "amount": number,
        "confidence": number,
        "bbox": nullable_box,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "invoice_id": nullable_string,
            "seller": nullable_string,
            "address": nullable_string,
            "timestamp": nullable_string,
            "subtotal": number,
            "tax": number,
            "discount": number,
            "total_cost": number,
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": item_properties,
                    "required": list(item_properties),
                },
            },
        },
        "required": [
            "invoice_id", "seller", "address", "timestamp", "subtotal", "tax",
            "discount", "total_cost", "items",
        ],
    }


def _gemini_schema() -> dict[str, Any]:
    """Gemini REST Schema form, kept separate from OpenAI's JSON Schema form."""
    def nullable(kind: str) -> dict[str, Any]:
        return {"type": kind, "nullable": True}

    item_properties = {
        "name": {"type": "STRING"},
        "quantity": {"type": "NUMBER"},
        "unit_price": {"type": "NUMBER"},
        "amount": {"type": "NUMBER"},
        "confidence": {"type": "NUMBER"},
        "bbox": {"type": "ARRAY", "items": {"type": "NUMBER"}, "nullable": True},
    }
    return {
        "type": "OBJECT",
        "properties": {
            "invoice_id": nullable("STRING"),
            "seller": nullable("STRING"),
            "address": nullable("STRING"),
            "timestamp": nullable("STRING"),
            "subtotal": {"type": "NUMBER"},
            "tax": {"type": "NUMBER"},
            "discount": {"type": "NUMBER"},
            "total_cost": {"type": "NUMBER"},
            "items": {
                "type": "ARRAY",
                "items": {"type": "OBJECT", "properties": item_properties, "required": list(item_properties)},
            },
        },
        "required": [
            "invoice_id", "seller", "address", "timestamp", "subtotal", "tax",
            "discount", "total_cost", "items",
        ],
    }


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/jpeg")
