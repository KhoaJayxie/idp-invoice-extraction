from pathlib import Path

import pytest

from backend.app.idp_engine.ocr_detector import OCRDetector
from backend.app.idp_engine.vlm_extractor import VLMExtractor


def test_paddle_result_adapters_accept_2x_and_dict_shapes():
    detector = OCRDetector()
    old_result = [[[[10, 10], [80, 10], [80, 30], [10, 30]], ("Tổng cộng", 0.91)]]
    old_tokens = detector._parse_paddle_result(old_result)
    assert old_tokens[0].text == "Tổng cộng"
    assert old_tokens[0].box == [10.0, 10.0, 80.0, 30.0]
    wrapped_tokens = detector._parse_paddle_result([old_result])
    assert wrapped_tokens[0].text == "Tổng cộng"

    new_tokens = detector._parse_paddle_result({
        "rec_texts": ["VAT"],
        "rec_scores": [0.88],
        "dt_polys": [[[1, 2], [30, 2], [30, 18], [1, 18]]],
    })
    assert new_tokens[0].text == "VAT"
    assert new_tokens[0].confidence == 0.88


def test_vlm_structured_parser_accepts_json_code_fence():
    parsed = VLMExtractor.parse_structured(
        '```json\n{"seller":"ABC","total_cost":100,"items":[],"field_confidence":{}}\n```'
    )
    assert parsed.seller == "ABC"
    assert parsed.total_cost == 100


def test_vlm_provider_requires_key_and_unknown_provider_is_clear(tmp_path: Path):
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"not used")
    with pytest.raises(RuntimeError, match="API key"):
        VLMExtractor("openai").extract(image)
    with pytest.raises(RuntimeError, match="chưa được hỗ trợ"):
        VLMExtractor("local", "key").extract(image)
