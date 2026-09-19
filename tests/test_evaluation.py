import pytest

from src.evaluation import(
    calculate_rtf,
    calculate_text_metrics,
    normalize_vietnamese_text,
)

def test_normalize_vietnamese_text():
    text= " Xin chào, tôi đến từ Biên Hòa"

    assert(
        normalize_vietnamese_text(text) == "xin chào tôi đến từ biên hòa"
    )

def test_normalize_preserves_vietnamese_diacritics():
    assert(
        normalize_vietnamese_text("dấu câu Tiếng Việt") == "dấu câu tiếng việt"
    )

def test_identical_text_has_zero_error():
    result = calculate_text_metrics(
        reference= "Hôm nay bạn có đi Bình Dương không?",
        prediction= "hôm nay bạn có đi bình dương không",
    )
    assert result["wer"] == 0.0
    assert result["cer"] == 0.0

def test_rtf():
    assert calculate_rtf(2.5, 10.0) == 0.25

def test_rtf_rejects_zero_duration():
    with pytest.raises(ValueError):
        calculate_rtf(2.0, 0.0)

def test_empty_reference_is_rejected():
    with pytest.raises(ValueError):
        calculate_text_metrics(
            reference= " ",
            prediction="xin chào",
        )