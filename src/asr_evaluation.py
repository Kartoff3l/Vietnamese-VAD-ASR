#take reference text, predicted text, runtime, audio duration
#then return normalized text and WER, CER and RTF
from __future__ import annotations

import re
import unicodedata
from typing import Any

from jiwer import cer, wer

def normalize_vietnamese_text(text: str) -> str:
    """Normalize Vietnamese transcript text before WER/CER evaluation.
    
    Rules:
    -Convert Unicode to NFC
    -Convert to lowercase.
    -Remove punctuation.
    -Collapse repeated whitespace.
    -Preserve Vietnamese diacritics.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def calculate_rtf(runtime_s: float, audio_duration_s: float) -> float:
    """
    Calculate Real-Time Factor.
    
    RTF= processing runtime / original audio duration.
    
    """
    if runtime_s < 0:
        raise ValueError("runtime_s must not be negative")

    if audio_duration_s <= 0:
        raise ValueError("audio_duration_s must bigger than zero")

    return runtime_s / audio_duration_s

def calculate_text_metrics(reference: str, prediction: str) ->dict[str,Any]:
    """
    Normalize reference/prediction and calculate WER and CER
    """
    reference_normalized = normalize_vietnamese_text(reference)
    prediction_normalized = normalize_vietnamese_text(prediction)

    if not reference_normalized:
        raise ValueError("reference text is empty after normalization")

    return{
        "reference_normalized": reference_normalized,
        "prediction_normalized": prediction_normalized,
        "wer": wer(reference_normalized, prediction_normalized),
        "cer": cer(reference_normalized, prediction_normalized),
    }

def evaluate_asr_result(
        reference: str,
        prediction: str,
        asr_runtime_s: float,
        audio_duration_s: float,
) -> dict[str, Any]:
    """"
    Evaluate one ASR prediction using normalized WER, CER, and ASR RTF
    """
    text_metrics = calculate_text_metrics(reference, prediction)
    asr_rtf = calculate_rtf(asr_runtime_s, audio_duration_s)

    return{
        **text_metrics,
        "asr_runtime_seconds": round(asr_runtime_s, 3),
        "audio_duration_s": round(audio_duration_s, 3),
        "asr_rtf": round(asr_rtf, 4),
    }
