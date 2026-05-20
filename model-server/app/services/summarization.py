from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.schemas.summarization import SummarizationResponse

SUMMARIZATION_MODEL_NAME: Final[str] = "sshleifer/distilbart-cnn-12-6"
MAX_SUMMARIZATION_INPUT_CHARS: Final[int] = 20_000
MAX_MODEL_INPUT_TOKENS: Final[int] = 1024


@dataclass(frozen=True)
class SummarizationComponents:
    tokenizer: Any
    model: Any
    device: str


def inference_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)


def load_model() -> Any:
    return AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)


@lru_cache(maxsize=1)
def get_summarization_components() -> SummarizationComponents:
    tokenizer = load_tokenizer()
    model = load_model()
    device = inference_device()
    model.to(device)
    model.eval()
    return SummarizationComponents(tokenizer=tokenizer, model=model, device=device)


def summarize_text(
    text: str, *, max_length: int = 160, min_length: int = 30
) -> SummarizationResponse:
    source_text = text[:MAX_SUMMARIZATION_INPUT_CHARS]
    components = get_summarization_components()
    inputs = components.tokenizer(
        source_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_INPUT_TOKENS,
    )
    inputs = {key: value.to(components.device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = components.model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
        )
    summary = components.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return SummarizationResponse(summary=summary, model=SUMMARIZATION_MODEL_NAME)
