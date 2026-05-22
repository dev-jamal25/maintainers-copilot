from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_TEXTS = 64
MAX_TEXT_CHARS = 20_000


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TEXTS)
    is_query: bool = Field(
        default=False,
        description="Apply the bge query instruction prefix (asymmetric retrieval).",
    )

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_blank(cls, value: list[str]) -> list[str]:
        if any(not text.strip() for text in value):
            raise ValueError("texts must not contain blank entries")
        return [text[:MAX_TEXT_CHARS] for text in value]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dim: int
