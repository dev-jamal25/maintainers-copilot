from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_PASSAGES = 100
MAX_TEXT_CHARS = 20_000


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    passages: list[str] = Field(..., min_length=1, max_length=MAX_PASSAGES)
    top_k: int | None = Field(default=None, ge=1, le=MAX_PASSAGES)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: list[RerankResult]
    model: str
