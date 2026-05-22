from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 20_000


class ClassifyRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    body: str = Field(default="", max_length=MAX_TEXT_CHARS)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be empty")
        return value


class ClassifyResponse(BaseModel):
    label: str
    scores: dict[str, float]
    model: str
