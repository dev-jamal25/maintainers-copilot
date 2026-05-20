from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_SUMMARY_INPUT_LENGTH = 200_000
MAX_SUMMARY_LENGTH_LIMIT = 512


class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_SUMMARY_INPUT_LENGTH)
    max_length: int = Field(default=160, ge=10, le=MAX_SUMMARY_LENGTH_LIMIT)
    min_length: int = Field(default=30, ge=5, le=256)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value

    @model_validator(mode="after")
    def max_length_must_exceed_min_length(self) -> Self:
        if self.max_length <= self.min_length:
            raise ValueError("max_length must be greater than min_length")
        return self


class SummarizationResponse(BaseModel):
    summary: str
    model: str
