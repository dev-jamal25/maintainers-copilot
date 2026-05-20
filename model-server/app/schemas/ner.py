from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_ENTITIES_LIMIT = 500


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200_000)
    max_entities: int = Field(default=100, ge=1, le=MAX_ENTITIES_LIMIT)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    confidence: float = Field(ge=0.0, le=1.0)


class NERResponse(BaseModel):
    entities: list[Entity]
    counts_by_type: dict[str, int]
    model: str
