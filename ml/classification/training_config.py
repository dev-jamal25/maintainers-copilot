from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from classification.data import REPO_ROOT, display_path, resolve_repo_path

JsonObject = dict[str, Any]

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
SEED = 42
WANDB_PROJECT = "maintainers-copilot-classification"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "classification" / "distilbert"
DISTILBERT_ENCODER_LAYER_COUNT = 6

FreezePolicyName = Literal["none", "embeddings", "embeddings_and_lower_layers"]


class TrainingConfigError(ValueError):
    """Raised when transformer training configuration is invalid."""


@dataclass(frozen=True)
class FreezePolicy:
    name: FreezePolicyName = "none"
    freeze_embeddings: bool = False
    freeze_encoder_layer_indices: tuple[int, ...] = field(default_factory=tuple)

    @classmethod
    def none(cls) -> FreezePolicy:
        return cls()

    @classmethod
    def embeddings_only(cls) -> FreezePolicy:
        return cls(name="embeddings", freeze_embeddings=True)

    @classmethod
    def embeddings_and_lower_layers(cls, layer_count: int) -> FreezePolicy:
        if layer_count <= 0:
            raise TrainingConfigError("layer_count must be positive")
        return cls(
            name="embeddings_and_lower_layers",
            freeze_embeddings=True,
            freeze_encoder_layer_indices=tuple(range(layer_count)),
        )

    def __post_init__(self) -> None:
        if self.name == "none":
            if self.freeze_embeddings or self.freeze_encoder_layer_indices:
                raise TrainingConfigError("'none' freeze policy cannot freeze model parameters")
        elif self.name == "embeddings":
            if not self.freeze_embeddings:
                raise TrainingConfigError("'embeddings' freeze policy must freeze embeddings")
            if self.freeze_encoder_layer_indices:
                raise TrainingConfigError("'embeddings' freeze policy cannot freeze encoder layers")
        elif self.name == "embeddings_and_lower_layers":
            if not self.freeze_embeddings:
                raise TrainingConfigError(
                    "'embeddings_and_lower_layers' freeze policy must freeze embeddings"
                )
            validate_layer_indices(self.freeze_encoder_layer_indices)
        else:
            raise TrainingConfigError(f"unsupported freeze policy {self.name!r}")

    def to_dict(self) -> JsonObject:
        return {
            "freeze_embeddings": self.freeze_embeddings,
            "freeze_encoder_layer_indices": list(self.freeze_encoder_layer_indices),
            "name": self.name,
        }


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = MODEL_NAME
    max_length: int = MAX_LENGTH
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    seed: int = SEED
    output_dir: Path = OUTPUT_DIR
    wandb_project: str = WANDB_PROJECT
    freeze_policy: FreezePolicy = field(default_factory=FreezePolicy.none)

    def __post_init__(self) -> None:
        if not self.model_name:
            raise TrainingConfigError("model_name must be non-empty")
        if self.max_length <= 0 or self.max_length > 512:
            raise TrainingConfigError("max_length must be between 1 and 512")
        if self.batch_size <= 0:
            raise TrainingConfigError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise TrainingConfigError("learning_rate must be positive")
        if self.epochs <= 0:
            raise TrainingConfigError("epochs must be positive")
        if self.seed < 0:
            raise TrainingConfigError("seed must be non-negative")
        if not self.wandb_project:
            raise TrainingConfigError("wandb_project must be non-empty")

    @property
    def resolved_output_dir(self) -> Path:
        return resolve_repo_path(self.output_dir)

    def to_dict(self) -> JsonObject:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "freeze_policy": self.freeze_policy.to_dict(),
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "model_name": self.model_name,
            "output_dir": display_path(self.resolved_output_dir),
            "seed": self.seed,
            "wandb_project": self.wandb_project,
        }


def validate_layer_indices(layer_indices: tuple[int, ...]) -> None:
    if not layer_indices:
        raise TrainingConfigError("freeze_encoder_layer_indices must not be empty")
    if tuple(sorted(layer_indices)) != layer_indices:
        raise TrainingConfigError("freeze_encoder_layer_indices must be sorted")
    if len(set(layer_indices)) != len(layer_indices):
        raise TrainingConfigError("freeze_encoder_layer_indices must be unique")
    invalid = [
        index for index in layer_indices if index < 0 or index >= DISTILBERT_ENCODER_LAYER_COUNT
    ]
    if invalid:
        raise TrainingConfigError(
            "freeze_encoder_layer_indices must be within DistilBERT layer range 0-5; "
            f"invalid: {invalid}"
        )


def default_training_config() -> TrainingConfig:
    return TrainingConfig()
