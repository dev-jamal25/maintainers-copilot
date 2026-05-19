from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from classification.training_config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    OUTPUT_DIR,
    SEED,
    WANDB_PROJECT,
    FreezePolicy,
    TrainingConfig,
    TrainingConfigError,
    default_training_config,
)


def test_default_training_config_matches_phase_4_contract() -> None:
    config = default_training_config()

    assert config.model_name == MODEL_NAME == "distilbert-base-uncased"
    assert config.max_length == MAX_LENGTH == 256
    assert config.batch_size == BATCH_SIZE == 16
    assert config.learning_rate == LEARNING_RATE == 2e-5
    assert config.epochs == EPOCHS == 3
    assert config.seed == SEED == 42
    assert config.output_dir == OUTPUT_DIR
    assert config.wandb_project == WANDB_PROJECT == "maintainers-copilot-classification"
    assert config.freeze_policy == FreezePolicy.none()


def test_training_config_serializes_to_json_safe_dict() -> None:
    config = TrainingConfig(
        output_dir=Path("artifacts/classification/custom-distilbert"),
        freeze_policy=FreezePolicy.embeddings_only(),
    )

    as_dict = config.to_dict()

    assert as_dict["model_name"] == "distilbert-base-uncased"
    assert as_dict["output_dir"] == "artifacts/classification/custom-distilbert"
    assert as_dict["freeze_policy"] == {
        "freeze_embeddings": True,
        "freeze_encoder_layer_indices": [],
        "name": "embeddings",
    }


def test_freeze_policy_can_target_embeddings_and_lower_layers() -> None:
    policy = FreezePolicy.embeddings_and_lower_layers(3)

    assert policy.freeze_embeddings is True
    assert policy.freeze_encoder_layer_indices == (0, 1, 2)
    assert policy.to_dict()["name"] == "embeddings_and_lower_layers"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_name": ""}, "model_name must be non-empty"),
        ({"max_length": 0}, "max_length must be between 1 and 512"),
        ({"max_length": 513}, "max_length must be between 1 and 512"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"epochs": 0}, "epochs must be positive"),
        ({"seed": -1}, "seed must be non-negative"),
        ({"wandb_project": ""}, "wandb_project must be non-empty"),
    ],
)
def test_training_config_rejects_invalid_values(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(TrainingConfigError, match=message):
        TrainingConfig(**kwargs)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (
            lambda: FreezePolicy(name="none", freeze_embeddings=True),
            "'none' freeze policy cannot freeze model parameters",
        ),
        (
            lambda: FreezePolicy(name="embeddings", freeze_embeddings=False),
            "'embeddings' freeze policy must freeze embeddings",
        ),
        (
            lambda: FreezePolicy(
                name="embeddings",
                freeze_embeddings=True,
                freeze_encoder_layer_indices=(0,),
            ),
            "'embeddings' freeze policy cannot freeze encoder layers",
        ),
        (
            lambda: FreezePolicy.embeddings_and_lower_layers(0),
            "layer_count must be positive",
        ),
        (
            lambda: FreezePolicy(
                name="embeddings_and_lower_layers",
                freeze_embeddings=True,
                freeze_encoder_layer_indices=(2, 1),
            ),
            "freeze_encoder_layer_indices must be sorted",
        ),
        (
            lambda: FreezePolicy(
                name="embeddings_and_lower_layers",
                freeze_embeddings=True,
                freeze_encoder_layer_indices=(0, 0),
            ),
            "freeze_encoder_layer_indices must be unique",
        ),
        (
            lambda: FreezePolicy(
                name="embeddings_and_lower_layers",
                freeze_embeddings=True,
                freeze_encoder_layer_indices=(0, 6),
            ),
            "within DistilBERT layer range 0-5",
        ),
    ],
)
def test_freeze_policy_rejects_invalid_values(
    policy: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(TrainingConfigError, match=message):
        policy()
