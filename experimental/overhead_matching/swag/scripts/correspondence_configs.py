"""Typed configuration for landmark correspondence training."""

from pathlib import Path

import msgspec

from common.python.serialization import MSGSPEC_STRUCT_OPTS


class CorrespondenceEncoderConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    text_input_dim: int
    text_proj_dim: int


class CorrespondenceClassifierMlpConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    mlp_hidden_dim: int
    dropout: float


class CorrespondenceTrainConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    data_dir: Path
    text_embeddings_path: Path
    output_dir: Path
    train_city: str
    val_city: str
    include_difficulties: list[str]
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    warmup_fraction: float
    gradient_clip_norm: float
    use_amp: bool
    num_workers: int
    seed: int
    encoder: CorrespondenceEncoderConfig
    classifier: CorrespondenceClassifierMlpConfig
    cosine_schedule: bool = False


def _enc_hook(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise NotImplementedError(f"Cannot serialize {type(obj)}")


def _dec_hook(type_, obj):
    if type_ is Path:
        return Path(obj)
    raise NotImplementedError(f"Cannot deserialize {type_}")


def save_config(config: CorrespondenceTrainConfig, path: Path) -> None:
    yaml_bytes = msgspec.yaml.encode(config, enc_hook=_enc_hook)
    path.write_bytes(yaml_bytes)


def load_config(path: Path) -> CorrespondenceTrainConfig:
    yaml_bytes = path.read_bytes()
    return msgspec.yaml.decode(
        yaml_bytes, type=CorrespondenceTrainConfig, dec_hook=_dec_hook
    )
