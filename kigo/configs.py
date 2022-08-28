from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import json
from pydantic import BaseModel
import yaml

from .utils import Directory, File


class DatasetConfig(BaseModel):
    path: Directory
    extensions: List[str]
    loader_worker_count: int


class ImageConfig(BaseModel):
    size: int
    channels: int

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.size, self.size, self.channels


class TrainingConfig(BaseModel):
    # Optimizer
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    # Batches
    batch_size: int
    ema_alpha: float
    # Logging and other IO
    yield_freq: int
    save_freq: int
    save_checkpoint_freq: int


class UBlockConfig(BaseModel):
    channels: int
    blocks: int
    groupnorm_groups: int
    attention_heads: int
    attention_head_channels: int
    dropout: float


class ModelConfig(BaseModel):
    blocks: List[UBlockConfig]
    outer_groupnorm_groups: int
    outer_channels: int
    output_channels: int
    input_channels: int
    snr_sinusoidal_embedding_width: int
    snr_embedding_width: int


class Config(BaseModel):
    '''Container for all other configs.'''
    ds: DatasetConfig
    img: ImageConfig
    tr: TrainingConfig
    model: ModelConfig

    @classmethod
    def from_yaml(cls, file: File) -> Config:
        with open(file) as fh:
            obj = yaml.safe_load(fh)
        assert isinstance(obj, dict)
        return cls(**obj)

    def to_yaml(self, file: Path) -> Config:
        with open(file, 'w') as fh:
            # self.json() is more robust than self.dict(), so we use it to
            # avoid having to cater to edge cases, such as e.g. properly
            # serializing Paths etc.
            obj = json.loads(self.json())
            yaml.safe_dump(obj, fh)
        return self
