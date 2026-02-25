from dataclasses import dataclass
from typing import Literal, Union
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .evaluator import EvaluatorCfg
from .schema import SchemaCfg
from .model import RLNetCfg, MLNetCfg, CFNetCfg


@dataclass
class Config:
    model: Union[RLNetCfg, MLNetCfg, CFNetCfg]
    schema: SchemaCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    strategy: Literal["pointwise", "pairwise", "listwise"]
    model_cls: Literal["rlnet", "mlnet", "cfnet"]
    dataset: str
    seed: int