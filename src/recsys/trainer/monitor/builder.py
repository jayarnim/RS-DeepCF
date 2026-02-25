import torch.nn as nn
from .monitor import Monitor
from .predictor import Predictor
from .calculator import Calculator
from .early_stopper import EarlyStopper
from .metric import METRIC_REGISTRY


def monitor_builder(
    model: nn.Module,
    cfg,
):
    kwargs = dict(
        model=model,
        schema=cfg.schema,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        criterion=METRIC_REGISTRY[cfg.metric],
        k=cfg.k,
        schema=cfg.schema,
    )
    calculator = Calculator(**kwargs)

    kwargs = dict(
        delta=cfg.delta,
        patience=cfg.patience,
        warmup=cfg.warmup,
    )
    early_stopper = EarlyStopper(**kwargs)

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
        early_stopper=early_stopper,
    )
    return Monitor(**kwargs)