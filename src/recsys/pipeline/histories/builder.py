import torch
from .histories import histories_generator
from .selector.builder import selector_builder


def histories_builder(
    interactions: torch.Tensor, 
    cfg,
):
    interactions = interactions[:-1, :-1]

    histories = dict()

    entities = ["user", "item"]
    matrices = [interactions, interactions.T]

    for entity, mat in zip(entities, matrices):
        num_anchor, num_target = mat.shape

        kwargs = dict(
            interactions=mat,
            cfg=cfg.selector,
        )
        selector = selector_builder(**kwargs)

        kwargs = dict(
            selector=selector,
            padding_idx=num_target,
        )
        histories[entity] = histories_generator(**kwargs)

    return histories