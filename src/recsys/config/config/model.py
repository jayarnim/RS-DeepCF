from dataclasses import dataclass


@dataclass
class RLNetCfg:
    num_users: int
    num_items: int
    projection_dim: int
    hidden_dim: list
    dropout: float


@dataclass
class MLNetCfg:
    num_users: int
    num_items: int
    projection_dim: int
    hidden_dim: list
    dropout: float


@dataclass
class CFNetCfg:
    rlnet: RLNetCfg
    mlnet: MLNetCfg