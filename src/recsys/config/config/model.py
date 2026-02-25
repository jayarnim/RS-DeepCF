from dataclasses import dataclass


@dataclass
class RLNetCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class MLNetCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class CFNetCfg:
    rlnet: RLNetCfg
    mlnet: MLNetCfg
