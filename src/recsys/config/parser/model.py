from ..config.model import (
    RLNetCfg, 
    MLNetCfg, 
    CFNetCfg,
)


def model(cfg):
    cls = cfg["model"]["name"]

    if cls=="rlnet":
        return rlnet(cfg)
    elif cls=="mlnet":
        return mlnet(cfg)
    elif cls=="cfnet":
        return cfnet(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def rlnet(cfg):
    return RLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def mlnet(cfg):
    return MLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def cfnet(cfg):
    rlnet = RLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["rlnet"],
    )
    mlnet = MLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["mlnet"],
    )
    return CFNetCfg(
        rlnet=rlnet,
        mlnet=mlnet,
    )