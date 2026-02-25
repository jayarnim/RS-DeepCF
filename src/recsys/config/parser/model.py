from ..config.model import (
    RLNetCfg,
    MLNetCfg,
    CFNetCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="rlnet":
        return rlnet(cfg)
    elif model=="mlnet":
        return mlnet(cfg)
    elif model=="cfnet":
        return cfnet(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def rlnet(cfg):
    return RLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def mlnet(cfg):
    return MLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def cfnet(cfg):
    rlnet = RLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["rlnet"]["projection_dim"],
        hidden_dim=cfg["model"]["rlnet"]["hidden_dim"],
        dropout=cfg["model"]["rlnet"]["dropout"],
    )
    mlnet = MLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["mlnet"]["projection_dim"],
        hidden_dim=cfg["model"]["mlnet"]["hidden_dim"],
        dropout=cfg["model"]["mlnet"]["dropout"],
    )
    return CFNetCfg(
        rlnet=rlnet,
        mlnet=mlnet,
    )

