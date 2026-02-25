import torch
import torch.nn as nn
from . import rlnet, mlnet
from .components.fusion import FusionLayer
from .components.prediction import PredictionLayer


class Module(nn.Module):
    def __init__(
        self,
        rlnet: nn.Module,
        mlnet: nn.Module,
    ):
        """
        Deepcf: A unified framework of representation learning and matching function learning in recommender system (Deng et al., 2019)
        -----
        Implements the base structure of Collaboartive Filtering Networks (CFNet),
        MF, MLP & history embedding based latent factor model,
        combining a Representation Learning Networks (RLNet) and a Matching Function Learning Networks (MLNet)
        to learn low-rank linear represenation & high-rank nonlinear user-item interactions.

        Args:
            rlnet (nn.Module)
            mlnet (nn.Module)
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.rlnet = rlnet
        self.mlnet = mlnet
        self.predictive_dim = rlnet.predictive_dim + mlnet.predictive_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        args = (
            self.rlnet(user_idx, item_idx),
            self.mlnet(user_idx, item_idx),
        )
        predictive_vec = self.fusion(*args)
        return predictive_vec

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.prediction(predictive_vec)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.prediction(predictive_vec)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        self.fusion = FusionLayer()
        
        kwargs = dict(
            dim=self.predictive_dim,
        )
        self.prediction = PredictionLayer(**kwargs)