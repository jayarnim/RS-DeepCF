import torch
import torch.nn as nn
from . import rlnet, mlnet


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
        self.matching_dim = rlnet.matching_dim + mlnet.matching_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        # modules
        matching_vec_rl = self.rlnet(user_idx, item_idx)
        matching_vec_ml = self.mlnet(user_idx, item_idx)

        # agg
        kwargs = dict(
            tensors=(matching_vec_rl, matching_vec_ml), 
            dim=-1,
        )
        matching_vec_fusion = torch.cat(**kwargs)

        return matching_vec_fusion

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        matching_vec = self.forward(user_idx, item_idx)
        logit = self.prediction(matching_vec).squeeze(-1)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        matching_vec = self.forward(user_idx, item_idx)
        logit = self.prediction(matching_vec).squeeze(-1)
        return logit

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.matching_dim,
            out_features=1, 
        )
        self.prediction = nn.Linear(**kwargs)