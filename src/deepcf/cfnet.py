import torch
from components.base import BaseModel
from .rlnet import RepresentationLearningNetworks
from .mlnet import MatchingLearningNetworks
from .layers.fusion import ConcatenationLayer
from .layers.prediction import ProjectionLayer


class CollaborativeFilteringNetworks(BaseModel):
    def __init__(
        self,
        rlnet: RepresentationLearningNetworks,
        mlnet: MatchingLearningNetworks,
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
        super().__init__(locals())

        # ENSEMBLE MODULES ==========
        self.rlnet = rlnet
        self.mlnet = mlnet
        self.pred_dim = rlnet.pred_dim + mlnet.pred_dim

        # FUSION ==========
        self.fusion = ConcatenationLayer()

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # ENSEMBLE LEARNING ==========
        args = (
            self.rlnet(user_idx, item_idx),
            self.mlnet(user_idx, item_idx),
        )
        # ENSEMBLE AGGREGATION ==========
        X_pred = self.fusion(*args)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        # INTERACTION MODELING ==========
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION ==========
        logit = self.prediction(X_pred)
        return logit