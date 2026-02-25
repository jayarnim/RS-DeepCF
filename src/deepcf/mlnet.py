import torch
from components.interactions import Interactions
from components.base import BaseModel
from .layers.embedding import build as build_embedding_layer
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class MatchingLearningNetworks(BaseModel):
    def __init__(
        self,
        interactions: Interactions, 
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        dropout: float,
    ):
        """
        Deepcf: A unified framework of representation learning and matching function learning in recommender system (Deng et al., 2019)
        -----
        Implements the base structure of Matching Function Learning Networks (MLNet),
        MLP & history embedding based latent factor model,
        sub-module of Collaboartive Filtering Networks (CFNet)
        to learn high-rank nonlinear user-item interactions.

        Args:
            interactions (Interactions): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+2, I+2])
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int): 
                dimensionality of user and item projection vectors.
            hidden_dim (list):
                layer dimensions for the matching function. 
                (e.g., [32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__(locals())

        self.pred_dim = hidden_dim[-1]

        # USER-ITEM INTERACTION MAT. VIEWER ==========
        self.interactions = interactions

        # HISTORY EMBEDDING ==========
        self.embedding = build_embedding_layer(
            name="history",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )

        # MATCHING FUNCTION LEARNING ==========
        self.matching = build_matching_layer(
            name="ncf",
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # SEARCH USER-ITEM INTERACION MAT. ==========
        user_vec, item_vec = self.interactions(user_idx, item_idx)
        # HISTORY EMBEDDING ==========
        user_emb, item_emb = self.embedding(user_vec, item_vec)
        # MATCHING FUNCTION LEARNING ==========
        X_pred = self.matching(user_emb, item_emb)
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
