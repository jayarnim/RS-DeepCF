import torch
import torch.nn as nn
from .components.embedding import HistoryEmbedding
from .components.matching.builder import matching_fn_builder
from .components.prediction import ProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        interactions: torch.Tensor, 
        num_users: int,
        num_items: int,
        projection_dim: int,
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
            interactions (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+1, I+1])
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            projection_dim (int): 
                dimensionality of user and item projection vectors.
            hidden_dim (list):
                layer dimensions for the matching function. 
                (e.g., [32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pred_dim = hidden_dim[-1]

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb, item_emb = self.embedding(user_idx, item_idx)
        X_pred = self.matching(user_emb, item_emb)
        return X_pred

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
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
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
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            interactions=self.interactions, 
            num_users=self.num_users,
            num_items=self.num_items,
            projection_dim=self.projection_dim,
        )
        self.embedding = HistoryEmbedding(**kwargs)

        kwargs = dict(
            name="ncf",
            input_dim=self.projection_dim*2,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.matching = matching_fn_builder(**kwargs)

        kwargs = dict(
            dim=self.hidden_dim[-1],
        )
        self.prediction = ProjectionLayer(**kwargs)