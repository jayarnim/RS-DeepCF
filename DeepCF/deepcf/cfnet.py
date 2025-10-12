import torch
import torch.nn as nn
from . import rlnet, mlnet


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden_rl: list,
        hidden_ml: list,
        dropout: float,
        interactions: torch.Tensor, 
    ):
        """
        Deepcf: A unified framework of representation learning and matching function learning in recommender system (Deng et al., 2019)
        -----
        Implements the base structure of Collaboartive Filtering Networks (CFNet),
        MF, MLP & history embedding based latent factor model,
        combining a Representation Learning Networks (RLNet) and a Matching Function Learning Networks (MLNet)
        to learn low-rank linear represenation & high-rank nonlinear user-item interactions.

        Args:
            n_users (int):
                total number of users in the dataset, U.
            n_items (int):
                total number of items in the dataset, I.
            n_factors (int):
                dimensionality of user and item latent representation vectors, K.
            hidden_rl (list):
                layer dimensions for the representation @ RLNet. 
                (e.g., [64, 32, 16, 8])
            hidden_ml (list): 
                layer dimensions for the matching function @ MLNet. 
                (e.g., [64, 32, 16, 8])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
            interaction (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+1, I+1])
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden_rl = hidden_rl
        self.hidden_ml = hidden_ml
        self.dropout = dropout
        self.register_buffer(
            name="interactions", 
            tensor=interactions,
        )

        # generate layers
        self._set_up_components()

    def forward(
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
        return self.score(user_idx, item_idx)

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
        logit = self.score(user_idx, item_idx)
        pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector = self.ensemble(user_idx, item_idx)
        logit = self.pred_layer(pred_vector).squeeze(-1)
        return logit

    def ensemble(self, user_idx, item_idx):
        # modules
        pred_vector_rl = self.rl.gmf(user_idx, item_idx)
        pred_vector_ml = self.ml.ncf(user_idx, item_idx)

        # agg
        kwargs = dict(
            tensors=(pred_vector_rl, pred_vector_ml), 
            dim=-1,
        )
        pred_vector = torch.cat(**kwargs)

        return pred_vector

    def _set_up_components(self):
        self._create_modules()
        self._create_layers()

    def _create_modules(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.rl = rlnet.Module(
            **kwargs,
            hidden=self.hidden_rl,
        )
        self.ml = mlnet.Module(
            **kwargs,
            n_factors=self.n_factors,
            hidden=self.hidden_ml,
        )

    def _create_layers(self):
        kwargs = dict(
            in_features=self.hidden_rl[-1] + self.hidden_ml[-1],
            out_features=1, 
        )
        self.pred_layer = nn.Linear(**kwargs)