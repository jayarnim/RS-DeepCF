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
        super(Module, self).__init__()

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
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector_rl = self.rl.rl(user_idx, item_idx)
        pred_vector_ml = self.ml.ml(user_idx, item_idx)

        kwargs = dict(
            tensors=(pred_vector_rl, pred_vector_ml), 
            dim=-1,
        )
        pred_vector = torch.cat(**kwargs)

        logit = self.logit_layer(pred_vector).squeeze(-1)

        return logit

    def _set_up_components(self):
        self._create_modules()
        self._create_layers()

    def _create_modules(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            hidden=self.hidden_rl,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.rl = rlnet.Module(**kwargs)

        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden_ml,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.ml = mlnet.Module(**kwargs)

    def _create_layers(self):
        kwargs = dict(
            in_features=self.hidden_rl[-1] + self.hidden_ml[-1],
            out_features=1, 
        )
        self.logit_layer = nn.Linear(**kwargs)