import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        interactions: torch.Tensor, 
    ):
        super(Module, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.interactions = interactions.to(self.device)

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self._score(user_idx, item_idx)

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
            _, logit = self._score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def _score(self, user_idx, item_idx):
        # user, item proj & bttn
        proj_user = self._user(user_idx, item_idx)
        proj_item = self._item(user_idx, item_idx)

        # matching function learning
        concat = torch.cat(
            tensors=(proj_user, proj_item), 
            dim=-1
        )
        pred_vector = self.mlp(concat)

        # logit
        logit = self.logit_layer(pred_vector).squeeze(-1)

        return pred_vector, logit

    def _user(self, user_idx, item_idx):
        # get user vector from interactions
        user_slice = self.interactions[user_idx, :-1].clone()
        
        # masking target items
        user_batch = torch.arange(user_idx.size(0))
        user_slice[user_batch, item_idx] = 0
        
        # projection
        proj_user = self.proj_u(user_slice.float())

        return proj_user

    def _item(self, user_idx, item_idx):
        # get item vector from interactions
        item_slice = self.interactions.T[item_idx, :-1].clone()
        
        # masking target users
        item_batch = torch.arange(item_idx.size(0))
        item_slice[item_batch, user_idx] = 0
        
        # projection
        proj_item = self.proj_i(item_slice.float())

        return proj_item

    def _init_layers(self):
        self.proj_u = nn.Linear(
            in_features=self.n_items,
            out_features=self.n_factors,
        )
        self.proj_i = nn.Linear(
            in_features=self.n_users,
            out_features=self.n_factors,
        )
        self.mlp = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )
        self.logit_layer = nn.Linear(
            in_features=self.hidden[-1],
            out_features=1,
        )

    def _generate_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 2)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_factors * 2}"
        assert CONDITION, ERROR_MESSAGE