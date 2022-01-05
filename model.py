from typing import List, Tuple, Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNEmbedding(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, num_layers=2, dropout=0.2, bidirectional=False):
        super(RNNEmbedding, self).__init__()
        self.rnn = nn.LSTM(
            input_size=in_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.emb = nn.Linear(
            in_features=2*hidden_size if bidirectional else hidden_size,
            out_features=out_features,
        )

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :] # last hidden state
        x = self.emb(x)
        return x


class UpliftWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
    ):
        """The output of the model must be logits with the shape of [batch_size, 1]."""
        super(UpliftWrapper, self).__init__()
        self.model = model
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
    ):
        """Calculate both Y and Uplift value"""
        # X shape: (B, N)
        if t.ndim == 2:
            t = t.squeeze()
        B = x.size(0)
        L = x.size(1)
        # print(f"input shape: {x.shape}")

        # first creating the inputs accordingly
        x_0 = torch.cat([x, torch.zeros([B, 1]).to(x.device)], dim=1)
        x_1 = torch.cat([x, torch.ones([B, 1]).to(x.device)], dim=1)

        y_0 = torch.sigmoid(self.model(x_0)).squeeze()
        y_1 = torch.sigmoid(self.model(x_1)).squeeze()

        # select true y value
        pred = torch.where(t == 1, y_1, y_0)
        
        return {
            "uplift": y_1 - y_0,
            "pred": pred,
        }



class UpliftWrapperForRNN(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        out_features: int,
    ):
        """The output of the model must be logits with the shape of [batch_size, 1]."""
        super(UpliftWrapperForRNN, self).__init__()
        self.model = model
        self.linear = nn.Linear(out_features+1, out_features+1)
        self.drop = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(out_features+1, 1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
    ):
        """Calculate both Y and Uplift value"""
        # X shape: (B, N)
        if t.ndim == 2:
            t = t.squeeze()
        B = x.size(0)

        # assuming embeddings are the same across different T's
        x = self.model(x)

        # first creating the inputs accordingly
        x_0 = torch.cat([x, torch.zeros([B, 1]).to(x.device)], dim=1)
        x_1 = torch.cat([x, torch.ones([B, 1]).to(x.device)], dim=1)

        x_0 = F.relu(self.drop(self.linear(x_0)))
        x_1 = F.relu(self.drop(self.linear(x_1)))

        y_0 = torch.sigmoid(self.classifier(x_0)).squeeze()
        y_1 = torch.sigmoid(self.classifier(x_1)).squeeze()

        # select true y value
        pred = torch.where(t == 1, y_1, y_0)
        
        return {
            "uplift": y_1 - y_0,
            "pred": pred,
        }



class DirectUpliftLoss(nn.Module):
    def __init__(
        self, 
        propensity_score: float = 0.5, 
        alpha: float = 0.5, 
        return_all: bool = False,
    ):
        """Implementation of Direct Uplift Loss

        Args:
            propensity_score: float - e(X) (default: 0.5)
            alpha: float - balance between prediction loss (regarding y) and uplift loss (regarding z)
            return_all: bool - returns `loss, (loss_uplift, loss_pred)` if return_all is True
        """
        super(DirectUpliftLoss, self).__init__()
        
        if alpha > 1 or alpha < 0:
            raise ValueError("alpha must be in [0, 1]")
        if propensity_score > 1 or propensity_score < 0:
            raise ValueError("propensity_score must be in [0, 1]")
        self.e_x = propensity_score
        self.alpha = alpha
        self.return_all = return_all

        self.loss_u = nn.MSELoss()
        self.loss_y = nn.BCELoss()

    def forward(
        self, 
        out: Dict[str, torch.Tensor], 
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        # in case y is torch.long
        y = y.to(torch.float)
        
        z = t * y / self.e_x - (1-t) * y / (1-self.e_x)
        # e_x = torch.tensor([self.e_x]).repeat(t.size(0)).to(t.device)
        # z = y * (t - e_x) / (e_x * (1 - e_x))

        loss_uplift = self.loss_u(out["uplift"], z)
        loss_pred   = self.loss_y(out["pred"], y)

        loss = (1-self.alpha) * loss_uplift + self.alpha * loss_pred

        if self.return_all:
            return loss, (loss_uplift, loss_pred)
        else:
            return loss

# TODO: Implement indirect uplift loss function
class IndirectUpliftLoss(nn.Module):
    pass