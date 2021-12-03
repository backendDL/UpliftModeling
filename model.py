from typing import List, Tuple, Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(
        self, 
        in_features: int = 7, 
        out_features: int = 16, 
        filter_widths: List[int] = [3, 3, 3, 3], 
        dropout: float = 0.25, 
        channels: int = 1024, 
        num_hidden_nodes: int = 16,
        momentum: float = 0.1,
    ):
        super(TCN, self).__init__()

        l = sum([(3 ** (i - 1)) * 2 for i in range(1, len(filter_widths) + 1)])
        
        # Validate input
        # for fw in filter_widths:
            # assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.in_features = in_features
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.layers_to_dense = nn.Sequential(
            nn.Linear(self.in_features, num_hidden_nodes),
            nn.BatchNorm1d(num_hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden_nodes, num_hidden_nodes),
            nn.BatchNorm1d(num_hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.pad = []
        self.causal_shift = []
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, out_features, 1)

        self.expand_conv = nn.Conv1d(num_hidden_nodes, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        layers_agg = []
        
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append(filter_widths[i]//2 * next_dilation)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], dilation=next_dilation, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=momentum))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=momentum))
            
            next_dilation *= filter_widths[i]

        layers_agg.append(nn.Linear(128 - l, 1))
        layers_agg.append(nn.BatchNorm1d(1, momentum=momentum))

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_agg = nn.Sequential(*layers_agg)

    def forward(self, x):
        # assert len(x.shape) == 3 # (N, Din, Tin)
        # assert x.shape[1] == self.in_features
        N, D_in, T_in = x.shape
        x = x.permute(0, 2, 1).reshape(-1, D_in)
        x = self.layers_to_dense(x).view(N, T_in, 8).permute(0, 2, 1)

        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x)))) # (N, C, Tin - 2)
        
        for i in range(len(self.pad)):
            pad = self.pad[i]
            shift = self.causal_shift[i]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift] # shift는 res에만 관여한다.
            
            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))
        
        x = self.drop(self.relu(self.shrink(x))) # (N, Dout, Tout)
        x = self.drop(self.relu(self.layers_agg(x.view(-1, x.shape[-1])).view(x.shape[0], x.shape[1])))
        return x

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum 
            # TODO: 기존 코드에서 decay 코드 활용하기 (메서드로 삽입)

class BaselineModel(nn.Module):
    def __init__(
        self, 
        in_features: int=7, 
        out_features: int=16, 
        filter_widths: int=[3, 3, 3, 3], 
        dropout: int=0.25, 
        channels: int=1024, 
        num_hidden_nodes: int=16
    ):
        super(BaselineModel, self).__init__()
        
        self.embedding = TCN(in_features, out_features, filter_widths, dropout, channels, 8)
        # embeds time-series log data with TCN

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_features, num_hidden_nodes),
            nn.BatchNorm1d(num_hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden_nodes, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.classifier(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, out_features: int, num_hidden_nodes: int, dropout: float = 0.2):
        """It simply adds one more node in addition to `out_features`.
        """
        super(SimpleClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_features+1, num_hidden_nodes),
            nn.BatchNorm1d(num_hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden_nodes, 1),
            # nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.classifier(x)


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