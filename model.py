import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features=7, out_features=16, filter_widths=[3, 3, 3, 3], dropout=0.25, channels=1024, num_hidden_nodes=16):
        super().__init__()

        l = sum([(3 ** (i - 1)) * 2 for i in range(1, len(filter_widths) + 1)])
        
        # Validate input
        # for fw in filter_widths:
            # assert fw % 2 != 0, 'Only odd filter widths are supported'
        
        self.in_features = in_features
        self.filter_widths = filter_widths
        
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.layers_to_dense = nn.Sequential(
            nn.Linear(self.in_features, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.pad = []
        self.causal_shift = []
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, out_features, 1)

        self.expand_conv = nn.Conv1d(8, channels, filter_widths[0], bias=False)
        
        layers_conv = []
        layers_bn = []
        layers_agg = []
        
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append(filter_widths[i]//2 * next_dilation)
            
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], dilation=next_dilation, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            
            next_dilation *= filter_widths[i]

        layers_agg.append(nn.Linear(128 - l, 1))
        layers_agg.append(nn.BatchNorm1d(1, momentum=0.1))

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        self.layers_agg = nn.Sequential(*layers_agg)
        self.end = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_features, num_hidden_nodes),
            nn.BatchNorm1d(num_hidden_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden_nodes, 1),
            nn.Sigmoid(),
        )
        
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
        x = self.end(x)
        return x
        
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum # TODO: 기존 코드에서 decay 코드 활용하기 (메서드로 삽입)
