import torch
from torch import nn
from torch.nn import init


class SEAttention1D(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):  
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d): 
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _ = x.size() 
        y = self.avg_pool(x).view(b, c)  
        y = self.fc(y).view(b, c, 1)  
        return x * y.expand_as(x)  



if __name__ == '__main__':
    input = torch.randn(50, 512, 7)  
    se = SEAttention1D(channel=512, reduction=8)
    output = se(input)
    print(output.shape)  
