import torch
from torch import nn
from torch.nn import init


class SEAttention1D(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 1D自适应平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):  # 改为1D卷积判断
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):  # 改为1D批归一化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _ = x.size()  # 输入形状 [B, C, L]
        y = self.avg_pool(x).view(b, c)  # 池化后形状 [B, C]
        y = self.fc(y).view(b, c, 1)  # 调整形状为 [B, C, 1]
        return x * y.expand_as(x)  # 广播相乘


# 示例使用
if __name__ == '__main__':
    input = torch.randn(50, 512, 7)  # 1D输入 [B, C, L]
    se = SEAttention1D(channel=512, reduction=8)
    output = se(input)
    print(output.shape)  # 应保持输入形状 [50, 512, 7]
