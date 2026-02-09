import torch
import torch.nn as nn
from dynamic_tanh import DynamicTanh


class DynamicTanhEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        # 使用DynamicTanh代替LayerNorm
        self.norm1 = DynamicTanh(d_model, channels_last=True)
        self.norm2 = DynamicTanh(d_model, channels_last=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        # 自注意力分支
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # 前馈分支
        ff_output = self.ffn(src)
        src = src + self.dropout(ff_output)
        return self.norm2(src)


# 测试替换效果
if __name__ == "__main__":
    layer = DynamicTanhEncoderLayer(d_model=128, nhead=4, dim_feedforward=256)
    test_input = torch.randn(997, 3, 128)  # (batch, seq, dim)
    output = layer(test_input)

    print("输出形状:", output.shape)  # torch.Size([32, 10, 512])
    print("参数统计：",
          "alpha:", layer.norm1.alpha.item(),
          "weight mean:", layer.norm1.weight.mean().item())
