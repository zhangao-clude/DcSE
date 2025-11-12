import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from DenseNet import DenseNet1D
from CBAM import CBAMLayer
import pickle
from parms_settings import settings

args = settings()

pretrain_embeddings = torch.tensor(np.load('embedding_matrix.npy'))


class DcSE(nn.Module):
    def __init__(self):
        super(DcSE, self).__init__()
        self.embedding_en = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=False)
        self.dense = DenseNet1D(in_channels=100, growth_rate=32, layers_per_block=5, dropout_rate=0.3, num_blocks=2)

        self.merger_dense = nn.Linear(in_features=256, out_features=64)
        self.dropout = nn.Dropout(p=0.2) 
        self.bn = nn.BatchNorm1d(num_features=256)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.output_layer = nn.Linear(in_features=64, out_features=1)

    def forward(self, enhancers):
        enhancers = enhancers.to(torch.long)

        emb_en = self.embedding_en(enhancers)  # 64，3000，100

        emb_en = emb_en.float()

        enhancer_conv = self.dense(emb_en.transpose(1, 2)).transpose(1, 2)  # 64 256 1
        enhancer_conv = enhancer_conv.squeeze(1)
        merge = self.bn(enhancer_conv)
        merge_dense = F.relu(self.merger_dense(merge))
        merge_dense = self.bn1(merge_dense)
        merge_dense = self.dropout(merge_dense)  
        output = torch.sigmoid(self.output_layer(merge_dense))

        return output