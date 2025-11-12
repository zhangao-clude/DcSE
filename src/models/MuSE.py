import numpy as np
import torch.nn as nn
import torch

pretrain_embeddings = torch.tensor(np.load('embedding_matrix.npy'))


class Onehot(nn.Module):
    def __init__(self):
        super(Onehot, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.pool1 = nn.MaxPool1d(2)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.pool2 = nn.MaxPool1d(5)
        self.cnn3 = nn.Conv1d(32, 32, 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.pool1(x)
        x = self.cnn2(x)
        x = self.pool2(x)
        x = self.cnn3(x)
        return x


class MuSE(nn.Module):
    def __init__(self):
        super(MuSE, self).__init__()
        self.emb = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=True)  

        self.cnn = nn.Conv1d(1, 32, 1)
        self.cnn1 = nn.Conv1d(100, 64, 41, padding=20)
        self.pool1 = nn.MaxPool1d(5)
        self.cnn2 = nn.Conv1d(64, 32, 41, padding=20)
        self.pool2 = nn.MaxPool1d(2)
        self.cnn3 = nn.Conv2d(32, 32, 2)
        self.pool3 = nn.MaxPool1d(3)
        self.onehot_cnn = Onehot()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3168, 640)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(640, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2, x3):
        x = x1
        x = x.to(torch.long)
        x = self.emb(x)  # 16 3000 100
        x = x.float()
        x3 = x3.float()
        x = x.permute(0, 2, 1)

        x = self.cnn1(x)
        x = self.pool1(x)  # (32, 64, 600)
        x = self.cnn2(x)
        x = self.pool2(x)  # (32, 32, 300)

        x2 = self.cnn(x2)  # 16 32 300
        x3 = self.onehot_cnn(x3)  # 16 32 299
        x = torch.stack((x, x2)).permute(1, 2, 3, 0)  # 16 32 300 2
        x = self.cnn3(x).squeeze()

        x = torch.stack((x, x3)).permute(1, 2, 3, 0)

        x = self.cnn3(x).squeeze()
        x = self.pool3(x)
        out = self.flatten(x)
        x = self.flatten(x)
        x = self.linear1(x)

        x = self.relu(self.dropout(x))

        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x, out


if __name__ == '__main__':
    embedding_matrix = torch.tensor(np.load('embedding_matrix.npy'))
    batch_size = 16
    x1 = torch.randint(0, embedding_matrix.shape[0], (batch_size, 3000)) 
    x2 = torch.randn(batch_size, 1, 300)  
    x3 = torch.randn(batch_size, 3000, 4) 


    model = MuSE()


    output, features = model(x1, x2, x3)

    print("Classification Output (Softmax):", output)


    print("Feature Vector (Before Softmax):", features)
