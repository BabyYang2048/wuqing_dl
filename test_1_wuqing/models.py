import torch
import torch.nn as nn


class BaseClassification(nn.Module):

    def __init__(self,vocab_size,hidden_dim,mode=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,hidden_dim)  #vocab_size个词，每个词维度为hidden_dim维
        self.drop = nn.Dropout(0.3)
        self.mode = mode.lower()
        if mode.lower() =='lstm':
            self.encode_layer = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        elif mode.lower() == "gru":
            self.encode_layer = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        else:
            self.encode_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),  # 全连接
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim * 2)  # 全连接（维度变化）
            )
        self.predict_layer = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, 2),
        )

    def forward(self, x1, x2):
        # print(x1, x2)
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = x1 - x2
        y = x1 + x2
        x1 = self.drop(x1)
        x2 = self.drop(x2)
        x1, x2, x = self.encode_layer(x1), self.encode_layer(x2), self.encode_layer(x)
        if self.mode in ['lstm', 'gru']:
            x1, x2, x, y = x1[0], x2[0], x[0], y[0]
        if self.mode != 'cnn':
            x1, x2, x = x1.mean(dim=1).squeeze(), x2.mean(dim=1).squeeze(), x.mean(dim=1).squeeze()
        # final_enc = torch.cat([x1, x2, x, y], dim=-1)
        final_enc = torch.cat([x1, x2, x], dim=-1)
        return self.predict_layer(final_enc)
