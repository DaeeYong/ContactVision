import torch.nn as nn
import torch

class FootContactTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, ff_dim, num_layers, dropout):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, embed_dim)
        self.pos_enc = nn.Parameter(nn.init.normal_(nn.Parameter(torch.zeros(1, 10000, embed_dim)), std=0.02))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(embed_dim, 4)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        x = self.input_fc(x) # (Batch, Frames, Dim: 39)
        x = x + self.pos_enc[:, :x.size(1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return (self.classifier(x))  # (Batch, Frames, 4:labels)