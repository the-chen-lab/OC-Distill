import torch
import torch.nn as nn
from transformers import AutoModel


class VitalTransformer(nn.Module):
    """Transformer encoder for vital sign time series."""
    def __init__(self, input_dim=24, hidden_dim=768, n_heads=4, n_layers=2, 
                 dropout=0.3, output_dim=768, max_len=10000):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, T, _ = x.size()
        x = self.input_proj(x)
        pos = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(0)
        x = x + pos

        x = x.permute(1, 0, 2)  # [T, B, H]
        x_encoded = self.encoder(x)
        x_encoded = x_encoded.permute(1, 0, 2)  # [B, T, H]

        pooled = x_encoded.mean(dim=1)  # Mean pooling over time
        return self.proj(pooled)


class RiskPredictor(nn.Module):
    """MLP head for downstream risk prediction tasks."""
    def __init__(self, input_dim=768, num_labels=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.fc(x)


class BioClinicalBERT(nn.Module):
    """Clinical BERT encoder for clinical notes."""
    def __init__(self, model_path="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, owners):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0, :]

        B = int(owners.max().item()) + 1
        sum_emb = h_cls.new_zeros((B, 768))
        sum_emb.index_add_(0, owners, h_cls)

        cnt = h_cls.new_zeros((B, 1))
        cnt.index_add_(0, owners, torch.ones_like(owners, dtype=torch.float32).unsqueeze(1))
        pooled = sum_emb / cnt

        return pooled


class ProjectionHead(nn.Module):
    """Non-linear projection head for contrastive learning."""
    def __init__(self, input_dim=768, proj_dim=128, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)
