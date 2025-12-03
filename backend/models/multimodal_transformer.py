import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        attn_out, _ = self.attn(q, k, v)
        x = self.ln1(q + attn_out)
        ff_out = self.ff(x)
        return self.ln2(x + ff_out)


class MultimodalEncoder(nn.Module):
    """
    Multimodal transformer encoder for:
    - Satellite LAI/NDVI tokens
    - Weather tokens
    - Landuse context
    - Text embeddings (FAO, social, plant profile)
    """

    def __init__(self, token_dim: int, text_dim: int = 64, latent_dim: int = 256, num_latents: int = 128, num_layers: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.token_proj = nn.Linear(token_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)

        self.cross = CrossAttentionBlock(latent_dim, num_heads=8)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=8,
                    dim_feedforward=latent_dim * 4,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.output_head = nn.Linear(latent_dim, 256)

    def forward(self, tokens, text_sequence, plant_profile):
        """
        tokens: (B, N, token_dim)
        text_sequence: (B, T, text_dim)  e.g. fused FAO+social
        plant_profile: (B, text_dim)
        """
        B, N, _ = tokens.shape
        _, T, _ = text_sequence.shape

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        tok = self.token_proj(tokens)
        text_tok = self.text_proj(text_sequence)  # (B, T, latent_dim)

        # Pool text tokens and add plant profile as a special token
        text_pooled = text_tok.mean(dim=1, keepdim=True)  # (B,1,latent_dim)
        plant_tok = self.text_proj(plant_profile).unsqueeze(1)  # (B,1,latent_dim)
        all_text = torch.cat([text_tok, text_pooled, plant_tok], dim=1)  # (B, T+2, latent_dim)

        latents = self.cross(latents, torch.cat([tok, all_text], dim=1), torch.cat([tok, all_text], dim=1))

        for layer in self.layers:
            latents = layer(latents)

        pooled = latents.mean(dim=1)
        return self.output_head(pooled)
