import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Re-defining TokenPositionalEmbedding
class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(block_size, d_model)
        self.dropout     = nn.Dropout(0.1)  # GPT-2 uses dropout after embedding

    def forward(self, x):
        B, T = x.size()  # (batch_size, sequence_length)
        # sanity check
        assert T <= self.pos_embed.num_embeddings, (
            f"Sequence length T={T} exceeds block_size={self.pos_embed.num_embeddings}"
        )
        tok_emb = self.token_embed(x)                    # (B, T, d_model)
        pos_ids = torch.arange(T, device=x.device)       # (T,)
        pos_emb = self.pos_embed(pos_ids)[None, :, :]    # (1, T, d_model)
        out = tok_emb + pos_emb                          # (B, T, d_model)
        return self.dropout(out)

# Re-defining CausalSelfAttention
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # Output projection (projects concatenated heads back to d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_rate)

        # The mask will be created in forward based on input shape
        # self.register_buffer(
        #     "mask",
        #     torch.tril(torch.ones((1, 1, 512, 512), dtype=torch.bool)),  # 512 = max block_size; adjust if you use different
        #     persistent=False
        # )

    def forward(self, x):
        """
        x: (B, T, d_model)  batch of embeddings
        returns: (B, T, d_model) same shape, after self-attention
        """
        B, T, C = x.size() # C = d_model
        assert C == self.n_heads * self.d_head

        # 1. project to queries, keys, values and reshape for multi-head
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, dh)

        # 2. compute scaled dot-product attention scores
        #    q @ k^T : (B, nh, T, dh) @ (B, nh, dh, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # 3. apply causal mask: prevent attending to future positions
        #    Create mask based on current sequence length T
        causal_mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool))[None, None, :, :] # (1, 1, T, T)
        att = att.masked_fill(~causal_mask, float('-inf'))

        # 4. softmax and dropout
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        # 5. attention output weighted sum
        #    (B, nh, T, T) @ (B, nh, T, dh) -> (B, nh, T, dh)
        out = att @ v

        # 6. combine heads and final projection
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        out = self.out_proj(out)
        return self.dropout(out)

# Re-defining FeedForward
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)

# Re-defining TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout_rate=0.1): # Corrected to accept dropout_rate
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout_rate) # Pass dropout_rate
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout_rate) # Pass dropout_rate

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# Re-defining GPTConfig
class GPTConfig:
    """Configuration for the GPT model."""
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.dropout    = dropout

# Re-defining GPT
class GPT(nn.Module):
    """
    GPT‐2–style model:
      • Token + Positional Embeddings
      • N Transformer blocks (pre-LN, causal self-attn, FFN)
      • Final LayerNorm
      • Tied LM Head
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 1) Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed   = nn.Embedding(config.block_size, config.d_model)
        self.dropout     = nn.Dropout(config.dropout)

        # self.token_pos_embed = TokenPositionalEmbedding(vocab_size, d_model, block_size)

        # 2) Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                block_size=config.block_size,
                dropout_rate=config.dropout # Pass dropout_rate here
            )
            for _ in range(config.n_layers)
        ])

        # 3) Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # 4) Language‑model head (tied to token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.token_embed.weight

        # ensure everything is initialized properly
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(self, idx: torch.LongTensor) -> torch.FloatTensor:
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        token_embeddings = self.token_embed(idx)
        positions = torch.arange(T, device=idx.device)
        pos_embeddings = self.pos_embed(positions)
        x = token_embeddings + pos_embeddings.unsqueeze(0)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] # Use config.block_size
            logits = self(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# --- Model Instantiation ---
# Make sure device is defined (e.g., device = 'cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define the configuration parameters (these must match the saved model)
vocab_size   = 50257
block_size   = 128
d_model      = 256
n_heads      = 4
n_layers     = 4
dropout_rate = 0.1

# Create a GPTConfig instance
cfg = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout_rate
)

if __name__ == "__main__":
    # Instantiate the GPT model
    model = GPT(cfg).to(device)

    print("GPT model instantiated successfully.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,} trainable parameters")