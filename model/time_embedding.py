import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Computes sinusoidal embeddings for timesteps.
    Adapted from Lucidrains' implementation and original Transformer paper.
    """
    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Embedding dimension must be even.")
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time (torch.Tensor): A tensor of shape (batch_size,) containing the timesteps.
                                Assumed to be integer values.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, dim) containing the embeddings.
        """
        device = time.device
        half_dim = self.dim // 2

        # Calculate frequencies (exponents for 10000)
        # Original formula: 10000^(2i / dim) in the denominator
        # Equivalent exponent term: -log(10000) * (2i / dim) = -log(10000) / dim * (0, 2, 4, ..., dim-2)
        # Simplified expone8124Mnt term: -log(10000) / half_dim * (0, 1, 2, ..., half_dim-1)

        embeddings = math.log(10000.0) / (half_dim -1) # Calculate the constant factor first
        # Note: Using (half_dim - 1) in denom ensures the last exponent is -log(10000)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # Shape: [half_dim]

        # Calculate arguments: time * frequencies
        # time shape: [batch_size], embeddings shape: [half_dim]
        # Unsqueeze time to [batch_size, 1] to allow broadcasting
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0) # Shape: [batch_size, half_dim]

        # Apply sin and cos and concatenate
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # Shape: [batch_size, dim]

        return embeddings

class TimeEmbeddingMLP(nn.Module):
    """
    Projects the sinusoidal time embedding into a desired dimension,
    often matching the channel counts in the U-Net blocks.
    """
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.sinusoidal_embedding = SinusoidalTimeEmbedding(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.SiLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, t):
        # 1. Get sinusoidal embedding
        sin_emb = self.sinusoidal_embedding(t) # Shape: [batch_size, embedding_dim]
        # 2. Project through MLP
        projected_emb = self.mlp(sin_emb) # Shape: [batch_size, projection_dim]
        return projected_emb