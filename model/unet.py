import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None, 
                 *, 
                 time_emb_dim=None, 
                 num_groups=32):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        # Time Embedding Projection
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, mid_channels) # Project to mid_channels
            )

        # Determine num_groups for GroupNorm, ensure it divides channels
        # Ensure num_groups divides mid_channels and out_channels cleanly if possible, or adjust.
        # For simplicity, let's ensure mid_channels and out_channels are divisible by num_groups if used.
        # A common strategy is to ensure channels >= num_groups. Clamping might be needed.
        num_groups_mid = max(1, num_groups if mid_channels >= num_groups and mid_channels % num_groups == 0 else 1)
        num_groups_out = max(1, num_groups if out_channels >= num_groups and out_channels % num_groups == 0 else 1)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups_mid, mid_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x, time_emb=None):
        """
        Args:
            x (torch.Tensor): Input features [B, C_in, H, W]
            time_emb (torch.Tensor, optional): Time embedding [B, time_emb_dim]. Defaults to None.
        """
        h = self.conv1(x)
        h = self.norm1(h)

        if self.time_mlp is not None and time_emb is not None:
            t_proj = self.time_mlp(time_emb) # [B, mid_channels]
            # Reshape t_proj to [B, mid_channels, 1, 1] for broadcasting
            t_proj = t_proj.unsqueeze(-1).unsqueeze(-1)
            h = h + t_proj # Add to features after norm, before activation

        h = self.act1(h) # Apply activation
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h) # Final activation

        return h

class DownSample(nn.Module):
    """ Downsamples the feature map using DoubleConv and MaxPool. """
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, num_groups=32):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim=time_emb_dim, num_groups=num_groups)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, time_emb=None):
        down = self.conv(x, time_emb) # Pass time_emb here
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    """ Upsamples feature map, concatenates skip connection, and uses DoubleConv. """
    def __init__(self, in_channels, out_channels, *, time_emb_dim=None, num_groups=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim=time_emb_dim, num_groups=num_groups) # Input channels = channels from upsample + skip channels

    def forward(self, x1_from_below, x2_skip, time_emb=None):
        """
        Args:
            x1_from_below: Tensor from the previous layer in the decoder (to be upsampled).
            x2_skip: Tensor from the corresponding skip connection in the encoder.
            time_emb: Time embedding tensor.
        """
        # Upsample the tensor from the layer below
        x1_upsampled = self.up(x1_from_below) # e.g., [B, C_up, 6, 6]

        # Get target spatial size from the upsampled tensor
        target_size_h = x1_upsampled.shape[2]
        target_size_w = x1_upsampled.shape[3]

        # Get skip connection size
        skip_size_h = x2_skip.shape[2]
        skip_size_w = x2_skip.shape[3]

        # Calculate cropping amounts (if skip is larger)
        # Example: Skip 7x7, Upsampled 6x6. Diff=1. Start=1//2=0. End=0+6=6. Slice [0:6]
        crop_h_start = (skip_size_h - target_size_h) // 2
        crop_h_end = crop_h_start + target_size_h
        crop_w_start = (skip_size_w - target_size_w) // 2
        crop_w_end = crop_w_start + target_size_w

        # Crop the skip connection tensor if needed
        if skip_size_h > target_size_h or skip_size_w > target_size_w:
             x2_cropped = x2_skip[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        else:
             # Optional: Handle cases where upsampled might be larger (e.g., padding)
             # If not expected, assume skip is same size or larger
             x2_cropped = x2_skip

        # Check sizes before concatenation (optional debug)
        # print(f"Concat shapes: {x1_upsampled.shape}, {x2_cropped.shape}")

        # Concatenate along the channel dimension
        x = torch.cat([x1_upsampled, x2_cropped], dim=1)

        return self.conv(x, time_emb)
        
class UNet3Layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 time_embedding_dim=256, # Example default
                 time_projection_dim=512, # Example default
                 num_groups=32):         # Example default
        super().__init__()

        self.time_mlp = TimeEmbeddingMLP(embedding_dim=time_embedding_dim, projection_dim=time_projection_dim)
        td = time_projection_dim # Shortcut for time dimension arg
        
        self.down_conv_1 = DownSample(in_channels, 64, time_emb_dim=td, num_groups=num_groups) # MNIST 28x28 -> 14x14
        self.down_conv_2 = DownSample(64, 128, time_emb_dim=td, num_groups=num_groups)         #             -> 7x7
        self.down_conv_3 = DownSample(128, 256, time_emb_dim=td, num_groups=num_groups)        #             -> 3x3 or 4x4

        self.bottle_neck = DoubleConv(256, 512, time_emb_dim=td, num_groups=num_groups) # Operates on smallest size

        self.up_conv_1 = UpSample(512, 256, time_emb_dim=td, num_groups=num_groups) # Input: bottleneck + skip3 (256) -> Output: 256 channels
        self.up_conv_2 = UpSample(256, 128, time_emb_dim=td, num_groups=num_groups) # Input: up1 + skip2 (128) -> Output: 128 channels
        self.up_conv_3 = UpSample(128, 64, time_emb_dim=td, num_groups=num_groups)  # Input: up2 + skip1 (64) -> Output: 64 channels

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        Args:
            x (torch.Tensor): Input noisy image [B, C_in, H, W]
            t (torch.Tensor): Timesteps [B]
        """
        # 1. Compute time embedding once
        time_emb = self.time_mlp(t)

        # 2. Downsampling path (pass time_emb)
        down_1, p1 = self.down_conv_1(x, time_emb)
        down_2, p2 = self.down_conv_2(p1, time_emb)
        down_3, p3 = self.down_conv_3(p2, time_emb)

        # 3. Bottleneck (pass time_emb)
        b = self.bottle_neck(p3, time_emb)

        # 4. Upsampling path (pass time_emb)
        up_1 = self.up_conv_1(b, down_3, time_emb)
        up_2 = self.up_conv_2(up_1, down_2, time_emb)
        up_3 = self.up_conv_3(up_2, down_1, time_emb)

        # 5. Final output layer
        out = self.out(up_3)
        return out