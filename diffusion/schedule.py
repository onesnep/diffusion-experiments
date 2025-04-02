import torch
import torch.nn as nn

class ForwardDiffusionProcess(nn.Module): # Renamed class slightly for convention
    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps

        # Define beta schedule
        beta = torch.linspace(beta_start, beta_end, timesteps)

        # Precompute required values (alphas, etc.)
        alpha = 1. - beta
        alpha_cumprod = torch.cumprod(alpha, axis=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0) # Same as alpha_cumprod_shifted

        # Register schedule values as buffers
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0) and others
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1. - alpha_cumprod))
        self.register_buffer('log_one_minus_alpha_cumprod', torch.log(1. - alpha_cumprod))
        self.register_buffer('sqrt_recip_alpha_cumprod', torch.sqrt(1. / alpha_cumprod))
        self.register_buffer('sqrt_recipm1_alpha_cumprod', torch.sqrt(1. / alpha_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0) variance (used in sampling)
        # posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod) # Original
        # Clipping variance for stability
        posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)
        posterior_variance_clipped = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer('posterior_variance', posterior_variance_clipped)
        # log calculation clipped because posterior variance is clipped
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance_clipped))

        # These might also be useful for sampling mean calculation later
        self.register_buffer('posterior_mean_coef1', beta * torch.sqrt(alpha_cumprod_prev) / (1. - alpha_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alpha_cumprod_prev) * torch.sqrt(alpha) / (1. - alpha_cumprod))


    def _get_index_from_list(self, values, t, x_shape):
        """ Helper function to index values list at type t and reshape """
        batch_size = t.shape[0]
        # Use gather on the correct device
        out = values.gather(-1, t) # Removed .cpu()
        # Reshape to [batch_size, 1, 1, 1] for image broadcasting and ensure device match
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward(self, x_0, t): # Renamed to 'forward' for nn.Module convention
        """
        Takes an image x_0 and a timestep t, returns the noisy image x_t and the noise epsilon.
        Args:
            x_0 (torch.Tensor): Clean input image [B, C, H, W]
            t (torch.Tensor): Timestep indices [B]
        Returns:
            tuple(torch.Tensor, torch.Tensor): (noisy_image x_t, noise epsilon)
        """
        # 1. Sample noise epsilon
        epsilon = torch.randn_like(x_0) # Sample FRESH noise ON THE SAME DEVICE as x_0

        # 2. Get precomputed values for timestep t
        sqrt_alpha_cumprod_t = self._get_index_from_list(self.sqrt_alpha_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._get_index_from_list(self.sqrt_one_minus_alpha_cumprod, t, x_0.shape)

        # 3. Calculate noisy image x_t
        # Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        noisy_image = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * epsilon

        return noisy_image, epsilon

    @torch.no_grad()
    def predict_start_from_noise(self, x_t, t, noise):
        # Formula: x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        sqrt_recip_alpha_cumprod_t = self._get_index_from_list(self.sqrt_recip_alpha_cumprod, t, x_t.shape)
        sqrt_recipm1_alpha_cumprod_t = self._get_index_from_list(self.sqrt_recipm1_alpha_cumprod, t, x_t.shape)
        return sqrt_recip_alpha_cumprod_t * x_t - sqrt_recipm1_alpha_cumprod_t * noise

    @torch.no_grad()
    def q_posterior(self, x_start, x_t, t):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        posterior_mean_coef1_t = self._get_index_from_list(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._get_index_from_list(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t

        posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped_t = self._get_index_from_list(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t

    @torch.no_grad()
    def p_sample(self, model_output, x_t, t):
        # Compute the next step sample x_{t-1} given the model prediction (noise) and current sample x_t
        # This implements the DDPM sampling step

        # Get posterior variance
        posterior_variance_t = self._get_index_from_list(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped_t = self._get_index_from_list(self.posterior_log_variance_clipped, t, x_t.shape)

        # Predict x_0 from noise
        x_start = self.predict_start_from_noise(x_t, t, model_output)
        x_start.clamp_(-1., 1.) # Clamp predicted x_0

        # Compute posterior mean involving x_0 and x_t
        posterior_mean_coef1_t = self._get_index_from_list(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._get_index_from_list(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t

        # Sample z ~ N(0, I) for noise term, unless t=0
        noise = torch.randn_like(x_t)
        mask = (t != 0).float().reshape(x_t.shape[0], *((1,) * (len(x_t.shape) - 1))) # No noise added at t=0

        # Calculate x_{t-1} sample
        sample = posterior_mean + mask * (0.5 * posterior_log_variance_clipped_t).exp() * noise

        return sample