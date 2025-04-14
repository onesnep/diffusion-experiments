import torch
import torch.nn as nn
import torch.optim as optim

#import torch.nn.functional as F
import yaml
from types import SimpleNamespace
import argparse
from tqdm import tqdm
import os


from models.unet import UNet3Layer
from diffusion.schedule import ForwardDiffusionProcess
from data.mnist import get_dataloaders
from sampling.ddpm import generate_images
from utils.general_utils import pad_output_to_target

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        self.model = self._initialize_model().to(device)
        # Use Torch compile if set
        if self.config.compile_model:
             self.model = torch.compile(model)        
        self.optimizer = self._initialize_optimizer()
        self.criterion = self._initialize_criterion()
        self.fdp = self._initialize_diffusion_process().to(self.device)
        self.train_loader, self.val_loader = self._get_dataloaders()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)
        self.fixed_noise = self._generate_fixed_noise().to(self.device)
        self.current_epoch = 0
        # Potentially add LR scheduler, logging setup (W&B, Tensorboard)

    def _setup_device(self):
        """Determines the appropriate device (GPU or CPU) and returns it."""
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU.")
            device = torch.device('cuda')
        else:
            print("CUDA not available. Using CPU.")
            device = torch.device('cpu')
        return device

    def _initialize_model(self):
        """Init UNet3Layer based on config"""
        model = UNet3Layer(
            in_channels=self.config.model_in_channels, 
            out_channels=self.config.model_out_channels, 
            time_projection_dim=self.config.model_time_proj_dim,
            num_groups=self.config.model_num_groups
        )
        return model

    def _initialize_optimizer(self):
        """Init chosen optimizer based on config (pass self.model.parameters())"""
        
        optimizer_map = {
            'AdamW': optim.AdamW,
            'Adam': optim.Adam,
            'SGD': optim.SGD,
        }
        
        optimizer_name = self.config.optimizer_name
        
        optimizer_class = optimizer_map.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer specified in config: {optimizer_name}. "
                             f"Available options are: {list(optimizer_map.keys())}")

        # Get optimizer-specific parameters from config
        # Assumes config has a dictionary like 'optimizer_params'
        try:
            optimizer_kwargs = dict(self.config.optimizer_params) # Create a mutable copy
        except AttributeError:
            raise ValueError("Configuration object must have an 'optimizer_params' attribute containing a dictionary of optimizer arguments (e.g., lr, weight_decay).")

        # Ensure 'lr' is present, as it's fundamental (optional check)
        if 'lr' not in optimizer_kwargs:
             print("Warning: 'lr' (learning rate) not found in config.optimizer_params. Optimizer might use default.")

        # Instantiate the optimizer
        try:
            optimizer = optimizer_class(
                self.model.parameters(),
                **optimizer_kwargs # Unpack the dictionary as keyword arguments
            )
            print(f"Initialized {optimizer_name} optimizer with params: {optimizer_kwargs}")
        except TypeError as e:
            raise ValueError(f"Error initializing {optimizer_name}. "
                             f"Check if config.optimizer_params {optimizer_kwargs} "
                             f"match the arguments for {optimizer_class.__name__}. Original error: {e}")

        # 6. Return the optimizer instance
        return optimizer

    def _initialize_criterion(self):
        # Init MSELoss
        criterion = nn.MSELoss()
        return criterion

    def _initialize_diffusion_process(self):
        # Init ForwardDiffusionProcess based on config (timesteps)
        timesteps = self.config.sampler_timesteps
        fdp = ForwardDiffusionProcess(timesteps).to(device)

    def _get_dataloaders(self):
        if self.config.dataset_name == 'mnist':
            factory = MNISTLoaderFactory(self.config)
        elif self.config.dataset_name == 'cifar10':
            factory = CIFARLoaderFactory(self.config)
        else:
            raise ValueError("Unsupported dataset")

        self.train_loader, self.val_loader = factory.get_dataloaders()
    
    def _generate_fixed_noise(self):
        # Generate fixed noise tensor for sampling visualization
        # Shape: [num_samples, channels, height, width]
        self.fixed_initial_noise = torch.randn(
            self.config.num_samples,
            self.config.model_in_channels,
            self.config.sample_image_height,
            self.config.sample_image_width,
            device=self.device
        )

    def train(self):
        print("Starting Training...")
        for epoch in range(self.config.training_epochs):
            self.current_epoch = epoch
            
            print("-" * 30)
            print(f"Epoch {epoch+1}/{EPOCHS}")
            
            train_loss = self.train_epoch()
            val_loss = self._validate_epoch()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Valid Loss: {val_loss:.4f}")
            print("-" * 30)

        print("Training Finished.")

    def _train_epoch(self):
        self.model.train()
        train_running_loss = 0
        train_iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.training_epochs} [Training]")
        for idx, batch in enumerate(train_iterator):
            # Get clean images and move to device
            x_0 = batch['pixel_values'].to(self.device)
            current_batch_size = x_0.shape[0] # Get actual batch size

            # Sample random timesteps t for the batch
            t = torch.randint(0, self.config.timesteps, (current_batch_size,), device=self.device).long()

            # Apply diffusion forward process (noising) to get x_t and epsilon
            x_t, epsilon = self.fdp(x_0, t) # Calls the forward method of fdp

            self.optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=torch.float16): # Cast ops to FP16/BF16
                # Predict noise using the U-Net model and pad
                predicted_noise = self.model(x_t, t)
                padded_predicted_noise = pad_output_to_target(predicted_noise, epsilon)
            
                # Calculate loss between predicted noise and actual noise
                loss = self.criterion(padded_predicted_noise, epsilon)

            # Backpropagation and Optimization
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            # Accumulate loss
            train_running_loss += loss.item()

            # Update tqdm progress bar description
            train_iterator.set_postfix(loss=loss.item())

        # Calculate average training loss for the epoch
        train_loss = train_running_loss / len(self.train_loader)
        return train_loss

    def _validate_epoch(self):
        self.model.eval()
        val_running_loss = 0
        val_iterator = tqdm(self.val_dataloader, desc=f"Epoch {self.current_epoch+1}/{self.config.training_epochs} [Validation]", leave=False)
        with torch.no_grad():
            for batch in val_iterator:
                x_0 = batch['pixel_values'].to(device)
                current_batch_size = x_0.shape[0]
                t = torch.randint(0, T, (current_batch_size,), device=device).long()
                x_t, epsilon = self.fdp(x_0, t)

                with torch.autocast(device_type=device, dtype=torch.float16): # Cast ops to FP16/BF16
                    predicted_noise = self.model(x_t, t)
                    padded_predicted_noise = pad_output_to_target(predicted_noise, epsilon)
                    loss = self.criterion(padded_predicted_noise, epsilon)
                val_running_loss += loss.item()
                val_iterator.set_postfix(loss=loss.item())

        val_loss = val_running_loss / len(val_dataloader)
        return val_loss

    def _generate_sample(self):
        generate_samples(model=self.model,
                 fdp=self.fdp,
                 T=self.config.sampler_timesteps,
                 epoch=self.current_epoch,
                 device=self.device,
                 initial_noise=self.fixed_initial_noise,
                 num_images=self.config.num_samples,
                 save_dir=self.config.sample_image_directory) # Specify where to save samples

    def _save_model(self):
        torch.save(self.model.state_dict(), f"{self.config.output_dir}" + f"{self.config.run_name}" f"_e{str(self.current_epoch)}.pth")
        print(f"Model saved to {self.config.model_save_path}_e{str(epoch)}.pth")


if __name__==__main__:
    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
    config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)

    # Now you can access parameters like:
    # config.training['epochs'], config.model['base_dim'], config.device etc.
    # Or flatten it if you prefer config.training_epochs etc. (requires more processing)

    # Pass config to your Trainer
    trainer = Trainer(config)
    trainer.train()