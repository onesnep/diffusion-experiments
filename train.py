import torch

# Init model, optimizer, diffusion class
model = UNet3Layer(in_channels=1, out_channels=1, time_projection_dim=512).to(device) # Example time_projection_dim
model = torch.compile(model)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
fdp = ForwardDiffusionProcess(timesteps=T).to(device)

# Training loop
print("Starting Training...")
for epoch in range(EPOCHS): # Use range directly, tqdm can wrap the iterable
    model.train()
    train_running_loss = 0
    train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
    for idx, batch in enumerate(train_iterator):
        # 1. Get clean images and move to device
        #    Ensure key matches output of transform_batch
        x_0 = batch['pixel_values'].to(device)
        current_batch_size = x_0.shape[0] # Get actual batch size

        # 2. Sample random timesteps t for the batch
        t = torch.randint(0, T, (current_batch_size,), device=device).long()

        # 3. Apply diffusion forward process (noising) to get x_t and epsilon
        x_t, epsilon = fdp(x_0, t) # Calls the forward method of fdp

        optimizer.zero_grad()

        # Use torch auto mixed precision
        with torch.autocast(device_type=device, dtype=torch.float16): # Cast ops to FP16/BF16

            # 4. Predict noise using the U-Net model and pad
            predicted_noise = model(x_t, t) # Shape: [B, 1, 24, 24]
            padded_predicted_noise = pad_output_to_target(predicted_noise, epsilon)
        
            # 5. Calculate loss between predicted noise and actual noise
            loss = criterion(padded_predicted_noise, epsilon)

        # 6. Backpropagation and Optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss
        train_running_loss += loss.item()

        # Update tqdm progress bar description
        train_iterator.set_postfix(loss=loss.item())


    # Calculate average training loss for the epoch
    train_loss = train_running_loss / len(train_dataloader) # Use len(dataloader) for average

    # Save model at current epoch
    save_model(model, epoch)

    # Generate Samples
    generate_samples(model=model,
                 fdp=fdp,
                 T=T,
                 epoch=epoch,
                 device=device,
                 initial_noise=fixed_initial_noise,
                 num_images=16,
                 save_dir="/nvme/notebooks/samples") # Specify where to save samples
    
    # --- Validation Step ---
    model.eval()
    val_running_loss = 0
    val_iterator = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]", leave=False)
    with torch.no_grad():
        for batch in val_iterator:
            x_0 = batch['pixel_values'].to(device)
            current_batch_size = x_0.shape[0]
            t = torch.randint(0, T, (current_batch_size,), device=device).long()
            x_t, epsilon = fdp(x_0, t)

            with torch.autocast(device_type=device, dtype=torch.float16): # Cast ops to FP16/BF16
                predicted_noise = model(x_t, t)
                padded_predicted_noise = pad_output_to_target(predicted_noise, epsilon)
                loss = criterion(padded_predicted_noise, epsilon)
            val_running_loss += loss.item()
            val_iterator.set_postfix(loss=loss.item())

    val_loss = val_running_loss / len(val_dataloader)
    # ---------------------

    print("-" * 30)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Valid Loss: {val_loss:.4f}")
    print("-" * 30)

RUN_NUMBER += 1
print("Training Finished.")