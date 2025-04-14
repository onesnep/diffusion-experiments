# Diffusion Experiment Refactoring TODO

## Setup & Configuration
- [x] Create `requirements.txt` with all dependencies (torch, torchvision, datasets, tqdm, Pillow, etc.)
- [x] Create `.gitignore` file (add venv, __pycache__, runs/, etc.)
- [x] Implement configuration handling in `train.py`:
    - [ ] Option A: Use `argparse` for command-line arguments (LR, epochs, paths, batch size, T, run_id etc.)
    - [x] Option B: Create `configs/base_config.yaml` and use `argparse` + `yaml` or `hydra` to load it.
- [ ] Create output directories (`runs/`, `checkpoints/`, `samples/`) dynamically in `train.py` based on config/args (e.g., using a run ID).

## Utilities (`utils/`)
- [x] Create `utils/__init__.py`
- [x] Create `utils/general_utils.py`
- [ ] Move `pad_output_to_target` function from notebook/`train.py` draft into `utils/general_utils.py` [cite: diffusion-experiments/train.py]
- [ ] Refactor `utils/gif.py`:
    - [ ] Rename `create_gif_from_images_notebook` to e.g., `create_progress_gif`.
    - [ ] Remove notebook-specific code (`IPython.display`).
    - [ ] Make directory, run_identifier, output_filename arguments configurable (pass via args/config).

## Data Handling (`data/`)
- [x] Create `data/__init__.py`
- [ ] Create `data/mnist_dataset.py`
- [ ] Move dataset loading (`load_dataset` call), `preprocess` transform definition, and `transform_batch` function from notebook into `data/mnist_dataset.py`.
- [ ] Create a function like `get_dataloaders(batch_size, num_workers, data_root, ...)` in `data/mnist_dataset.py` that returns train/val dataloaders.

## Sampling (`sampling/`)
- [x] Create `sampling/__init__.py`
- [ ] Refactor `sampling/ddpm.py`:
    - [ ] Rename `generate_samples` to e.g., `generate_images` for clarity.
    - [ ] Add necessary imports (`tqdm`, `os`, `plt`, `torchvision`, `pad_output_to_target` from `utils.general_utils`).
    - [ ] Remove undefined variables from training scope (`epsilon`, `BATCH_Size`, `LEARNING_RATE`, `RUN_NUMBER`). Pass needed info like `epoch`, `run_id` via function arguments. [cite: diffusion-experiments/sampling/ddpm.py]
    - [ ] Make `save_dir` and image filename format depend on arguments/config (pass `run_id`, `epoch` etc.). [cite: diffusion-experiments/sampling/ddpm.py]

## Training Script (`train.py`)
- [x] **Structure:** Implement the `Trainer` class structure (or chosen functional structure with a `main` function).
- [ ] **Imports:** Add imports for `torch`, `optim`, `nn`, `tqdm`, `argparse`/`yaml`, and project modules (`UNet3Layer`, `TimeEmbeddingMLP`, `ForwardDiffusionProcess`, `get_dataloaders`, `generate_images`, `pad_output_to_target`). [cite: diffusion-experiments/train.py, diffusion-experiments/model/unet.py, diffusion-experiments/model/time_embedding.py, diffusion-experiments/diffusion/schedule.py, diffusion-experiments/sampling/ddpm.py]
- [x] **Initialization:** Move setup logic (device selection, fixed noise generation, AMP scaler init) inside `main` or `Trainer.__init__`, using args/config. [cite: diffusion-experiments/train.py]
- [x] **Component Instantiation:** Instantiate `UNet3Layer`, `ForwardDiffusionProcess`, `AdamW`, `MSELoss` using values from args/config. [cite: diffusion-experiments/train.py]
- [x] **Data Loading:** Call `get_dataloaders` from `data.mnist_dataset` in `main` or `Trainer.__init__`.
- [x] **Training/Validation Loop:** Integrate the core logic from the current `train.py` draft into the chosen structure (`Trainer._train_epoch`, `Trainer._validate_epoch` or helper functions). [cite: diffusion-experiments/train.py]
- [ ] **Helper Functions:**
    - [ ] Define `save_model` (or `_save_checkpoint` in Trainer) to save model/optimizer state using configurable paths. [cite: diffusion-experiments/train.py]
    - [ ] Ensure `pad_output_to_target` is called correctly (imported from `utils.general_utils`). [cite: diffusion-experiments/train.py]
- [ ] **Sampling Call:** Update the call to the refactored `generate_images` function (imported from `sampling.ddpm`), passing necessary arguments from config/state (model, fdp, epoch, device, fixed_noise, paths, etc.). [cite: diffusion-experiments/train.py, diffusion-experiments/sampling/ddpm.py]
- [ ] **Logging/Paths:** Ensure all print statements, save paths, etc., use args/config values (like `run_id`, configured directories).

## Standalone Sampling Script (`sample.py`) (Optional but Recommended)
- [ ] Create `sample.py`.
- [ ] Add `argparse` for arguments: checkpoint path, output dir, num images, seed, etc.
- [ ] Load model architecture (`UNet3Layer`) and load state dict from checkpoint.
- [ ] Instantiate `ForwardDiffusionProcess`.
- [ ] Call the refactored `generate_images` function from `sampling.ddpm`.

## Final Steps
- [ ] Test `train.py` execution with sample args/config.
- [ ] (Optional) Test `sample.py` execution with a saved checkpoint.
- [ ] Update `README.md` with setup and usage instructions. [cite: diffusion-experiments/README.md]