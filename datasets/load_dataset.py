from abc import ABC, abstractmethod
from datasets import load_dataset


class BaseDataLoaderFactory(ABC):
    def __init__(self, config):
        self.config = config
        self.train_dataset, self.val_dataset = self._load_raw_datasets()
        self._apply_transforms()

    @abstractmethod
    def _load_raw_datasets(self):
        # Implementation specific to dataset type
        pass

    @abstractmethod
    def _get_preprocessing_transform(self):
        # Return the specific torchvision transform
        pass

    @abstractmethod
    def _create_transform_batch_function(self):
        # Return the function used for set_transform or collate_fn
        pass

    def _apply_transforms(self):
        transform_fn = self._create_transform_batch_function()
        if self.train_dataset:
            self.train_dataset.set_transform(transform_fn)
        if self.val_dataset:
            self.val_dataset.set_transform(transform_fn)

    def get_dataloaders(self):
        # Common logic to create DataLoader instances
        # (Can be overridden if dataset needs specific sampler etc.)
        train_loader = DataLoader(self.train_dataset, ...)
        val_loader = DataLoader(self.val_dataset, ...)
        return train_loader, val_loader

def transform_batch(batch):
    """Applies preprocessing to a batch of images."""
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    batch["image"] = [preprocess(img) for img in batch["image"]]
    return {"pixel_values": batch["image"]}

def get_dataloaders_hf(config):
    """Load dataset from huggingface"""

    print(f"Loading dataset: {config.dataset_id}")
    train_dataset = load_dataset(HF_DATASET_ID, split='train', streaming=False)
    val_dataset = load_dataset(HF_DATASET_ID, split='test', streaming=False)
    print("Datasets loaded.")

    # Apply transform (before dataloader)
    train_dataset.set_transform(transform_batch)
    val_dataset.set_transform(transform_batch)

    # Dataloader would now yield batches like {'pixel_values': tensor}

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=8,  # Example: Start with 4
                                pin_memory=True) # Use with GPU and num_workers > 0
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=8,  # Example: Start with 4
                                pin_memory=True) # Use with GPU and num_workers > 0