from torchvision import transforms
from datasets import load_dataset

class MNISTLoaderFactory(BaseDataLoaderFactory):
    def _load_raw_datasets(self):
        train_ds = load_dataset(self.config.dataset_id, split='train', streaming=self.config.use_streaming)
        val_ds = load_dataset(self.config.dataset_id, split='test', streaming=self.config.use_streaming)
        return train_ds, val_ds

    def _get_preprocessing_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def _create_transform_batch_function(self):
        preprocess_t = self._get_preprocessing_transform()
        image_key = 'image' # Or get from config?
        def transform_batch(batch):
            batch[image_key] = [preprocess_t(img) for img in batch[image_key]]
            return {"pixel_values": batch[image_key]}
        return transform_batch

class CIFARLoaderFactory(BaseDataLoaderFactory):
     def _load_raw_datasets(self):
         train_ds = load_dataset('cifar10', split='train', ...)
         val_ds = load_dataset('cifar10', split='test', ...)
         return train_ds, val_ds

     def _get_preprocessing_transform(self):
         # CIFAR specific transforms (e.g., 3 channels, different normalization)
         return transforms.Compose([...])

     def _create_transform_batch_function(self):
         # ... using 'img' key for CIFAR ...
         pass