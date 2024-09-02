import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import torchvision.transforms as transforms
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

def normalize_image(img: np.ndarray) -> np.ndarray:
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def resize_image(img: np.ndarray, resize_shape: Tuple[int, ...]) -> np.ndarray:
    zoom_factors = [resize_shape[i] / img.shape[i] for i in range(len(resize_shape))]
    return zoom(img, zoom_factors, order=1)

@dataclass
class NiftiDatasetBase(Dataset):
    image_paths: List[str]
    labels: List[int]
    resize_shape: Optional[Tuple[int, ...]] = None
    augment: bool = False
    transform: Optional[transforms.Compose] = field(default=None, init=False)

    def __post_init__(self):
        if self.augment:
            self.transform = self.build_transforms()
        print(f"Initialized dataset with {len(self.image_paths)} images.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._load_and_preprocess_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

    def _load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        img = nib.load(img_path).get_fdata()
        img = normalize_image(img)

        if self.resize_shape and img.shape != tuple(self.resize_shape):
            img = resize_image(img, self.resize_shape)

        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img

    def build_transforms(self) -> transforms.Compose:
        raise NotImplementedError("This method should be implemented by subclasses.")

@dataclass
class NiftiDataset3D(NiftiDatasetBase):
    def build_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
        ])

@dataclass
class NiftiDataset2D(NiftiDatasetBase):
    slice_axis: str = 'axial'
    slices_per_volume: List[int] = field(default_factory=list, init=False)
    sliced_images: List[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self):
        super().__post_init__()
        self.slice_axis = self.slice_axis.lower()
        self._prepare_slices()

    def __len__(self) -> int:
        return len(self.sliced_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_slice = torch.tensor(self.sliced_images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            img_slice = self.transform(img_slice)
        return img_slice, label

    def _prepare_slices(self):
        all_slices = []
        all_labels = []
        
        print(f"Preparing slices for {len(self.image_paths)} images...")

        for img_idx, (img_path, label) in enumerate(zip(self.image_paths, self.labels)):
            img = nib.load(img_path).get_fdata()
            img = normalize_image(img)

            slices = self._get_slices(img)
            if self.resize_shape:
                slices = [resize_image(s, self.resize_shape) for s in slices]

            if self.augment:
                slices = self.apply_deterministic_augmentation(slices)

            all_slices.extend(np.expand_dims(s, axis=0) for s in slices)
            all_labels.extend([label] * len(slices))
            
        self.sliced_images = all_slices
        self.labels = all_labels
        
        print(f"Total slices: {len(self.sliced_images)}, Total labels: {len(self.labels)}")

    def _get_slices(self, img: np.ndarray) -> List[np.ndarray]:
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        axis = axis_map.get(self.slice_axis)
        if axis is None:
            raise ValueError("Invalid slice_axis, choose from 'axial', 'coronal', 'sagittal'.")
        slices = [np.take(img, i, axis=axis) for i in range(img.shape[axis])]
        return slices

    def apply_deterministic_augmentation(self, slices: List[np.ndarray]) -> List[np.ndarray]:
        """Apply the same random augmentation to all slices."""
        transform = self.build_transforms()
        augmented_slices = []
        for s in slices:
            s = np.expand_dims(s, axis=0)
            s = transform(torch.tensor(s, dtype=torch.float32)) 
            augmented_slices.append(s.squeeze(0).numpy())
        return augmented_slices

    def build_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ])


def prepare_data(
    data_dir: str, 
    test_size: float = 0.2, 
    val_size: float = 0.1, 
    batch_size: int = 4, 
    resize_shape: Optional[Union[Tuple[int, int, int], Tuple[int, int]]] = None, 
    shuffle: bool = True, 
    num_workers: int = 0, 
    return_loaders: bool = True, 
    dataset_type: str = '3D', 
    slice_axis: str = 'axial', 
    class_names: Optional[List[str]] = None,
    augment: bool = False
) -> Union[Tuple[DataLoader, DataLoader, DataLoader], Tuple[Dataset, Dataset, Dataset]]:
    
    class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    image_paths = [os.path.join(data_dir, class_name, img) 
                   for class_name in class_dirs 
                   for img in os.listdir(os.path.join(data_dir, class_name))]
    labels = [i for i, class_name in enumerate(class_dirs) for _ in os.listdir(os.path.join(data_dir, class_name))]

    print(f"Total images: {len(image_paths)}, Total labels: {len(labels)}")

    DatasetClass = NiftiDataset3D if dataset_type == '3D' else NiftiDataset2D
    dataset_args = {'resize_shape': resize_shape, 'augment': augment}
    if dataset_type == '2D':
        dataset_args['slice_axis'] = slice_axis

    if test_size == 1.0:
        test_dataset = DatasetClass(image_paths=image_paths, labels=labels, **dataset_args)
        return None, None, DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels)

    print(f"Training set: {len(train_paths)} images, Validation set: {len(val_paths)} images, Test set: {len(test_paths)} images")

    train_dataset = DatasetClass(image_paths=train_paths, labels=train_labels, **dataset_args)
    val_dataset = DatasetClass(image_paths=val_paths, labels=val_labels, **dataset_args)
    test_dataset = DatasetClass(image_paths=test_paths, labels=test_labels, **dataset_args)

    print("Number of images per class in training set:")
    unique_labels, counts = np.unique(train_dataset.labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"{class_names[label]}: {count}")

    if return_loaders:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    return train_dataset, val_dataset, test_dataset
