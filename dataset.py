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

@dataclass
class NiftiDataset3D(Dataset):
    image_paths: List[str]
    labels: List[int]
    resize_shape: Optional[Tuple[int, int, int]] = None
    augment: bool = False
    transform: Optional[transforms.Compose] = field(default=None, init=False)

    def __post_init__(self):
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._load_and_preprocess_image(self.image_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

    def _load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        img = nib.load(img_path).get_fdata()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.resize_shape and img.shape != tuple(self.resize_shape):
            img = self._resize_image(img, self.resize_shape)

        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img

    def _resize_image(self, img: np.ndarray, resize_shape: Tuple[int, int, int]) -> np.ndarray:
        zoom_factors = [resize_shape[i] / img.shape[i] for i in range(3)]
        return zoom(img, zoom_factors, order=1)

@dataclass
class NiftiDataset2D(Dataset):
    image_paths: List[str]
    labels: List[int]
    resize_shape: Optional[Tuple[int, int]] = None
    slice_axis: str = 'axial'
    augment: bool = False
    transform: Optional[transforms.Compose] = field(default=None, init=False)
    slices_per_volume: List[int] = field(default_factory=list, init=False)
    sliced_images: List[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self):
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
            ])
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
        for img_path in self.image_paths:
            img = nib.load(img_path).get_fdata()
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            slices = self._get_slices(img)
            if self.resize_shape:
                slices = [self._resize_image(s, self.resize_shape) for s in slices]

            self.sliced_images.extend(np.expand_dims(s, axis=0) for s in slices)
            self.slices_per_volume.append(len(slices))

        self.labels = [label for label, count in zip(self.labels, self.slices_per_volume) for _ in range(count)]

    def _get_slices(self, img: np.ndarray) -> List[np.ndarray]:
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        axis = axis_map.get(self.slice_axis)

        if axis is None:
            raise ValueError("Invalid slice_axis, choose from 'axial', 'coronal', 'sagittal'.")

        return [np.take(img, i, axis=axis) for i in range(img.shape[axis])]

    def _resize_image(self, img: np.ndarray, resize_shape: Tuple[int, int]) -> np.ndarray:
        zoom_factors = [resize_shape[i] / img.shape[i] for i in range(2)]
        return zoom(img, zoom_factors, order=1)

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
    
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    image_paths = []
    labels = []
    class_counts = {}

    for i, class_name in enumerate(sorted(class_dirs)):
        class_path = os.path.join(data_dir, class_name)
        class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        image_paths.extend(class_images)
        labels.extend([i] * len(class_images))
        class_counts[class_name] = len(class_images)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels)

    if dataset_type == '3D':
        train_dataset = NiftiDataset3D(train_paths, train_labels, resize_shape=resize_shape, augment=augment)
        val_dataset = NiftiDataset3D(val_paths, val_labels, resize_shape=resize_shape)
        test_dataset = NiftiDataset3D(test_paths, test_labels, resize_shape=resize_shape)
    else:
        train_dataset = NiftiDataset2D(train_paths, train_labels, resize_shape=resize_shape, slice_axis=slice_axis, augment=augment)
        val_dataset = NiftiDataset2D(val_paths, val_labels, resize_shape=resize_shape, slice_axis=slice_axis)
        test_dataset = NiftiDataset2D(test_paths, test_labels, resize_shape=resize_shape, slice_axis=slice_axis)

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
