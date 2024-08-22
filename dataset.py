import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import torchvision.transforms as transforms

class NiftiDataset3D(Dataset):
    def __init__(self, image_paths, labels, resize_shape=None, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.resize_shape = resize_shape
        self.augment = augment
        if augment:
            self.transform = transform or transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = nib.load(img_path).get_fdata()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.resize_shape is not None:
            if img.shape != tuple(self.resize_shape):
                img = self.resize_image(img, self.resize_shape)

        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(torch.tensor(img, dtype=torch.float32))

        return img, torch.tensor(label, dtype=torch.long)

    def resize_image(self, img, resize_shape):
        zoom_factors = [resize_shape[i] / img.shape[i] for i in range(3)]
        return zoom(img, zoom_factors, order=1)

class NiftiDataset2D(Dataset):
    def __init__(self, image_paths, labels, resize_shape=None, transform=None, slice_axis='axial', augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.resize_shape = resize_shape
        self.augment = augment
        if augment:
            self.transform = transform or transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transform
        self.slice_axis = slice_axis.lower()  # 'axial', 'coronal', or 'sagittal'
        self.slices_per_volume = []
        self.sliced_images = []
        self.prepare_slices()

    def prepare_slices(self):
        for img_path in self.image_paths:
            img = nib.load(img_path).get_fdata()
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            if self.slice_axis == 'axial':
                slices = [img[:, :, i] for i in range(img.shape[2])]
            elif self.slice_axis == 'coronal':
                slices = [img[:, i, :] for i in range(img.shape[1])]
            elif self.slice_axis == 'sagittal':
                slices = [img[i, :, :] for i in range(img.shape[0])]
            else:
                raise ValueError("Invalid slice_axis, choose from 'axial', 'coronal', 'sagittal'.")

            if self.resize_shape is not None:
                slices = [self.resize_image(s, self.resize_shape[:2]) for s in slices]

            slices = [np.expand_dims(s, axis=0) for s in slices]
            self.sliced_images.extend(slices)
            self.slices_per_volume.append(len(slices))

        self.labels = [label for label, count in zip(self.labels, self.slices_per_volume) for _ in range(count)]

    def __len__(self):
        return len(self.sliced_images)

    def __getitem__(self, idx):
        img_slice = self.sliced_images[idx]
        label = self.labels[idx]

        if self.transform:
            img_slice = self.transform(torch.tensor(img_slice, dtype=torch.float32))

        return img_slice, torch.tensor(label, dtype=torch.long)

    def resize_image(self, img, resize_shape):
        zoom_factors = [resize_shape[i] / img.shape[i] for i in range(2)]
        return zoom(img, zoom_factors, order=1)

def prepare_data(data_dir, test_size=0.2, val_size=0.1, batch_size=4, resize_shape=None, shuffle=True, num_workers=0, return_loaders=True, dataset_type='3D', slice_axis='axial', class_names=None, augment=False):
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

    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=test_size, random_state=42, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels)

    if dataset_type == '3D':
        train_dataset = NiftiDataset3D(train_paths, train_labels, resize_shape=resize_shape, augment=augment)
        val_dataset = NiftiDataset3D(val_paths, val_labels, resize_shape=resize_shape)
        test_dataset = NiftiDataset3D(test_paths, test_labels, resize_shape=resize_shape)
    else:
        train_dataset = NiftiDataset2D(train_paths, train_labels, resize_shape=resize_shape, slice_axis=slice_axis, augment=augment)
        val_dataset = NiftiDataset2D(val_paths, val_labels, resize_shape=resize_shape, slice_axis=slice_axis)
        test_dataset = NiftiDataset2D(test_paths, test_labels, resize_shape=resize_shape, slice_axis=slice_axis)

    print(f"Number of images per class in training set:")
    unique_labels, counts = np.unique(train_dataset.labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"{class_names[label]}: {count}")

    if return_loaders:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader
    else:
        return train_dataset, val_dataset, test_dataset
