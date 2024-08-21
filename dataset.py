import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

class NiftiDataset(Dataset):
    def __init__(self, image_paths, labels, resize_shape=(128, 128, 128), transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.resize_shape = resize_shape
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = nib.load(img_path).get_fdata()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        if self.resize_shape is not None:
            img = self.resize_image(img, self.resize_shape)
        img = np.expand_dims(img, axis=0)

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def resize_image(self, img, resize_shape):
        zoom_factors = [resize_shape[i] / img.shape[i] for i in range(3)]
        return zoom(img, zoom_factors, order=1)

def prepare_data(data_dir, test_size=0.2, val_size=0.1, batch_size=4, resize_shape=(128, 128, 128), shuffle=True, num_workers=0, return_loaders=True):
    ad_path = os.path.join(data_dir, "AD")
    nc_path = os.path.join(data_dir, "NC")

    ad_images = [os.path.join(ad_path, img) for img in os.listdir(ad_path)]
    nc_images = [os.path.join(nc_path, img) for img in os.listdir(nc_path)]

    ad_labels = [0] * len(ad_images)
    nc_labels = [1] * len(nc_images)

    image_paths = ad_images + nc_images
    labels = ad_labels + nc_labels

    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=test_size, random_state=42, stratify=labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels)

    train_dataset = NiftiDataset(train_paths, train_labels, resize_shape=resize_shape)
    val_dataset = NiftiDataset(val_paths, val_labels, resize_shape=resize_shape)
    test_dataset = NiftiDataset(test_paths, test_labels, resize_shape=resize_shape)

    if return_loaders:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader
    else:
        return train_dataset, val_dataset, test_dataset
