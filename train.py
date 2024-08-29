import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import argparse
from dataset import prepare_data
from convnext import ConvNeXt3D, ConvNeXt2D
from sklearn.metrics import accuracy_score
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train and validate ConvNeXt model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--device', type=int, default=0, help='CUDA device number (default: 0)')
    parser.add_argument('--save_folder', type=str, required=True, help='Directory to save model weights')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--resize_shape', type=int, nargs=3, default=None, help='Resize shape for the NIfTI images (default: None for no resizing)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the data to be used as test set (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of the training data to be used as validation set (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the data loaders (default: 4)')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for the data loaders (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--model_type', type=str, choices=['2D', '3D'], default='3D', help="Choose between '2D' and '3D' model")
    parser.add_argument('--slice_axis', type=str, choices=['axial', 'coronal', 'sagittal'], default=None, help="Axis to slice along for 2D models (only for 2D)")
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation")
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes for the model')
    parser.add_argument('--use_all_axes', action='store_true', help="Use all 2D axes (axial, coronal, sagittal) for training separate models and ensemble their predictions")
    return parser.parse_args()

def validate_args(args):
    if args.model_type == '3D' and args.slice_axis is not None:
        raise ValueError("The '--slice_axis' option is only valid for '2D' model type. Remove it for '3D' models.")
    if args.model_type == '2D' and not args.use_all_axes and args.slice_axis is None:
        raise ValueError("The '--slice_axis' option must be specified for '2D' model type unless '--use_all_axes' is enabled.")

def get_model(args, axis=None):
    """ Returns a ConvNeXt model. For 2D models, `axis` can specify which axis the model is for. """
    if args.model_type == '3D':
        return ConvNeXt3D(in_chans=1, num_classes=args.num_classes).to(args.device)
    else:
        model_cls = ConvNeXt2D
        if args.use_all_axes:
            assert axis in ['axial', 'coronal', 'sagittal'], "Invalid axis"
            print(f"Initializing ConvNeXt2D model for {axis} axis")
        return model_cls(in_chans=1, num_classes=args.num_classes).to(args.device)

def save_model(model, epoch, val_acc, weights_folder, axis=None):
    axis_suffix = f"_{axis}" if axis else ""
    weight_filename = f"epoch_{epoch+1}_val_acc_{val_acc:.4f}{axis_suffix}.pth"
    torch.save(model.state_dict(), os.path.join(weights_folder, weight_filename))

def create_save_folder(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    weights_folder = os.path.join(args.save_folder, f"model_weights_{timestamp}_{args.model_type.upper()}")
    os.makedirs(weights_folder, exist_ok=True)
    return weights_folder

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

def validate_one_epoch(models, dataloaders, criterion, device, use_all_axes=False):
    all_preds = []
    all_labels = []

    for model, dataloader in zip(models, dataloaders):
        model.eval()
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                preds_list.append(preds.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        all_preds.append(np.concatenate(preds_list))
        all_labels = np.concatenate(labels_list)

    if use_all_axes:
        # Ensemble validation with hard voting
        all_preds = np.array(all_preds)
        ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)
        accuracy = accuracy_score(all_labels, ensemble_preds)
    else:
        accuracy = accuracy_score(all_labels, all_preds[0])

    return accuracy

def test_model(models, dataloaders, device, use_all_axes=False):
    correct, total = 0, 0
    all_preds = []

    for model, dataloader in zip(models, dataloaders):
        model.eval()
        preds_list = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                preds_list.append(preds.cpu().numpy())

                if not use_all_axes:
                    correct += torch.sum(preds == labels).item()
                    total += labels.size(0)

        all_preds.append(np.concatenate(preds_list))

    if use_all_axes:
        # Hard voting ensemble
        all_preds = np.array(all_preds)
        ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)
        correct = np.sum(ensemble_preds == labels.cpu().numpy())
        total = len(labels)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def main():
    args = parse_args()
    validate_args(args)

    class_names = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    args.num_classes = len(class_names)
    print(f"Detected {args.num_classes} classes: {', '.join(class_names)}")

    if args.use_all_axes:
        loaders = []
        models = []
        for axis in ['axial', 'coronal', 'sagittal']:
            train_loader, val_loader, test_loader = prepare_data(
                data_dir=args.data_dir,
                test_size=args.test_size,
                val_size=args.val_size,
                batch_size=args.batch_size,
                resize_shape=tuple(args.resize_shape) if args.resize_shape else None,
                num_workers=args.num_workers,
                dataset_type='2D',
                slice_axis=axis,
                class_names=class_names,
                augment=args.augment
            )
            loaders.append((train_loader, val_loader, test_loader))
            models.append(get_model(args, axis=axis))
    else:
        train_loader, val_loader, test_loader = prepare_data(
            data_dir=args.data_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            batch_size=args.batch_size,
            resize_shape=tuple(args.resize_shape) if args.resize_shape else None,
            num_workers=args.num_workers,
            dataset_type=args.model_type,
            slice_axis=args.slice_axis,
            class_names=class_names,
            augment=args.augment
        )
        models = [get_model(args)]
        loaders = [(train_loader, val_loader, test_loader)]

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    criterion = nn.CrossEntropyLoss()
    for model, (train_loader, val_loader, _) in zip(models, loaders):
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        weights_folder = create_save_folder(args)

        for epoch in range(args.num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc = validate_one_epoch(models, [loader[1] for loader in loaders], criterion, device, use_all_axes=args.use_all_axes)
            print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

            save_model(model, epoch, val_acc, weights_folder, axis=args.slice_axis if not args.use_all_axes else axis)

    test_accuracy = test_model(models, [loader[2] for loader in loaders], device, use_all_axes=args.use_all_axes)

if __name__ == "__main__":
    main()
