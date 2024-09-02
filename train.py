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
        print(f"Initializing ConvNeXt3D model")
        return ConvNeXt3D(in_chans=1, num_classes=args.num_classes).to(args.device)
    else:
        model_cls = ConvNeXt2D
        if args.use_all_axes:
            assert axis in ['axial', 'coronal', 'sagittal'], "Invalid axis"
            print(f"Initializing ConvNeXt2D model for {axis} axis")
        return model_cls(in_chans=1, num_classes=args.num_classes).to(args.device)

def save_all_models(models, epoch, val_acc, weights_folder, use_all_axes, model_type):
    """ Save all models' weights in one file or save the single model's weights. """
    print(f"Saving models for epoch {epoch+1}, validation accuracy: {val_acc:.4f}")
    if model_type == '2D' and use_all_axes:
        weights = {axis: model.state_dict() for axis, model in models.items()}
        weight_filename = f"epoch_{epoch+1}_val_acc_{val_acc:.4f}_all_axes.pth"
        print(f"Saving all axes models to {weight_filename}")
    else:
        weights = list(models.values())[0].state_dict()  # Only one model is present in 3D
        weight_filename = f"epoch_{epoch+1}_val_acc_{val_acc:.4f}.pth"
        print(f"Saving model to {weight_filename}")
    
    save_path = os.path.join(weights_folder, weight_filename)
    print(f"Saving model weights to {save_path}")
    
    try:
        torch.save(weights, save_path)
        print(f"Successfully saved models to {save_path}")
    except Exception as e:
        print(f"Error saving models: {e}")

def create_save_folder(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    weights_folder = os.path.join(args.save_folder, f"model_weights_{timestamp}_{args.model_type.upper()}")
    os.makedirs(weights_folder, exist_ok=True)
    return weights_folder

def train_one_epoch(args, models, loaders, criterion, optimizers, device):
    """ Train all models for one epoch """
    for model in models.values():
        model.train()

    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    all_labels = []
    all_preds = []  # Store predictions

    for axis in models.keys():
        model = models[axis]
        dataloader = loaders[axis][0]  # Train loader
        optimizer = optimizers[axis]

        for inputs, labels in tqdm(dataloader, desc=f"Training {axis}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct_preds += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    # For 3D model, simply calculate accuracy directly
    ensemble_accuracy = accuracy_score(all_labels, all_preds)
    epoch_loss = running_loss / total_samples
    print(f"Training loss: {epoch_loss:.4f}, accuracy: {ensemble_accuracy:.4f}")
    return epoch_loss, ensemble_accuracy


def validate_one_epoch(args, models, loaders, criterion, device):
    """ Validate all models for one epoch """
    for model in models.values():
        model.eval()

    all_preds = []
    all_labels = []

    for axis in models.keys():
        model = models[axis]
        dataloader = loaders[axis][1]  # Validation loader

        axis_preds = []
        axis_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=f"Validating {axis}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                axis_preds.extend(preds.cpu().numpy())
                axis_labels.extend(labels.cpu().numpy())

        all_preds.extend(axis_preds)
        all_labels.extend(axis_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Validation accuracy: {accuracy:.4f}")

    return accuracy

def test_model(args, models, loaders, device):
    """ Test all models on the test dataset """
    for model in models.values():
        model.eval()

    all_preds = []
    all_labels = []

    for axis in models.keys():
        model = models[axis]
        dataloader = loaders[axis][2]  # Test loader

        axis_preds = []
        axis_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=f"Testing {axis}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                axis_preds.extend(preds.cpu().numpy())
                axis_labels.extend(labels.cpu().numpy())

        all_preds.extend(axis_preds)
        all_labels.extend(axis_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


def main():
    args = parse_args()
    validate_args(args)

    class_names = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    args.num_classes = len(class_names)
    print(f"Detected {args.num_classes} classes: {', '.join(class_names)}")

    if args.use_all_axes and args.model_type == '2D':
        loaders = {}
        models = {}
        optimizers = {}
        axes = ['axial', 'coronal', 'sagittal']
        for axis in axes:
            print(f"Preparing data for {axis} axis")
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
            loaders[axis] = (train_loader, val_loader, test_loader)
            models[axis] = get_model(args, axis=axis)
            optimizers[axis] = optim.Adam(models[axis].parameters(), lr=args.learning_rate)
            print(f"Model for {axis} axis initialized")
    else:
        print(f"Preparing data for model: {args.model_type}, Axis: {args.slice_axis if args.model_type == '2D' else 'N/A'}")
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
        models = {args.slice_axis: get_model(args)} if args.model_type == '2D' else {'axial': get_model(args)}
        loaders = {args.slice_axis: (train_loader, val_loader, test_loader)} if args.model_type == '2D' else {'axial': (train_loader, val_loader, test_loader)}
        optimizers = {args.slice_axis: optim.Adam(models[args.slice_axis].parameters(), lr=args.learning_rate)} if args.model_type == '2D' else {'axial': optim.Adam(models['axial'].parameters(), lr=args.learning_rate)}
        axes = [args.slice_axis] if args.model_type == '2D' else ['axial']

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    criterion = nn.CrossEntropyLoss()
    weights_folder = create_save_folder(args)
    print(f"Weights will be saved in {weights_folder}")

    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}")
        train_loss, train_acc = train_one_epoch(args, models, loaders, criterion, optimizers, device)
        val_acc = validate_one_epoch(args, models, loaders, criterion, device)
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

        save_all_models(models, epoch, val_acc, weights_folder, use_all_axes=args.use_all_axes and args.model_type == '2D', model_type=args.model_type)

    test_accuracy = test_model(args, models, loaders, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
