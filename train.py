import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import argparse
from dataset import prepare_data
from convnext import ConvNeXt3D, ConvNeXt2D

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
    return parser.parse_args()


def validate_args(args):
    if args.model_type == '3D' and args.slice_axis is not None:
        raise ValueError("The '--slice_axis' option is only valid for '2D' model type. Remove it for '3D' models.")
    if args.model_type == '2D' and args.slice_axis is None:
        raise ValueError("The '--slice_axis' option must be specified for '2D' model type.")


def get_model(args):
    model_cls = ConvNeXt3D if args.model_type == '3D' else ConvNeXt2D
    return model_cls(num_classes=args.num_classes).to(args.device)


def save_model(model, epoch, val_acc, weights_folder):
    weight_filename = f"epoch_{epoch+1}_val_acc_{val_acc:.4f}.pth"
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


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


def test_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy


def main():
    args = parse_args()
    validate_args(args)

    class_names = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    args.num_classes = len(class_names)
    print(f"Detected {args.num_classes} classes: {', '.join(class_names)}")

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

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    args.device = device

    model = get_model(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    weights_folder = create_save_folder(args)

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        save_model(model, epoch, val_acc, weights_folder)

    test_accuracy = test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
