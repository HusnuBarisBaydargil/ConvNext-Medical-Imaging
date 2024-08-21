import os
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
from convnext import ConvNeXt3D
from dataset import prepare_data
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Train and validate ConvNeXt3D model")
parser.add_argument('--data_dir', type=str, default="/path/to/adni/twoclass/folder", help='Path to the dataset directory')
parser.add_argument('--device', type=int, default=3, help='CUDA device number (default: 3)')
parser.add_argument('--save_folder', type=str, default="/path/to/save/folder", help='Directory to save model weights')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--resize_shape', type=int, nargs=3, default=[128, 128, 128], help='Resize shape for the NIfTI images (default: 128x128x128)')
parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the data to be used as test set (default: 0.2)')
parser.add_argument('--val_size', type=float, default=0.1, help='Proportion of the training data to be used as validation set (default: 0.1)')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the data loaders (default: 4)')
parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for the data loaders (default: 0)')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes (default: 2)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer (default: 0.001)')

args = parser.parse_args()

train_loader, val_loader, test_loader = prepare_data(
    data_dir=args.data_dir,
    test_size=args.test_size,
    val_size=args.val_size,
    batch_size=args.batch_size,
    resize_shape=tuple(args.resize_shape),
    num_workers=args.num_workers
)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
model = ConvNeXt3D(num_classes=args.num_classes).to(device)

if args.num_classes == 2:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
weights_folder = os.path.join(args.save_folder, f"model_weights_{timestamp}")
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if args.num_classes == 2:
            labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            if args.num_classes == 2:
                labels = labels.float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            if args.num_classes == 2:
                preds = (outputs > 0).int()
            else:
                _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            if args.num_classes == 2:
                labels = labels.float().unsqueeze(1)
            outputs = model(inputs)
            if args.num_classes == 2:
                preds = (outputs > 0).int()
            else:
                _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

for epoch in range(args.num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    
    weight_filename = f"epoch_{epoch+1}_val_acc_{val_acc:.4f}.pth"
    torch.save(model.state_dict(), os.path.join(weights_folder, weight_filename))

test_accuracy = test(model, test_loader, device)
