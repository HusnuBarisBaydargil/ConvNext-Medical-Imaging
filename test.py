import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from dataset import prepare_data
from convnext import ConvNeXt3D, ConvNeXt2D
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Test ConvNeXt model")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--device', type=int, default=0, help='CUDA device number (default: 0)')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the saved model weights file')
    parser.add_argument('--model_type', type=str, choices=['2D', '3D'], required=True, help="Choose between '2D' and '3D' model")
    parser.add_argument('--resize_shape', type=int, nargs=3, default=None, help='Resize shape for the NIfTI images (default: None for no resizing)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the data loader (default: 4)')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for the data loader (default: 10)')
    parser.add_argument('--slice_axis', type=str, choices=['axial', 'coronal', 'sagittal'], default=None, help="Axis to slice along for 2D models (only for 2D)")
    parser.add_argument('--use_all_axes', action='store_true', help="Use all 2D axes (axial, coronal, sagittal) for ensemble testing")
    return parser.parse_args()

def validate_args(args):
    if args.model_type == '3D' and (args.slice_axis is not None or args.use_all_axes):
        raise ValueError("The '--slice_axis' and '--use_all_axes' options are only valid for '2D' model type.")
    if args.model_type == '2D' and not args.use_all_axes and args.slice_axis is None:
        raise ValueError("The '--slice_axis' option must be specified for '2D' model type unless '--use_all_axes' is enabled.")

def load_model(args, num_classes, axis=None, weight_dict=None):
    model_cls = ConvNeXt3D if args.model_type == '3D' else ConvNeXt2D
    model = model_cls(num_classes=num_classes).to(args.device)
    
    if args.use_all_axes:
        # Load the appropriate axis weights
        assert weight_dict is not None, "Expected weight dictionary for all axes"
        model.load_state_dict(weight_dict[axis])
    else:
        # Load single axis weights
        weight_path = args.model_weights
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(weight_path, map_location=device))
    
    return model

def test_model(models, dataloaders, device, use_all_axes=False):
    all_preds = []
    all_labels = []

    for model, dataloader in zip(models, dataloaders):
        model.eval()
        preds_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                preds_list.append(preds.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        all_preds.append(np.concatenate(preds_list))
        if not all_labels:
            all_labels = np.concatenate(labels_list)

    if use_all_axes:
        # Hard voting ensemble
        all_preds = np.array(all_preds)
        ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)
        correct = np.sum(ensemble_preds == all_labels)
    else:
        correct = np.sum(all_preds[0] == all_labels)
    
    total = len(all_labels)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def main():
    args = parse_args()
    validate_args(args)

    class_names = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {', '.join(class_names)}")

    if args.use_all_axes:
        loaders = []
        models = []
        axes = ['axial', 'coronal', 'sagittal']
        
        # Load the weight dictionary containing all axes' weights
        weight_dict = torch.load(args.model_weights, map_location=f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
        
        for axis in axes:
            test_loader = prepare_data(
                data_dir=args.data_dir,
                test_size=1.0, 
                val_size=0.0,  
                batch_size=args.batch_size,
                resize_shape=tuple(args.resize_shape) if args.resize_shape else None,
                num_workers=args.num_workers,
                dataset_type='2D',
                slice_axis=axis,
                class_names=class_names,
                augment=False
            )[2]
            loaders.append(test_loader)
            models.append(load_model(args, num_classes, axis=axis, weight_dict=weight_dict))
    else:
        test_loader = prepare_data(
            data_dir=args.data_dir,
            test_size=1.0, 
            val_size=0.0,  
            batch_size=args.batch_size,
            resize_shape=tuple(args.resize_shape) if args.resize_shape else None,
            num_workers=args.num_workers,
            dataset_type=args.model_type,
            slice_axis=args.slice_axis,
            class_names=class_names,
            augment=False
        )[2]
        models = [load_model(args, num_classes)]
        loaders = [test_loader]

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    test_accuracy = test_model(models, loaders, device, use_all_axes=args.use_all_axes)

if __name__ == "__main__":
    main()
