# ConvNeXt3D Model Training

This repository contains the code to train a ConvNeXt3D model on 3D medical imaging data, such as NIfTI images. The script allows you to customize various aspects of the training process, including the dataset path, model parameters, and training configurations.

## Running the Training Script

To run the training script, use the following command and change the variables accordingly:

```bash
python train.py --data_dir /path/to/your/data --save_folder /path/to/save/weights --num_classes 4 --learning_rate 0.001 --num_epochs 50
