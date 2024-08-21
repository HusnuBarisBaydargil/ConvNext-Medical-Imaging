# ConvNeXt3D Model Training

This repository contains the code to train a ConvNeXt3D model on 3D medical imaging data, such as NIfTI images. The script allows you to customize various aspects of the training process, including the dataset path, model parameters, and training configurations.

## Running the Training Script

To run the training script, use the following command and change the variables accordingly:

```bash
python train.py --data_dir /path/to/your/data \
                --save_folder /path/to/save/weights \
                --num_classes 2 \
                --device 3 \
                --num_epochs 50 \
                --resize_shape 128 128 128 \
                --test_size 0.2 \
                --val_size 0.1 \
                --batch_size 4 \
                --num_workers 6 \
                --learning_rate 0.001
```

