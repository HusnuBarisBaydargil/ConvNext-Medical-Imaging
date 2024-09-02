# ConvNeXt3D & ConvNeXt2D Model Training

This repository contains the code to train ConvNeXt models on 3D NIfTI images or 2D slices derived from these images. The script allows you to customize various aspects of the training process, including the dataset path, model parameters, and training configurations. You can also specify the model type (2D or 3D), the axis along which 2D slices should be taken, and whether to apply data augmentation during training.
If you only have DICOM files with extensions such as this, `/Path/to/ADNI/AD/003_S_1059/Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution/2007-01-10_11_15_31.0/I37090/ADNI_003_S_1059_PT_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20070122123023427_1_S25144_I37090.dcm`, you can convert them into NIfTI format using the script `dicom_to_nifti.py`

# Model Types

* 3D Model: Trains on full 3D NIfTI images.
* 2D Model: Trains on 2D slices derived from 3D NIfTI images. You can specify the axis along which the slices are taken (axial, coronal, or sagittal).

## Running the Training Script

# Example for 3D Model
To train a ConvNeXt3D model on 3D NIfTI images, use the following command:

```bash
python train.py --data_dir /path/to/your/data \
                --save_folder /path/to/save/weights \
                --model_type 3D \
                --num_classes 2 \
                --device 3 \
                --num_epochs 50 \
                --test_size 0.2 \
                --val_size 0.1 \
                --batch_size 4 \
                --num_workers 6 \
                --learning_rate 0.001 \
                --augment
```
# Example for 2D Model

To train a ConvNeXt2D model on 2D slices from 3D NIfTI images, use the following command:

```bash
python train.py --data_dir /path/to/your/data \
                --save_folder /path/to/save/weights \
                --model_type 2D \
                --slice_axis axial \
                --num_classes 2 \
                --device 3 \
                --num_epochs 50 \
                --test_size 0.2 \
                --val_size 0.1 \
                --batch_size 4 \
                --num_workers 6 \
                --learning_rate 0.001 \
                --augment
```

# Example for Multi-dimensional 2D Model

To train a ConvNeXt2D model on all 2D slices (axial, coronal, and sagittal), use the following command:

```bash
python train.py --data_dir /path/to/your/data \
                --save_folder /path/to/save/weights \
                --model_type 2D \
                --num_classes 2 \
                --device 3 \
                --num_epochs 50 \
                --test_size 0.2 \
                --val_size 0.1 \
                --batch_size 4 \
                --num_workers 6 \
                --learning_rate 0.001 \
                -- use_all_axes \
                --augment
```

# Testing the model with saved weights

```bash
python test.py \
    --data_dir /path/to/data/dir \
    --model_weights /path/to/model/dir/model.pth \
    --model_type 3D \
    --device 3 \
    --batch_size 6 \
    --num_workers 6 
```
