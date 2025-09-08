# **Urban Scene Understanding with Deep Learning**
*Muhammed Yahya Avar - Praneshraj Tiruppur Nagarajan Dhyaneswar*

This project explores urban scene understanding using deep learning, focusing on boundary detection with the Cityscapes dataset.
The work was developed as part of the Advanced Analytics Lab course and demonstrates how convolutional neural networks (CNNs), specifically U-Net, can help detect object boundaries in complex urban environments—a crucial step for applications like autonomous driving.

Key tasks:

- Data preprocessing and analytics

- Training and evaluation of a U-Net architecture

- Boundary detection performance evaluation (IoU, Dice Score)

## Dataset

The project uses the Cityscapes Dataset
, a large-scale dataset for semantic urban scene understanding.

- 5,000 finely annotated images (pixel-level semantic, instance, panoramic segmentation)

- 20,000 coarsely annotated images

- This project used a subset of 2,975 images for training and evaluation.

Each dataset sample includes:

- leftImg8bit: RGB input images

- polygons.json: Polygon boundaries for semantic objects

## Preprocessing

Implemented in Submission_Data_Preprocessing_and_Analytics_13052205.ipynb
.

- Images resized from 1024×1024 → 256×256 (for memory efficiency).

Albumentations used for augmentation:

- RandomBrightnessContrast

- GaussianBlur

- ShiftScaleRotate

- Normalize (ImageNet mean/std)

Validation set: only normalization applied.

## Model – U-Net

Implemented in Submission_Cityscapes_UNET_13052025.ipynb
.

- Encoder: 4 convolutional blocks with downsampling

- Bottleneck: 512 channels

- Decoder: 3 upsampling + skip connection blocks

- Output Layer: 1×1 convolution → boundary mask

Training setup:

- Optimizer: Adam (lr=1e-3)

- Loss: Binary Cross-Entropy with Logits

- Batch size: 8 (with gradient accumulation)

- Epochs: 40 per data chunk

- Mixed precision training on CUDA

## Evaluation

Metrics used:

- Loss (BCE)

- Intersection over Union (IoU)

- Dice Coefficient

## Results:

- Mean IoU: 0.3016

- Dice Score: 0.4809

- Final training loss: 0.0948

## Visualizations

Examples of predicted boundaries compared against ground truth are included in the notebooks.
