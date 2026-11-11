# SDSE-Net

SDSE-Net is a 2D deep learning network for semantic segmentation of medical
images. The model retains the main structure and multi-scale skip connections
of the RWKV-UNet encoder, adds a Transformer bottleneck at the deepest encoder
level, and uses an improved decoder to restore the spatial resolution.

This repository contains the model definition, Synapse/ACDC dataset loaders,
training scripts, testing scripts, and batch testing tools. The model entry
point is [`SDSE_Net.py`](./SDSE_Net.py), and the top-level network class is
`SDSE_Net`.

## Table of Contents

- [Project Structure](#project-structure)
- [Model Overview](#model-overview)
- [Model Architecture and Segmentation Results](#model-architecture-and-segmentation-results)
- [Environment Setup](#environment-setup)
- [Downloading Datasets and Model Weights](#downloading-datasets-and-model-weights)
- [Data Preparation](#data-preparation)
- [Pre-trained Weights](#pre-trained-weights)
- [Training](#training)
- [Testing and Inference](#testing-and-inference)
- [Automatic Batch Testing](#automatic-batch-testing)
- [Output Files](#output-files)
- [Notes](#notes)
- [Acknowledgments and Citation](#acknowledgments-and-citation)

## Project Structure

```text
SDSE-Net/
├── SDSE_Net.py              # SDSE_Net, encoder, and decoder definitions
├── train.py                 # Training entry point
├── test.py                  # Testing and inference entry point
├── trainer.py               # Synapse, ACDC, and BUSI training workflows
├── run_workflow.py          # Automatic batch testing after training
├── auto_test_Synapse.py     # Batch testing of checkpoints
├── utils.py                 # Loss, inference, and evaluation utilities
├── module/
│   └── basic_modules.py     # Basic network modules
├── ccm/
│   ├── ccm.py               # CCM feature fusion module
│   └── main_blocks.py       # CCM dependency modules
├── datasets/
│   ├── dataset_synapse.py   # Synapse data loading
│   └── dataset_acdc.py      # ACDC data loading
├── cuda/
│   ├── wkv_op.cpp           # WKV C++ extension
│   └── wkv_cuda.cu          # WKV CUDA implementation
├── figs/
│   ├── Figure_1.png         # Model architecture diagram
│   └── Figure_2.png         # Segmentation result visualization
├── lists/                   # Dataset list files
├── ckpts/                   # Example checkpoints
└── requirements.txt         # Python dependencies
```

## Model Overview

The main design of SDSE-Net is as follows:

1. **RWKV encoder**  
   `RWKV_UNet_encoder_B` is used as the encoder. It progressively reduces the
   spatial resolution and increases the number of channels through multiple
   feature extraction stages.
2. **Transformer bottleneck**  
   After the deepest encoder level, `stage4`, the channel dimension is
   increased. Multiple Transformer bottleneck modules model global context,
   after which the channel dimension is mapped back to the dimensions required
   by the decoder.
3. **Multi-scale skip connections**  
   Intermediate encoder features are fused and enhanced through the
   `CCMix`/`CCM` modules, then concatenated with decoder features at the
   corresponding scales.
4. **Improved decoder**  
   The decoding stage consists of four `UpBlock` modules. Each module includes
   convolutional feature processing, a projection layer, and bilinear
   upsampling to progressively restore the feature-map resolution.
5. **Segmentation head**  
   Finally, a `1x1` convolution maps 24 channels to the number of classes. The
   output shape is `(batch_size, num_classes, height, width)`.

The default configuration of `SDSE_Net` is:

| Configuration | Default value |
|---|---:|
| Number of input channels | 1 |
| Input image size | 224 × 224 |
| Number of output classes | 9 for Synapse |
| Encoder | `RWKV_UNet_encoder_B` |
| Pre-trained encoder weights | `net_B.pth` |

For single-channel input, the model copies the image to three channels at the
start of the forward pass to match the encoder's input structure.

## Model Architecture and Segmentation Results

The model architecture diagram and segmentation result visualizations are
available in the `figs` directory.

### Model Architecture

![SDSE-Net model architecture](./figs/Figure_1.png)

### Segmentation Result Visualization

![SDSE-Net segmentation results](./figs/Figure_2.png)

## Environment Setup

Python 3.9 is recommended, together with a CUDA-enabled PyTorch environment.
The project includes the CUDA extension source files `cuda/wkv_op.cpp` and
`cuda/wkv_cuda.cu`, so a working CUDA compilation environment is required to
run the model.

```bash
conda create -n sdse-net python=3.9
conda activate sdse-net

# Install a version of PyTorch and torchvision matching your local CUDA version
pip install torch torchvision
pip install -r requirements.txt
```

The main dependencies listed in `requirements.txt` include:

- PyTorch components
- `timm`
- `einops`
- `numpy`, `scipy`, and `scikit-image`
- `h5py`
- `albumentations`
- `SimpleITK` and `MedPy`
- `tensorboard`

## Downloading Datasets and Model Weights

Because of GitHub's per-file size limit, the following five files are not
included in this repository. They are hosted in the
[Hugging Face model repository](https://huggingface.co/aili-yasi/SDSE-Net/tree/main).
Download the corresponding files from that page and place them in the project
directory as shown below:

| Hugging Face file | Destination after download | Purpose |
|---|---|---|
| `net_B.pth` | `net_B.pth` in the project root | Pre-trained encoder weights |
| `Synapse.zip` | `data/Synapse.zip` in the project root, then extract to `data/Synapse/` | Synapse dataset |
| `transunet_ACDC.zip` | `data/transunet_ACDC.zip` in the project root, then extract to `data/ACDC/` | ACDC dataset |
| `ckpts/Synapse_base.pth` | `ckpts/Synapse_base.pth` | Complete Synapse model checkpoint |
| `ckpts/acdc_base.pth` | `ckpts/acdc_base.pth` | Complete ACDC model checkpoint |

After downloading and extracting the files, the project should contain at least:

```text
SDSE-Net/
├── net_B.pth
├── data/
│   ├── Synapse/
│   │   ├── train_npz/
│   │   └── test_vol_h5/
│   └── ACDC/
│       ├── ACDC_training_slices/
│       └── ACDC_training_volumes/
└── ckpts/
    ├── Synapse_base.pth
    └── acdc_base.pth
```

If extraction creates an additional same-named parent directory, adjust the
directory hierarchy so that the training and testing scripts can access
`data/Synapse/` or `data/ACDC/` directly. Always use the project root as the
working directory when running the scripts.

## Data Preparation

### Synapse

The training script reads the following paths by default:

```text
data/Synapse/train_npz/
lists/lists_Synapse/train.txt
lists/lists_Synapse/test_vol.txt
```

The recommended data layout is:

```text
data/
└── Synapse/
    ├── train_npz/
    │   ├── case0005_slice000.npz
    │   └── ...
    └── test_vol_h5/
        ├── case0001.npy.h5
        └── ...
```

Synapse uses different loading procedures for training and testing:

- During training, each line in `lists/lists_Synapse/train.txt` contains a
  sample name without an extension. The data loader reads the `image` and
  `label` arrays from `train_npz/<sample_name>.npz`.
- During testing, each line in `lists/lists_Synapse/test_vol.txt` contains a
  case name without an extension. The data loader reads the `image` and
  `label` arrays from `test_vol_h5/<case_name>.npy.h5`.
- Training images are processed by `RandomGenerator`, which applies brightness
  and contrast changes and random cropping when needed, resizes them to
  `img_size × img_size` using nearest-neighbor interpolation, and converts
  them to single-channel tensors.
- Test volumes do not use training augmentation. The testing script performs
  inference on the 2D slices of each case through `test_single_volume` and
  calculates Dice and HD95 for each class.

### ACDC

The default ACDC data root is `data/ACDC`, with the following structure:

```text
data/
└── ACDC/
    ├── ACDC_training_slices/
    │   ├── patient001_frame01_slice_0.h5
    │   └── ...
    └── ACDC_training_volumes/
        ├── patient001_frame01.h5
        └── ...
```

The ACDC data split and file-loading logic are defined in
[`datasets/dataset_acdc.py`](./datasets/dataset_acdc.py):

- `patient001` to `patient020`: test set
- `patient021` to `patient030`: validation set
- All remaining cases: training set

The loading procedures are:

- During training, slice-level `.h5` files are read from
  `ACDC_training_slices/`. Their `image` and `label` data are processed with
  `RandomGenerator`, which applies random flipping, random rotation,
  brightness/contrast changes, and random cropping before resizing to
  `img_size × img_size`.
- During validation and testing, volume `.h5` files are read from
  `ACDC_training_volumes/` without `RandomGenerator`.
- Each ACDC `.h5` file must contain `image` and `label` datasets. During
  testing, the case name is obtained by removing `.h5` from the filename.

### BUSI

The training and testing scripts retain configuration entry points for the
BUSI dataset, but this repository currently does not include
`datasets/dataset_busi.py`. To run BUSI, the corresponding dataset
implementation and data directory must be added.

## Pre-trained Weights

By default, `SDSE_Net` attempts to load the following pre-trained encoder
weights:

```text
net_B.pth
```

You can specify a weight file during training with `--pretrained_path`:

```bash
python train.py --pretrained_path path/to/net_B.pth
```

To disable pre-trained encoder weights, pass an empty value or set
`pretrained_path=None` when constructing the model. During testing, the
pre-trained encoder weights are not loaded again by default; instead, the
complete segmentation model checkpoint is loaded.

## Training

Run training commands from the project root.

### Synapse Training Example

```bash
python train.py ^
  --dataset Synapse ^
  --max_epochs 30 ^
  --batch_size 24 ^
  --base_lr 0.001 ^
  --img_size 224 ^
  --pretrained_path net_B.pth
```

In PowerShell or Linux/macOS, the command can also be written on one line:

```bash
python train.py --dataset Synapse --max_epochs 30 --batch_size 24 --base_lr 0.001 --img_size 224 --pretrained_path net_B.pth
```

### ACDC Training Example

```bash
python train.py --dataset ACDC --max_epochs 30 --batch_size 24 --base_lr 0.001 --img_size 224 --pretrained_path net_B.pth
```

### Common Training Arguments

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | `Synapse` | `Synapse`, `ACDC`, or `busi` |
| `--max_epochs` | `30` | Maximum number of training epochs |
| `--max_iterations` | `30000` | Maximum number of iterations, used for output directory naming |
| `--batch_size` | `24` | Batch size per GPU |
| `--base_lr` | `0.001` | Initial learning rate |
| `--img_size` | `224` | Input image side length |
| `--gpu` | `0` | GPU index to use |
| `--n_gpu` | `1` | Number of GPUs |
| `--seed` | `1234` | Random seed |
| `--pretrained_path` | `net_B.pth` | Path to pre-trained encoder weights |

Training uses the AdamW optimizer and records cross-entropy loss, Dice loss,
and total loss. The training functions for each dataset are located in
[`trainer.py`](./trainer.py).

## Testing and Inference

### Test with the Default Checkpoint

```bash
python test.py --dataset Synapse --img_size 224
```

The testing script searches the training output directory for `best_model.pth`.
If it does not exist, it tries to find the corresponding epoch checkpoint.

### Test with a Specific Checkpoint

```bash
python test.py ^
  --dataset Synapse ^
  --img_size 224 ^
  --path_specific ckpts/Synapse_base.pth
```

On Linux/macOS or PowerShell, the one-line form is:

```bash
python test.py --dataset Synapse --img_size 224 --path_specific ckpts/Synapse_base.pth
```

### Save Prediction Results

Add `--is_savenii` to save test results to the `predictions` directory:

```bash
python test.py --dataset Synapse --img_size 224 --path_specific ckpts/Synapse_base.pth --is_savenii
```

During testing, Dice and HD95 are calculated for each class, and the average
results are printed to the log.

## Automatic Batch Testing

`auto_test_Synapse.py` automatically finds `.pth` files in a specified
directory, sorts them by the numbers in their filenames, and tests them one by
one:

```bash
python auto_test_Synapse.py --weights_dir outputs/rwkvSynapse224/exp1_pretrain_...
```

You can also use [`run_workflow.py`](./run_workflow.py) to automatically run
training and batch testing:

```bash
python run_workflow.py --dataset Synapse --max_epochs 30 --img_size 224
```

This script:

1. Calls `train.py` to train the model.
2. Calculates the output directory from the training arguments.
3. Calls `auto_test_Synapse.py` to test the checkpoints in that directory.

## Output Files

The training output directory is determined by the dataset, image size, number
of training epochs, batch size, learning rate, and random seed. Its default
form is similar to:

```text
outputs/
└── rwkvSynapse224/
    └── exp1_pretrain_bs24_lr0.001_224/
        ├── best_model.pth
        ├── epoch_*.pth
        └── ...
```

Training logs and TensorBoard scalars are written to the corresponding
experiment directory. Use TensorBoard to view the training curves:

```bash
tensorboard --logdir outputs
```

Testing logs are written to `test_log` by default. When `--is_savenii` is
enabled, prediction results are written to
`predictions/<experiment>/<checkpoint>/`.

## Notes

1. Always run scripts from the project root. Otherwise, relative paths such as
   `data/`, `lists/`, `cuda/`, and the weight paths may not be resolved
   correctly.
2. The current training and testing entry points use CUDA. Before running,
   verify that `torch.cuda.is_available()` is `True` and install a PyTorch
   version compatible with the local driver.
3. Importing `SDSE_Net.py` compiles or loads the WKV CUDA extension. The first
   run may take a considerable amount of time.
4. `--img_size` should match the size used by the training checkpoint.
5. `--num_classes` is set automatically according to the dataset configuration
   and usually does not need to be changed manually.
6. Use `^` for line continuation in the Windows command prompt and a
   backslash `\` in Linux/macOS. Single-line commands can also be used.

## Acknowledgments and Citation

This project uses or references parts of the following projects:

- [TransUNet](https://github.com/Beckschen/TransUNet)
- [RWKV-UNet](https://github.com/juntaoJianggavin/RWKV-UNet)
- [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)
