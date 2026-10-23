## Introduction
This is our implementation of *Region-Aware Asynchronous Attention for Incremental Implicitly-Refined Classification*.

**TL;DR**: A hierarchy-aware encoder-decoder architecture using superpixel-based segmentation and asynchronous dual-branch decoding for IIRC.

**Abstract**:
IIRC is an incremental learning paradigm in which each class allows multiple granularity labels, but only one label is given at each incremental session. 
Compared with traditional continual learning paradigms, IIRC suffers from the severe Hierarchical Forgetting conundrum: models have to absorb new knowledge while memorize  old knowledge with updating hierarchical relation of old and new classes. Existing methods face limitations in feature extraction and decoding: CNN-based encoders are confined to local receptive fields and fail to capture global hierarchical semantics, while standard Vision Transformers (ViT) use rigid square patches that break object visual boundaries, making them unable to align coarse-level and fine-level features to consistent visual regions. Additionally, directly coupling extracted features with linear classifiers accelerates semantic drift and exacerbates forgetting under streaming data. We build a category-to-region attention mechanism which enforce hierarchical semantics parsing to keep pace with regional semantics parsing, and propose the Region-Aware Asynchronous Attention (RA}). RA3 consists of the Region-aware Self-Attention (RSA) encoder and the Asynchronous Cross-Attention (ACA) decoder. Towards the broken visual boundary of images incurred from cube tokens of Vision Transformers, the RSA encoder designs the superpixel embedding module and superpixel group pooling module to collect visual feature from fine level to coarse level, which aggregates the region-aware attention clues. Towards the hierarchical semantics confusion incurred from synchronous attention, the ACA decoder designs a dual-branch cross-attention structure to extract coarse-level and fine-level semantics according to different-level input token representation and different learning rate. We conduct experiments on three datasets: IIRC-CIFAR, IIRC-ImageNet-S and IIRC-ImageNet-L. The results indicate that, except the first dataset whose resolution is small, RA3 achieves the best results on the second and third datasets, with 2.55\%, 2.94\% AIPWJS increments against the second best results, respectively. 


## IIRC Setup
![IIRC Setup](./IIRC-Setup.png)

## Our Method: RA3
![HiSeg Framework](./HSFS-Framework.png)

## Dependencies
```
torch==2.1.2
torchvision==0.16.2
timm==0.4.9
dgl>=1.0.0
numpy>=1.24.0
opencv-python>=4.4.0
lmdb>=1.4.1
Pillow>=9.0.0
continuum>=1.0.27
PyYAML>=6.0
```

## Usage

##### 1. Install dependencies
First, create a conda virtual environment and activate it:
```
conda create --name HSFS python=3.10
conda activate HSFS
```
Install PyTorch 2.1.2 with the appropriate CUDA version for your infrastructure, for example:
```
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```
Install remaining dependencies:
```
pip install -r requirements.txt
```
Install the `cast_models` package:
```
cd cast_models && pip install -e . && cd ..
```

##### 2. Run code
- IIRC-CIFAR
    ```
    bash train.sh 0,1 --options options/model/iirc_cifar100_dytox_hcast.yaml --name hsfs --data-path PATH_TO_DATASET --output-basedir PATH_TO_CHECKPOINTS
    ```
- IIRC-ImageNet-Subset
    ```
    bash train.sh 0,1 --options options/model/iirc_cifar100_dytox_hcast.yaml --name hsfs --data-set iirc_imagenet_subset --input-size 224 --patch-size 8 --depth 2 2 1 --num-clusters 64 16 8 --data-path PATH_TO_DATASET --output-basedir PATH_TO_CHECKPOINTS
    ```
- IIRC-ImageNet-Lite
    ```
    bash train.sh 0,1 --options options/model/iirc_cifar100_dytox_hcast.yaml --name hsfs --data-set iirc_imagenet_lite --input-size 224 --patch-size 8 --depth 2 2 1 --num-clusters 64 16 8 --data-path PATH_TO_DATASET --output-basedir PATH_TO_CHECKPOINTS
    ```
