# 202112-25-Roommate-Apartment-Finding-Platform
EECS E6893 Big Data Analytics final project

# e4040-2021fall-project-REAN-jz3313-rc3372-xz3014

## Introduction
Residual Attention Network is a convolutional neural network using attention mechanism which can incorporate with state-of-the-art feed forward network architecture in an end-to-end training fashion.

Residual Attention Networks are described in the paper "Residual Attention Network for Image Classification"(https://arxiv.org/pdf/1704.06904.pdf).

This project uses TensorFlow and Keras as the building blocks to attempt a reproduction of the original RAN paper.


## Prerequisites
### VM Environment
The project is hosted on GCP VM instance with the following configuration:
 - machine type: n1-standard-8
 - GPUs: 1 x NVIDIA Tesla T4
 - OS: Ubuntu 18.04.4

### Packages
Our project uses Python 3.6.9, and following are the packages used in the project.

 - CUDA: 11.0
 - cudnn: 8.0
 - Python 3.6.9
 - tensorflow 2.4.0

## Datasets

CIFAR-10: CIFAR-10 data set contains 60, 000 32 × 32 color images in 10 classes, with 50, 000 training images and 10, 000 test images. 

CIFAR-100: CIFAR-100 data set contains 60, 000 32 × 32 color images in 100 classes, with 50, 000 training images and 10, 000 test images.

The CIFAR-10 and CIFAR-100 data set can be accessed from tensorflow dataset using like `tf.keras.datasets.cifar10.load_data() ` and need not to be stored in Github Repo or Google Drive.

## Model

Model is stored in Google Drive: https://drive.google.com/drive/folders/1dZYexOiwrqoOLvYGVXS7hMbWtxudR9mv



## Results
We both implemented Attention-56 and Attention-92 on CIFAR-10 and CIFAR-100. The accuracies are listed below.   

CIFAR-10:  
| Model | CIFAR-10 Acc (%)  | Orig. Paper Acc (%) | CIFAR-10 time/epoch (s) | Params | 
| :---: | :---: | :---: | :---: | :---: |
| Attention-56 | 83.94 | 94.48| 110 | 59M | 
| Attention-92 | 83.82 | 95.01| 110 | 112M | 

CIFAR-100:  
| Model | CIFAR-100 Acc (%) | Orig. Paper Acc (%) | CIFAR-100 time/epoch (s) | Params | 
| :---: | :---: | :---: | :---: | :---: |
| Attention-56 | 54.39 | N/A | 180 | 59M |
| Attention-92 | 54.14 | 78.29 | 180 | 112M |



## Organization

Main Jupiter Notebook: Task-CIFAR10.ipynb, Task-CIFAR100.ipynb

Source code: including in the src

Figures: figures showing that the work are done in GCP, key results

Logs: Model training logs

```
./
├── README.md
├── Task-CIFAR10.ipynb
├── Task-CIFAR100.ipynb
├── figures
│   ├── cifar100_attention56.png
│   ├── cifar100_attention92.png
│   ├── cifar10_attention56.png
│   ├── cifar10_attention92.png
│   ├── gcp_work_example_screenshot_1.png
│   ├── gcp_work_example_screenshot_2.png
│   └── gcp_work_example_screenshot_3.png
├── logs
│   ├── cifar10
│   │   ├── 20211219-201804
│   │   │   ├── train
│   │   │   │   ├── events.out.tfevents.1639945085.nndl.25596.7096.v2
│   │   │   │   ├── events.out.tfevents.1639945099.nndl.profile-empty
│   │   │   │   └── plugins
│   │   │   │       └── profile
│   │   │   │           └── 2021_12_19_20_18_19
│   │   │   │               ├── nndl.input_pipeline.pb
│   │   │   │               ├── nndl.kernel_stats.pb
│   │   │   │               ├── nndl.memory_profile.json.gz
│   │   │   │               ├── nndl.overview_page.pb
│   │   │   │               ├── nndl.tensorflow_stats.pb
│   │   │   │               ├── nndl.trace.json.gz
│   │   │   │               └── nndl.xplane.pb
│   │   │   └── validation
│   │   │       └── events.out.tfevents.1639945200.nndl.25596.23536.v2
│   │   ├── 20211219-215701
│   │   │   ├── train
│   │   │   │   ├── events.out.tfevents.1639951022.nndl.25596.314304.v2
│   │   │   │   ├── events.out.tfevents.1639951040.nndl.profile-empty
│   │   │   │   └── plugins
│   │   │   │       └── profile
│   │   │   │           └── 2021_12_19_21_57_20
│   │   │   │               ├── nndl.input_pipeline.pb
│   │   │   │               ├── nndl.kernel_stats.pb
│   │   │   │               ├── nndl.memory_profile.json.gz
│   │   │   │               ├── nndl.overview_page.pb
│   │   │   │               ├── nndl.tensorflow_stats.pb
│   │   │   │               ├── nndl.trace.json.gz
│   │   │   │               └── nndl.xplane.pb
│   │   │   └── validation
│   │   │       └── events.out.tfevents.1639951207.nndl.25596.342090.v2
│   │   └── 20211219-234747
│   │       ├── train
│   │       │   ├── events.out.tfevents.1639957668.instance-1.3419.21832.v2
│   │       │   ├── events.out.tfevents.1639957696.instance-1.profile-empty
│   │       │   └── plugins
│   │       │       └── profile
│   │       │           └── 2021_12_19_23_48_16
│   │       │               ├── instance-1.input_pipeline.pb
│   │       │               ├── instance-1.kernel_stats.pb
│   │       │               ├── instance-1.memory_profile.json.gz
│   │       │               ├── instance-1.overview_page.pb
│   │       │               ├── instance-1.tensorflow_stats.pb
│   │       │               ├── instance-1.trace.json.gz
│   │       │               └── instance-1.xplane.pb
│   │       └── validation
│   │           └── events.out.tfevents.1639957792.instance-1.3419.39676.v2
│   └── cifar100
│       ├── 20211219-152901
│       │   ├── train
│       │   │   ├── events.out.tfevents.1639927741.nndl.25554.18734.v2
│       │   │   ├── events.out.tfevents.1639927753.nndl.profile-empty
│       │   │   └── plugins
│       │   │       └── profile
│       │   │           └── 2021_12_19_15_29_13
│       │   │               ├── nndl.input_pipeline.pb
│       │   │               ├── nndl.kernel_stats.pb
│       │   │               ├── nndl.memory_profile.json.gz
│       │   │               ├── nndl.overview_page.pb
│       │   │               ├── nndl.tensorflow_stats.pb
│       │   │               ├── nndl.trace.json.gz
│       │   │               └── nndl.xplane.pb
│       │   └── validation
│       │       └── events.out.tfevents.1639927853.nndl.25554.25614.v2
│       └── 20211219-164544
│           ├── train
│           │   ├── events.out.tfevents.1639932344.nndl.25554.207939.v2
│           │   ├── events.out.tfevents.1639932363.nndl.profile-empty
│           │   └── plugins
│           │       └── profile
│           │           └── 2021_12_19_16_46_03
│           │               ├── nndl.input_pipeline.pb
│           │               ├── nndl.kernel_stats.pb
│           │               ├── nndl.memory_profile.json.gz
│           │               ├── nndl.overview_page.pb
│           │               ├── nndl.tensorflow_stats.pb
│           │               ├── nndl.trace.json.gz
│           │               └── nndl.xplane.pb
│           └── validation
│               └── events.out.tfevents.1639932529.nndl.25554.235725.v2
└── src
    └── modules.py
```

