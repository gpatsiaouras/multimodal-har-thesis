# Multimodal Human Activity Recognition (HAR) - Thesis project

## Description
>TBD

## Prerequisites
#### 1. Download UTD-MHAD dataset
To train the algorithm you need to have the [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) downloaded in the `./datasets` directory.   
To download the dataset run the following command: 
```shell script
$ ./download_datasets.sh
```
#### 2. Create a virtual environment (python-venv)
```shell script
$ python3.6 -m venv venv
$ source venv/bin/activate
```

## Training / Testing
> Make sure you activated the virtual environment by running `source venv/bin/activate`
> prior to running any of the following commands

| Train/Test  | Execute command | Notes |
| ------------- | ------------- | ------------- |
| Train inertial network | `python train_inertial.py`  | Can be optionally called with a yaml file to load parameters (e.g parameters/inertial/optimized.yaml). Saves the model weights automatically after a complete training in <root>/saved_models/YYYYMMDD_HHSS_CNN1D_epX_bsX.pt |
| Test inertial network | `python test_inertial.py <root>/saved_models/my_saved_model.pt` | Tests the inertial CNN1D network with the test dataset. Must be run with saved model weights from the <root>/saved_models/ directory |
| Train RGB network | `python train_rgb.py` | Trains a CNN2D network in SDFDI images generated from video files. Can be called with a yaml file to load parameters. Saves the model weights automatically after a complete training in <root>/saved_models/YYYYMMDD_HHSS_mobilenet_v2_epX_bsX.pt |
| Test RGB network | `python test_rgb.py <root>/saved_models/my_saved_model.pt` | Tests the rgb mobilenet_v2 network with the test dataset. Must be run with saved model weights from the <root>/saved_models/ directory |

## Visualizations of transforms
> Make sure you activated the virtual environment by running `source venv/bin/activate`
> prior to running any of the following commands

| Transform  | Execute command | Notes |
| ------------- | ------------- | ------------- |
| Visualize jittering transform in inertial data  | `python visualize_jittering.py` |
| Visualize sampler transform in inertial data | `python visualize_sampler.py` |
| Visualize SDFDI transformation of a video | `python visualize_sdfdi.py` | It shows the original video and then prints the SDFDI image to make the comparison clear |
| Visualize SDFDI (live) using a camera | `python visualize_sdfdi_camera.py` | Performs the SDFDI calculation for every 30 frames of the video from your webcam |
| Visualize Skeleton | `python visualize_skeleton.py` | Visualizes the skeleton in 3D with joint locations, bones and joint names |

## TO DO (code-wise)
* Refactor the code for training and testing to work for multiple modalities and models, in order to avoid duplicated code
