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

## Experiments / Training / Testing
> Make you activated the virtual environment by running `source venv/bin/activate`
> prior to running any of the following commands

| Experiment  | Execute command | Notes |
| ------------- | ------------- | ------------- |
| Train inertial network | `python train_inertial.py`  | Can be optionally called with a yaml file to load parameters (e.g parameters/inertial/optimized.yaml). Saves the model weights automatically after a complete training in <root>/saved_models/YYYYMMDD_HHSS_CNN1D_epX_bsX.pt |
| Test inertial network | `python test_inertial.py <root>/saved_models/my_saved_model.pt` | Tests the inertial CNN1D network with the test dataset. Must be run with saved model weights from the <root>/saved_models/ directory |
| Visualize jittering transform in inertial data  | `python visualize_jittering.py` |
| Visualize sampler transform in inertial data | `python visualize_sampler.py` |