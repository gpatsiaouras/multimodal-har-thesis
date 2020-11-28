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

| Experiment  | Execute command |
| ------------- | ------------- |
| Train inertial network | `python train_inertial.py`  |
| Visualize jittering transform in inertial data  | `python visualize_jittering.py` |