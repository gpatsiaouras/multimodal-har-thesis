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

## Train the network
```shell script
$ cd src/
$ python train.py
```
