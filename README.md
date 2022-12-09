# Deep Neural Network Compression

This repository contains the code for the paper Deep Neural Network Compression for Embedded Systems. The repository is structured as follows:

* `models/` contains the source code for the experiments in the paper.
* `media/` contains the images used in the paper and code to generate them.

Within the `models/` directory, there are subdirectories for each of the models used in the paper as well as scripts to train and evaluate each model. Running the scripts on a desktop computer will attempt to train and evaluate the model, running the script on the Jetson Nano 2GB will only evaluate the models. Each of these subdirectories contains the following files:

* `baseline_model.h5` contains the baseline model used in the paper.
* `quantized_model.tflite` contains the quantized model used in the paper.
* `pruned_model.tflite` contains the pruned model used in the paper.
* `clustered_model.tflite` contains the clustered model used in the paper.

## Environmental Setup

Hardware used:  
NVIDIA Jetson Nano 2GB  
Compute cluster with 2x Intel Xeon E5-2667 v3's, NVIDIA Titvan V 12GB, and NVIDIA P6000 24GB

### Setting up on the compute cluster

The compute cluster was accessed through ssh. The development environment on the cluster consisted of a docker container running [NVIDIA's L4T Tensorflow image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorflow) with Tensorflow's model optimization library added. The docker container runs a Jupyter notebook which is accessed through a local machine over ssh.

### Setting up the NVIDIA Jetson Nano

Instructions for the Jetson Nano are found on [NVIDIA's website](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit). However, many issues arose when following these steps. Ultimately, the following steps were taken to get the Jetson Nano up and running:

1. Setup an Ubuntu 18.04 LTS OS on a local machine.
2. Install the NVIDIA SDK Manager on the local machine.
3. Ground the Jetson Nano's Force Recovery pin.
4. Connect the Jetson Nano to the local machine via USB.
5. Use the NVIDIA SDK Manager to flash the Jetson Nano with the latest Jetpack.

## Running the code

The code for the paper is contained in the `models/` directory. The directory contains scripts to train and evaluate each model. Running the script on a desktop computer will attempt to train and evaluate the model, running the script on the Jetson Nano 2GB will only evaluate the models as long as pre-trained weights are available. The scripts are as follows:

* `mnist.py` trains and evaluates the MNIST model.
* `cifar10.py` trains and evaluates the CIFAR-10 model.
* `gtsrb.py` trains and evaluates the GTSRB model.

## Results

The following are results of the different models run on an NVIDIA Jetson Nano 2GB.

### MNIST

---------------------------------
Baseline model results:  
Accuracy: 98.80%  
Latency: 159.77 ms  
Size: 13.39 MB


Quantized model results:  
Accuracy: 98.80%  
Latency: 1.43 ms  
Size: 1.12 MB  


Pruned model results:  
Accuracy: 98.80%  
Latency: 2.51 ms  
Size: 4.45 MB  



Clustered model results:  
Accuracy: 98.80%  
Latency: 2.52 ms  
Size: 4.45 MB  

### CIFAR-10

---------------------------------
Baseline model results:  
Accuracy: 79.40%  
Latency: 170.39 ms  
Size: 15.07 MB  


Quantized model results:  
Accuracy: 79.10%  
Latency: 4.12 ms  
Size: 1.26 MB  



Pruned model results:  
Accuracy: 79.10%  
Latency: 7.30 ms  
Size: 5.01 MB  



Clustered model results:  
Accuracy: 77.50%  
Latency: 7.26 ms  
Size: 5.01 MB  




### GTSRB

---------------------------------
Baseline model results:  
Accuracy: 92.00%  
Latency: 174.72 ms  
Size: 14.09 MB  


Quantized model results:  
Accuracy: 89.50%  
Latency: 2.65 ms  
Size: 1.18 MB  


Pruned model results:  
Accuracy: 88.90%  
Latency: 5.34 ms  
Size: 4.68 MB  


Clustered model results:  
Accuracy: 90.60%  
Latency: 4.85 ms  
Size: 4.68 MB  


