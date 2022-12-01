# Deep Neural Network Compression

This repository contains the code for the paper Deep Neural Network Compression for Embedded Systems. The repository is structured as follows:

* `models/` contains the source code for the experiments in the paper.
* `media/` contains the images used in the paper and code to generate them.

Within the `models/` directory, there are subdirectories for each of the models used in the paper as well as scripts to train and evaluate each model. Running the scripts on a desktop computer will attempt to train and evaluate the model, running the script on the Jetson Nano 2GB will only evaluate the models. Each of these subdirectories contains the following files:

* `baseline_model.h5` contains the baseline model used in the paper.
* `quantized_model.tflite` contains the quantized model used in the paper.
* `pruned_model.tflite` contains the pruned model used in the paper.
* `clustered_model.tflite` contains the clustered model used in the paper.

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


