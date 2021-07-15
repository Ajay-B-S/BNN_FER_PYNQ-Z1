# BNN_FER_PYNQ-Z1
Binary Neural Network based Facial emotion recognition on Pynq-Z1 FPGA board.

BNN paper (FINN) : https://arxiv.org/pdf/1612.07119.pdf 
Pynq setup : https://pynq.readthedocs.io/en/v1.3/index.html
BNN github : https://github.com/Xilinx/BNN-PYNQ
Datasets used: FER2013 and JAFFE.
Clone : git clone https://github.com/Xilinx/BNN-PYNQ.git --recursive

Training : 
1. Training is done on PC.
2. In the cloned repo under bnn folder, we choosed the MNIST training example and with this reference, we did the training for Facial emotion recognition using FER2013 for 6 emotions (HAPPY, SAD, ANGRY, DISGUST, FEAR, SURPRISE)

Inference:
The weights (1/2 bit) and activations (1/2 bit) are uploaded to the project created on jupyter notebook:
The basic project is cloned from https://github.com/Xilinx/BNN-PYNQ.git
Pynq-Z1 board interacts wtih jupyter notebook for the inference.
Inference code is modified and is in the file : LFC-BNN_MNIST_VJ_LBP_Final.py
