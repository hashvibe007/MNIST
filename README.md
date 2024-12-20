# MNIST Neural Network

![Model Tests](https://github.com/{username}/{repository}/actions/workflows/model_tests.yml/badge.svg)

[Your existing README content...]

## Model Details
- Parameters: < 25,000
- Test Accuracy: > 99.4%

## Tests
This repository includes automated tests that verify:
1. The model has less than 25,000 parameters
2. The model achieves > 99.4% accuracy on the MNIST test set

The test status badge above shows whether the model currently passes these requirements.

<!-- MNIST dataset -->
---------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 24, 24, 24]           3,480
              ReLU-6           [-1, 24, 24, 24]               0
       BatchNorm2d-7           [-1, 24, 24, 24]              48
           Dropout-8           [-1, 24, 24, 24]               0
            Conv2d-9           [-1, 32, 22, 22]           6,944
             ReLU-10           [-1, 32, 22, 22]               0
      BatchNorm2d-11           [-1, 32, 22, 22]              64
          Dropout-12           [-1, 32, 22, 22]               0
        MaxPool2d-13           [-1, 32, 11, 11]               0
           Conv2d-14           [-1, 16, 11, 11]             528
           Conv2d-15              [-1, 8, 9, 9]           1,160
             ReLU-16              [-1, 8, 9, 9]               0
      BatchNorm2d-17              [-1, 8, 9, 9]              16
          Dropout-18              [-1, 8, 9, 9]               0
           Conv2d-19             [-1, 16, 7, 7]           1,168
             ReLU-20             [-1, 16, 7, 7]               0
      BatchNorm2d-21             [-1, 16, 7, 7]              32
          Dropout-22             [-1, 16, 7, 7]               0
           Conv2d-23             [-1, 20, 5, 5]           2,900
             ReLU-24             [-1, 20, 5, 5]               0
      BatchNorm2d-25             [-1, 20, 5, 5]              40
          Dropout-26             [-1, 20, 5, 5]               0
           Conv2d-27             [-1, 24, 5, 5]             504
           Conv2d-28             [-1, 18, 3, 3]           3,906
             ReLU-29             [-1, 18, 3, 3]               0
      BatchNorm2d-30             [-1, 18, 3, 3]              36
          Dropout-31             [-1, 18, 3, 3]               0
           Conv2d-32             [-1, 10, 1, 1]           1,630
             ReLU-33             [-1, 10, 1, 1]               0
      BatchNorm2d-34             [-1, 10, 1, 1]              20
          Dropout-35             [-1, 10, 1, 1]               0
        AvgPool2d-36             [-1, 10, 1, 1]               0
           Conv2d-37             [-1, 10, 1, 1]             110
================================================================
Total params: 22,778
Trainable params: 22,778
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.34
Params size (MB): 0.09
Estimated Total Size (MB): 1.43
----------------------------------------------------------------

<!-- Train & Test accuracy in 20 EPoch -->



