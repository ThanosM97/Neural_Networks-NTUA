# Deep Learning

## Dataset 
![CIFAR-100](https://datarepository.wolframcloud.com/resources/images/69f/69f1e629-81e6-4eaa-998f-f6734fcd2cb3-io-4-o.en.gif)

The [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), like the CIFAR-10,are labeled subsets of the “80 million tiny images” dataset. They were created by Alex Krizhevsky, first author of the [AlexNet - 2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)), Vinod Nair (first author of [ReLU - 2010](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)) and Geoffrey Hinton.

The CIFAR-100 dataset includes **600000 colored images of 32x32 pixels clustered into 100 categories**. In each category correspond 500 images for training and 100 images for testing, which means that the train-test split is defined. In addition, **each of the 100 categories belongs to one of 20 sub-categories**, for example all of the following categories "maple", "oak", "palm", "pine" and "willow" belong to the subcategory "trees". Each image has two labels, the "fine" one, which corresponds to its category, and the "coarse" label, which corresponds to its subcategory. **We used the "fine" labels and a subset of 20 categories of them**.

## Goal
Image classification using the dataset mentioned above. We were asked to create some architectures from scratch, but also apply Transfer Learning on pretrained Neural Nets. In addition, the purpose of this exercise was not only to optimize the accuracy of our models, but also to find the models that perform best regarding their memory usage and their training time.

## Framework
For this lab exercise we used the TensorFlow2 framework.

## Pre-processing dataset:
The following techniques are applied on different occasions in the notebook:

- Data augmentation
- Image scaling
- TFRecords creation

Then we split the dataset into train/validation/test subsets and used the following pipelined methods to create our input datasets:

- Shuffle
- Repeat
- Batch
- Prefetch


## Transfer Learning

On this part of the lab we applied transfer learning on the pre-trained models of [Keras Applications](https://keras.io/applications/). In addition we used
the state-of-the-art [EfficientNets](https://github.com/qubvel/efficientnet) implemented by [Pavel Yakubovskiy](https://github.com/qubvel).

Original paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

We first examined the effect of the number of trainable layers regarding th accuracy of the model. Then we chose the model that performed best and examined its performance with the use of different optimizers and steps per training epoch. On the results below, we used early stopping with a maximum number of epochs of 200.


## Results

### From-scratch models (32 input size)
| Network           | Trainable parameters | Non-Trainable parameters | Steps/ epoch | Validation Steps | Epochs | Memory   | Accuracy |
|-------------------|----------------------|--------------------------|--------------|------------------|--------|----------|----------|
| Simple CNN        | 128,420              | 0                        | 30           | 5                | 50     | 1.58 MB  | 0,56     |
| First_32          | 116,196              | 0                        | 50           | 10               | 150    | 1.45 MB  | 0,57     |
| First_32_opt      | 116,196              | 0                        | 60           | 10               | 136    | 1.45 MB  | 0,57     |
| second_32         | 247,556              | 64                       | 40           | 5                | 100    | 3.03 MB  | 0.68     |
| second_32_b32     | 247,556              | 64                       | 50           | 5                | 100    | 3.03 MB  | 0.65     |
| second_32_b64     | 247,556              | 64                       | 50           | 5                | 100    | 3.03 MB  | 0.68     |
| second_32_opt     | 247,556              | 64                       | 50           | 5                | 99     | 3.03 MB  | 0,42     |
| second_32_b32_opt | 247,556              | 64                       | 50           | 5                | 68     | 3.03 MB  | 0.5      |
| second_32_b64_opt | 247,556              | 64                       | 50           | 5                | 61     | 3.03 MB  | 0.63     |
| third_32          | 2,557,781            | 448                      | 40           | 10               | 90     | 30.77 MB | 0,65     |
| third_32_opt      | 2,557,781            | 448                      | 50           | 10               | 120    | 30.77 MB | 0,59     |
| fourth_32         | 1,753,252            | 0                        | 40           | 10               | 70     | 21.1 MB  | 0,63     |
| fourth_32_opt     | 1,753,252            | 0                        | 50           | 10               | 100    | 21.1 MB  | 0,63     |

### From-scratch models (78 input size)
| Network           | Trainable parameters | Non-Trainable parameters | Steps/ epoch | Validation Steps | Epochs | Memory   | Accuracy |
|-------------------|----------------------|--------------------------|--------------|------------------|--------|----------|----------|
| first_78         | 902,628   | 0   | 40 | 10 | 100 | 10.89 MB | 0,57 |
| first_78_opt     | 902,628   | 0   | 50 | 10 | 52  | 10.89 MB | 0,56 |
| second_78        | 2,213,636 | 64  | 40 | 10 | 70  | 26.63 MB | 0,66 |
| second_78_opt    | 2,213,636 | 64  | 50 | 10 | 34  | 26.63 MB | 0,61 |
| fifth_78         | 142,866   | 570 | 40 | 10 | 80  | 1.79 MB  | 0,64 |
| fifth_78_b32     | 142,866   | 570 | 40 | 10 | 80  | 1.79 MB  | 0.68 |
| fifth_78_b64     | 142,866   | 570 | 40 | 10 | 80  | 1.79 MB  | 0.64 |
| fifth_78_opt     | 142,866   | 570 | 50 | 10 | 85  | 1.79 MB  | 0,44 |
| fifth_78_b32_opt | 142,866   | 570 | 50 | 10 | 120 | 1.79 MB  | 0.33 |
| fifth_78_b64_opt | 142,866   | 570 | 50 | 10 | 178 | 1.79 MB  | 0.41 |
| sixth_78         | 960,036   | 0   | 40 | 10 | 90  | 11.6 MB  | 0,53 |
| sixth_78_opt     | 960,036   | 0   | 50 | 10 | 54  | 11.6 MB  | 0.53 |

### Different Optimizers for the best model

| Network                       | Trainable parameters | Non-Trainable parameters | Steps/ epoch | Validation Steps | Epochs | Memory  | Accuracy |
|-------------------------------|----------------------|--------------------------|--------------|------------------|--------|---------|----------|
| second_32_b64 (Adam)          | 247,556              | 64                       | 50           | 5                | 100    | 3.03 MB | 0.68     |
| second_model_32_b64_opNadam   | 247,556              | 64                       | 50           | 5                | 100    | 3.03 MB | 0.58     |
| second_model_32_b64_opRMSprop | 247,556              | 64                       | 50           | 5                | 100    | 3.03 MB | 0.51     |
| second_model_32_b64_opSGD     | 247,556              | 64                       | 50           | 5                | 200    | 3.03 MB | 0.18     |

### Transfer Learning
| Network     | Trainable Layers | Trainable parameters | Epochs | Memory | Accuracy |
|-------------|------------------|----------------------|--------|--------|----------|
| VGG16       | All              | 14,765,988           | 20     | 169MB  | 0.79     |
|             | 90%              | 14,764,196           | 24     | 169MB  | 0.79     |
|             | 80%              | 14,727,268           | 19     | 169MB  | 0.79     |
|             | Top              | 51,300               | 198    | 57MB   | 0.5      |
| VGG19       | All              | 20,075,684           | 20     | 230MB  | 0.77     |
|             | 90%              | 20,036,964           | 25     | 230MB  | 0.77     |
|             | 80%              | 19,963,108           | 20     | 229MB  | 0.76     |
|             | Top              | 51,300               | 194    | 77MB   | 0.5      |
| ResNet50    | All              | 23,739,492           | 51     | 273MB  | 0.64     |
|             | 90%              | 23,654,244           | 16     | 272MB  | 0.05     |
|             | 80%              | 23,529,828           | 11     | 271MB  | 0.05     |
|             | Top              | 204,900              | 41     | 93MB   | 0.05     |
| ResNet50V2  | All              | 23,724,260           | 33     | 272MB  | 0.57     |
|             | 80%              | 23,515,492           | 37     | 271MB  | 0.36     |
|             | Top              | 204,900              | 200    | 93MB   | 0.14     |
| MobileNet   | All              | 3,309,476            | 54     | 38ΜB   | 0.62     |
|             | 90%              | 3,306,148            | 29     | 38ΜB   | 0.41     |
|             | 80%              | 3,296,868            | 46     | 38ΜB   | 0.41     |
|             | Top              | 102,500              | 200    | 14ΜB   | 0.05     |
| DenseNet121 | All              | 7,056,356            | 41     | 83MB   | 0.74     |
|             | 90%              | 6,777,572            | 27     | 80MB   | 0.42     |
|             | 80%              | 6,404,964            | 40     | 77MB   | 0.48     |
|             | 70%              | 5,858,340            | 50     | 73MB   | 0.52     |
|             | Top              | 102,500              | 103    | 29MB   | 0.41     |
| Xception    | All              | 21,011,852           | 23     | 241MB  | 0,90     |
|                   | 90% | 20,957,356 | 22  | 241MB | 0.79 |
|                   | 80% | 20,821,036 | 20  | 240MB | 0.77 |
|                   | Top | 204,900    | 74  | 82MB  | 0.59 |
| InceptionV3       | All | 21,973,252 | 26  | 253MB | 0.78 |
|                   | 90% | 21,545,796 | 22  | 249MB | 0.54 |
|                   | 80% | 21,268,564 | 17  | 247MB | 0.3  |
|                   | Top | 204,900    | 107 | 86MB  | 0.28 |
| InceptionResNetV2 | All | 54,429,892 | 36  | 626MB | 0.83 |
|                   | 90% | 53,785,252 | 12  | 620MB | 0.25 |
|                   | 80% | 53,340,852 | 17  | 617MB | 0.44 |
|                   | Top | 153,700    | 99  | 210MB | 0.51 |
| EfficientNetB0 | All | 4,135,648  | 79  | 48MB  | 0.7  |
|                | 90% | 4,130,100  | 87  | 48MB  | 0.68 |
|                | 80% | 4,109,214  | 90  | 48MB  | 0.66 |
|                | Top | 128,100    | 200 | 17MB  | 0.41 |
| EfficientNetB4 | All | 17,727,916 | 74  | 205MB | 0.7  |
|                | 90% | 17,701,860 | 60  | 205MB | 0.67 |
|                | 80% | 17,641,156 | 62  | 204MB | 0.67 |
|                | Top | 179,300    | 200 | 70MB  | 0.36 |
| EfficientNetB7 | All | 64,043,060 | 72  | 738MB | 0.72 |
|                | 90% | 63,955,960 | 78  | 737MB | 0.69 |
|                | 80% | 63,751,216 | 77  | 734MB | 0.7  |
|                | Top | 256,100    | 35  | 249ΜB | 0.05 |

### XCEPTION model using different optimizers

| Optimizer                        | Epochs | Accuracy | Training time |
|----------------------------------|--------|----------|---------------|
| Adam                             | 23     | 0.9      | 6m 43s        |
| Adam with reducing learning rate | 52     | 0.91     | 8m 41s        |
| Adagrad                          | 122    | 0.86     | 20m 43s       |
| Nadam                            | 30     | 0.9      | 5m 5s         |
| SGD                              | 42     | 0.87     | 6m 45s        |

### XCEPTION model with early stopping and reducing learning rate
| Training Steps | Epochs | Accuracy | Training time |
|----------------|--------|----------|---------------|
| 10             | 34     | 0.91     | 2m 29s        |
| 20             | 29     | 0.92     | 3m 1s         |
| 60             | 34     | 0.9      | 7m 32s        |

