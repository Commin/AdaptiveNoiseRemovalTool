## [An Adaptive Noise Removal Tool for IoT Image Processing Under Influence of Weather Conditions](https://dl.acm.org/doi/10.1145/3384419.3430393) 

### Introduction
This paper provides a tool that adaptively removes noise due to various weather conditions in image data collected by IoT camera devices. Our tool utilizes a multi-class weather classification model to classify several common weather labels for input images. Then, it removes noise by running suitable noise removal algorithms, which also supports edge computing for data analysis by adjusting image data size so that the data fits in a small memory in each edge device.

## Prerequisites

- Python 3.6, PyTorch=1.3.1, torchvision=0.4.2
- Requirements: opencv-python, numpy
- Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.6.5(higher versions also work well)
- Details for `environment.yml`

## Datasets

PRN and PReNet are evaluated on four datasets*: 
Rain100H [1], Rain100L [1], Rain12 [2] and Rain1400 [3]. 
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), 
and place the unzipped folders into `./datasets/test/`.

To train the models, please download training datasets: 
RainTrainH [1], RainTrainL [1] and Rain12600 [3] from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), 
and place the unzipped folders into `./datasets/train/`. 

*_We note that:_

_(i) The datasets in the website of [1] seem to be modified. 
    But the models and results in recent papers are all based on the previous version, 
    and thus we upload the original training and testing datasets 
    to [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg) 
    and [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g)._ 

_(ii) For RainTrainH, we strictly exclude 546 rainy images that have the same background contents with testing images.
    All our models are trained on remaining 1,254 training samples._


## Getting Started

### Testing

Run `weather.ipynb`.

1. Download pre-trained model.
2. Load input image.
3. Run our tool to classify its weather and noise removal.
4. Output the generated output image.


# Citation

```
@inproceedings{Chen2020Image,
author = {Chen, Mingkang and Sun, Jingtao and Saga, Kazushige and Tanjo, Tomota and Aida, Kento},
title = {An Adaptive Noise Removal Tool for IoT Image Processing under Influence of Weather Conditions: Poster Abstract},
year = {2020},
publisher = {Association for Computing Machinery},
pages = {655–656}
}


 ```
