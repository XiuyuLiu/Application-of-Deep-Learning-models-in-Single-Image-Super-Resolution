# Optimization of Single Image Super Resolution based on Deep Learning
Contributor: Xiuyu Liu; Qingqing Cao
## Description
Single image super-resolution (SISR), which aims to reconstruct a high-resolution (HR) image from a low-resolution (LR) observation, has been an active research topic in the area of image processing in recent decades.  Traditional methods improving image’s resolution include interpolation-based and construction based methods. In the near few years, people used learning-based methods to map low-resolution images to high-resolution images. In this project, we start from basic convolution neural networks (SRCNN) and experiment with different upsampling methods, architectures, and loss functions. Finally, we combined traditional signal processing methods with neural networks to improve images’ resolution.
The SRCNN models implemented in this project include:
- Baseline SRCNN model
- Modified SRCNN models with different up-sampling methods
- Modified SRCNN models with differrent archiectures
- Modified SRCNN models with different loss functions
- Applications of Signal Processing in SRCNN model

Here is the framework of this project:
<img src="images/Picture1.png" alt="drawing" width="500"/>
## Dataset of Images
In this project, we use the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) for training and validation. It is a 1000-image dataset containing RGB images with a large diversity of contents. When training our deep learning model, we use 800 high resolution images and acquire corresponding low resolution using 4X downsampling. The images are also randomly cropped, flipped, and rotated to increase diversity. We are using another 100 images for validation. 
## Baseline SRCNN model
SRCNN is a deep learning model that was proposed in 2014 and  directly learns an end-to-end mapping between the low/high-resolution images. As shown below, the deep learning model has a lightweight structure, containing only three convolution layers. This network was published in the paper, "Image Super-Resolution Using Deep Convolutional Networks" by Chao Dong, et al. in 2014. Referenced Research Paper https://arxiv.org/abs/1501.00092. Our first task in this project is to implement the SRCNN model from scratch and use it as a baseline model. 

You can find our implementation [Here](https://github.com/XiuyuLiu/Optimization-of-SISR-based-on-DL/blob/main/Task1_architecture.ipynb). In the SRCNN model, a deep convolutional neural network takes the low-resolution image as the input and outputs the high-resolution one. The input images are first normalized and upsampled using bilinear interpolation methods to match the size of input and output images. Then, three convolution layers are used with ‘same’ padding and ‘relu’ activation. Then, the output images are denormalized to restore the RGB format. Mean Squared Loss is used as a loss function and an Adam optimizer with learning rate decay is used to train the model. In this project, the SRCNN model is developed in Keras, which is a high-level deep learning API on the basis of Tensorflow that provides a convenient way to define and train models. 
<img src="images/Picture2.png" alt="drawing" width="500"/> (credit to: https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c)

## Modified SRCNN models with different up-sampling methods
To learn a map between low-resolution images and high-resolution images, in SRCNN, the low resolution images are first interpolated to obtain a ‘coarse’ high resolution image. The advantage of handling upsampling by the Bilinear interpolation is that the CNN only needs to learn  how to refine the coarse image, which is simpler. The disadvantage is that the predefined upsampling methods may amplify noise and cause blurring. Inspired by this, here we explore using a sub-pixel convolution layer as an upsampling layer and experiment with four different up-sampling methods. As is shown below, the four methods include pre-upsampling, post-upsampling, progressive upsampling, and up and down (iterative) upsampling methods.

<img src="images/Picture3.png" alt="drawing" width="500"/>


## Modified SRCNN models with different Architectures
Although the baseline SRCNN model has demonstrated its capability to learn a map between low-resolution image and high-resolution image, it is still a relatively shallow neural network model with only three layers. Inspired by this point, we try to improve the performance of the SRCNN model by adding more layers. At the same time, to combat vanishing gradient problems and to accelerate convergence, we introduce global skip connection (residual block) and different kernel sizes in one layer (inception block). Each residual block contains two convolution layers and allows local skip connection. Each inception block contains 4 parallel convolution layers with kernel size of 9, 5, 3, and 1. The results of convolution layers are concatenated to generate the output.

<img src="images/Picture4.png" alt="drawing" width="500"/>

## Modified SRCNN models with Different Loss Functions
Loss functions are used to measure the difference between the high-resolution images (ground truth) and super-resolution images (predicted results). This difference can be used to optimize the deep learning model. In the baseline SRCNN model, we use mean-squared-error (MSE) to quantify the pixel-wise difference between ground-truth image and generated image. This loss function directly maximizes the PSNR metric value and is reported to often generate images lacking high-frequency details. In this project, we also experimented with the content loss and texture loss to improve the generated image quality.

<img src="https://user-images.githubusercontent.com/46135164/135192308-4bfec792-00bb-49c8-892a-bfae769c4ca4.png" alt="drawing" width="300"/>

In the above equation, content loss is calculated using the high-level features of the HR image and SR image, which is extracted from a pre-trained VGG-19 net. Here we use the max-pooling layer results in the first and last blocks from the pre-trained network. Then, the texture loss is defined as the correlation between different feature channels. Then, we use the Gram matrix to calculate the correlation between the feature maps. The texture features are also extracted from the pre-trained VGG-19 net. 
In the end, we combine the pixel loss, content loss, and texture loss to build a loss function (Perceptual Loss) that minimizes the pixel error, content error, and texture error at the same time:
<img src="https://user-images.githubusercontent.com/46135164/135192500-ab0e3e6a-ac13-4571-8b4a-b1c27ea5db3c.png" alt="drawing" width="300"/>

## Applications of Signal Processing in Super-Resolution Problem
In this section, we applied a traditional edge enhancement filter on the upsampled low-resolution images and output super-resolution images, as pre-processing and post-processing respectively. Edge enhancement technique aims to enhance the visibility of edges. It is widely applied in the printing and publishing industry,  radiographic images, manufacturing and military applications. 
The reason we apply this technique is that we usually observe blurry artifacts in low-resolution images, and edges usually contribute significantly to high-frequency image components. We hope to improve the sharpness and esthetic aspect of a picture by applying the edge-enhancement filter before and after the neural network.



