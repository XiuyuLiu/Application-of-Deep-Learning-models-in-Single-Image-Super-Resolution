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

## Baseline SRCNN model
SRCNN is a deep learning model that was proposed in 2014 and  directly learns an end-to-end mapping between the low/high-resolution images. As shown below, the deep learning model has a lightweight structure, containing only three convolution layers. Our first task in this project is to implement the SRCNN model from scratch and use it as a baseline model. The following sections show the details of the implementation.

<img src="https://user-images.githubusercontent.com/46135164/135186114-ee22c66d-d4d6-4c58-8812-59980d296ac6.png" alt="drawing" width="800"/>










<img src="https://user-images.githubusercontent.com/46135164/135184984-5834a621-7bbf-492d-93dd-43c98ea75282.png" alt="drawing" width="800"/>

