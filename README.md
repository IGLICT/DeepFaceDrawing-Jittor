# DeepFaceDrawing: Deep Generation of Face Images from Sketches (Jittor)

This system is implemented using the <a href="https://github.com/Jittor/Jittor" target="_blank">Jittor</a>, and you need to install Jittor first. 

HomePage: <a href="http://www.geometrylearning.com/DeepFaceDrawing/" target="_blank">http://www.geometrylearning.com/DeepFaceDrawing/</a>

![Teaser Image](images/teaser.jpg)

## Abstract
Recent deep image-to-image translation techniques allow fast generation of face images from freehand sketches. However, existing solutions tend to overfit to sketches, thus requiring professional sketches or even edge maps as input. To address this issue, our key idea is to implicitly model the shape space of plausible face images and synthesize a face image in this space to approximate an input sketch. We take a local-to-global approach. We first learn feature embeddings of key face components, and push corresponding parts of input sketches towards underlying component manifolds defined by the feature vectors of face component samples. We also propose another deep neural network to learn the mapping from the embedded component features to realistic images with multi-channel feature maps as intermediate results to improve the information flow. Our method essentially uses input sketches as soft constraints and is thus able to produce high-quality face images even from rough and/or incomplete sketches. Our tool is easy to use even for non-artists, while still supporting fine-grained control of shape details. Both qualitative and quantitative evaluations show the superior generation ability of our system to existing and alternative solutions. The usability and expressiveness of our system are confirmed by a user study.

## Prerequisites

1. System

　- Ubuntu 16.04 or later

　- NVIDIA GPU + CUDA 9.2 

2. Software

　- Python 3.7

　- Jittor. More details in <a href="https://github.com/Jittor/Jittor" target="_blank">Jittor</a>

  ```
  sudo apt install python3.7-dev libomp-dev

  sudo python3.7 -m pip install git+https://github.com/Jittor/jittor.git

  python3.7 -m jittor.test.test_example
  ```

　- Packages

  ```
  sh install sh
  ```

## How to use

Drawing sketch using DeepFaceDrawing GUI. 

  ```
  python3.7 demo.py
  ```

## Citation

If you found this code useful please cite our work as:

    @article{chenDeepFaceDrawing2020,
        author = {Chen, Shu-Yu and Su, Wanchao and Lin, Gao and Xia, Shihong and Fu, Hongbo},
        title = {{DeepFaceDrawing}: Deep Generation of Face Images from Sketches},
        journal = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH  2020)},
        year = {2020},
        volume = 39,
        pages = {72:1--72:16},
        number = 4
    }