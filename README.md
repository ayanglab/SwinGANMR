# SwinGANMR

by Jiahao Huang (j.huang21@imperial.ac.uk)

This is the official implementation of our proposed ST-GAN, EES-GAN and TES-GAN:

Fast MRI Reconstruction: How Powerful Transformers Are?

Please cite:

```
@ARTICLE{2022arXiv220109400H,
       author = {{Huang}, Jiahao and {Wu}, Yinzhe and {Wu}, Huanjun and {Yang}, Guang},
        title = "{Fast MRI Reconstruction: How Powerful Transformers Are?}",
      journal = {arXiv e-prints},
     keywords = {Electrical Engineering and Systems Science - Image and Video Processing, Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
         year = 2022,
        month = jan,
          eid = {arXiv:2201.09400},
        pages = {arXiv:2201.09400},
archivePrefix = {arXiv},
       eprint = {2201.09400},
 primaryClass = {eess.IV},
}
```

![Overview_of_SwinGANMR](./tmp/files/SwinGANMR.png)


## Requirements

matplotlib==3.3.4

opencv-python==4.5.3.56

Pillow==8.3.2

pytorch-fid==0.2.0

scikit-image==0.17.2

scipy==1.5.4

tensorboardX==2.4

timm==0.4.12

torch==1.9.0

torchvision==0.10.0


## Training and Testing
Use different options (json files) to train different networks.

### Calgary Campinas multi-channel dataset (CC) 

To train ST-GAN on CC:

`python main_train_stganmr.py --opt ./options/STGAN/train_stganmr_CC_G1D30.json`

To train EES-GAN on CC:

`python main_train_eesganmr.py --opt ./options/EESGAN/train_eesganmr_CC_G1D30.json`

To train TES-GAN on CC:

`python main_train_tesganmr.py --opt ./options/TESGAN/train_tesganmr_CC_G1D30.json`

To test ST-GAN on CC:

`python main_test_stganmr_CC.py --opt ./options/STGAN/test/test_stganmr_CC_G1D30_sample.json`

To test EES-GAN on CC:

`python main_test_eesganmr_CC.py --opt ./options/EESGAN/test/test_eesganmr_CC_G1D30_sample.json`

To test TES-GAN on CC:

`python main_test_tesganmr_CC.py --opt ./options/TESGAN/test/test_tesganmr_CC_G1D30_sample.json`


This repository is based on:

Swin Transformer for Fast MRI 
([code](https://github.com/ayanglab/SwinMR) and [paper](https://arxiv.org/abs/2201.03230));

SwinIR: Image Restoration Using Swin Transformer 
([code](https://github.com/JingyunLiang/SwinIR) and [paper](https://arxiv.org/abs/2108.10257));

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
([code](https://github.com/microsoft/Swin-Transformer) and [paper](https://arxiv.org/abs/2103.14030)).