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

`python main_train_stganmr.py --opt ./options/STGAN/example/train_stganmr_CCnpi_G1D30.json`

To train EES-GAN on CC:

`python main_train_eesganmr.py --opt ./options/EESGAN/example/train_eesganmr_CCnpi_G1D30.json`

To train TES-GAN on CC:

`python main_train_tesganmr.py --opt ./options/TESGAN/example/train_tesganmr_CCnpi_G1D30.json`

To test ST-GAN on CC:

`python main_test_stganmr_CC.py --opt ./options/STGAN/example/test/test_stganmr_CCnpi_G1D30.json`

To test EES-GAN on CC:

`python main_test_eesganmr_CC.py --opt ./options/EESGAN/example/test/test_eesganmr_CCnpi_G1D30.json`

To test TES-GAN on CC:

`python main_test_tesganmr_CC.py --opt ./options/TESGAN/example/test/test_tesganmr_CCnpi_G1D30.json`


This repository is based on:

Swin Transformer for Fast MRI 
([code](https://github.com/ayanglab/SwinMR) and [paper](https://www.sciencedirect.com/science/article/pii/S0925231222004179));

SwinIR: Image Restoration Using Swin Transformer 
([code](https://github.com/JingyunLiang/SwinIR) and [paper](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.html));

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
([code](https://github.com/microsoft/Swin-Transformer) and [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)).


