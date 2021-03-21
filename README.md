## Fast Saliency

This repository contains the code for our two fast saliency detection papers: "**[SAMNet: Stereoscopically Attentive Multi-scale Network for Lightweight Salient Object Detection](https://yun-liu.github.io/papers/(TIP'2021)SAMNet%20-%20Stereoscopically%20Attentive%20Multi-scale%20Network%20for%20Lightweight%20Salient%20Object%20Detection.pdf)**" (IEEE TIP) and "**[Lightweight Salient Object Detection via Hierarchical Visual Perception Learning](https://yun-liu.github.io/papers/(TCYB'2020)Lightweight%20Salient%20Object%20Detection%20via%20Hierarchical%20Visual%20Perception%20Learning.pdf)**" (IEEE TCYB).

We use Python 3.5, PyTorch 0.4.1, cuda 9.0, and numpy 1.17.3 to test the code. We recommend to use [Anaconda](https://www.anaconda.com/).

### Citations

If you are using the code/model/data provided here in a publication, please consider citing:

    @article{liu2020samnet,
      title={{SAMNet}: Stereoscopically Attentive Multi-scale Network for Lightweight Salient Object Detection},
      author={Liu, Yun and Zhang, Xin-Yu and Bian, Jia-Wang and Zhang, Le and Cheng, Ming-Ming},
      journal={IEEE Transactions on Image Processing},
      year={2021},
      publisher={IEEE}
    }

    @article{liu2020lightweight,
      title={Lightweight Salient Object Detection via Hierarchical Visual Perception Learning},
      author={Liu, Yun and Gu, Yu-Chao and Zhang, Xin-Yu and Wang, Weiwei and Cheng, Ming-Ming},
      journal={IEEE Transactions on Cybernetics},
      year={2020},
      publisher={IEEE}
    }
  
### Precomputed saliency maps

Precomputed saliency maps for 12 widely-used saliency datasets (_i.e._, ECSSD, DUT-OMRON, DUTS-TE, HKU-IS, SOD, THUR15K, Judd, MSRA-B, MSRA10K, PASCALS, SED1, SED2) are available in the [`SaliencyMaps`](https://github.com/yun-liu/FastSaliency/tree/master/SaliencyMaps) folder. Note that if a compressed file is larger than 100 Mb, we divided it into two files.

### Testing and training

Before running the code, you should first link the saliency datasets to the ROOT folder:

  ```
  ln -s /path/to/saliency/datasets/ Data
  ```
  
Hence, the generated `Data` folder contains all datasets. We have put data lists in the folder of [`Lists`](https://github.com/yun-liu/FastSaliency/tree/master/Lists). The pretrained models are put in the [`Pretrained`](https://github.com/yun-liu/FastSaliency/tree/master/Pretrained) folder.

#### Testing SAMNet

For example, we use the following command to test SAMNet on the ECSSD dataset:

  ```
  python test.py --file_list ECSSD.txt --pretrained Pretrained/SAMNet_with_ImageNet_pretrain.pth --model Models.SAMNet --savedir ./Outputs
  ```
  
The generated saliency maps will be outputted into the folder of `./Outputs/ECSSD/`.

#### Training SAMNet

By default, we use the DUTS training set for training:

  ```
  python train.py --pretrained Pretrained/SAMNet_backbone_pretrain.pth --model Models.SAMNet --savedir ./Results
  ```
  
#### Testing HVPNet

  ```
  python test.py --file_list ECSSD.txt --pretrained Pretrained/HVPNet_with_ImageNet_pretrain.pth --model Models.HVPNet --savedir ./Outputs
  ```
  
#### Training HVPNet

  ```
  python train.py --pretrained Pretrained/HVPNet_backbone_pretrain.pth --model Models.HVPNet --savedir ./Results
  ```
