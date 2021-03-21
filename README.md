## Fast Saliency

This repository contains the code for our two fast saliency detection papers: "**SAMNet: Stereoscopically Attentive Multi-scale Network for Lightweight Salient Object Detection**" (IEEE TIP) and "**Lightweight Salient Object Detection via Hierarchical Visual Perception Learning**" (IEEE TCYB).

We use PyTorch 0.4.1 and cuda 9.0 to test the code.

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

Precomputed saliency maps for 12 widely-used saliency datasets are available in the `SaliencyMaps` folder. Note that if a compressed file is larger than 100 Mb, we divided it into two files.


### Testing SAMNet

  ```
  python test.py --file_list ECSSD.txt --pretrained Pretrained/SAMNet_with_ImageNet_pretrain.pth --model Models.SAMNet
  ```

### Training SAMNet

  ```
  python train.py --pretrained Pretrained/SAMNet_backbone_pretrain.pth --model Models.SAMNet
  ```
  
### Testing HVPNet

  ```
  python test.py --file_list SED1.txt --pretrained Pretrained/HVPNet_with_ImageNet_pretrain.pth --model Models.HVPNet
  ```
  
### Training HVPNet

  ```
  python train.py --pretrained Pretrained/HVPNet_backbone_pretrain.pth --model Models.HVPNet
  ```
