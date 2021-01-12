# CIZSL.v2
CIZSL++: Creativity Inspired Generative Zero-shot Learning, Mohamed Elhoseiny, Kai Yi, Mohamed Elfeki, Arxiv, 2020 

# Requirements
Python 3.5

Pytorch 1.6

sklearn, scipy, matplotlib, numpy, random, copy


# Processed Feature Data 
You can download the text-based dataset at [dataset CUBird and NABird](https://www.dropbox.com/s/9qovr86kgogkl6r/CUB_NAB_Data.zip). For attribute-based data, you can access to [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly). 

Please put the uncompressed data to the folder "data".

# Reproduce Key Results
### CIZSLv1 updated version
'python train_CIZSL.py --dataset CUB --splitmode 'hard' --creativity_weight 0.1'
'python train_CIZSL.py --dataset CUB --splitmode 'easy' --creativity_weight 0.0001'
'python train_CIZSL.py --dataset NAB --splitmode 'hard' --creativity_weight 0.1'
'python train_CIZSL.py --dataset NAB --splitmode 'easy' --creativity_weight 1'

### TODO

# Reference
- Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng and Ahmed Elgammal "A Generative Adversarial Approach for Zero-Shot Learning from Noisy Texts", CVPR, 2018
- Mohamed Elhoseiny, Mohamed Elfeki, Creativity Inspired Zero Shot Learning, Thirty-sixth International Conference on Computer Vision (ICCV), 2019

If you find this code is useful, please cite:

'''
@article{elhoseiny2021cizsl++,
  title={CIZSL++: Creativity Inspired Generative Zero-Shot Learning},
  author={Elhoseiny, Mohamed and Yi, Kai and Elfeki, Mohamed},
  journal={arXiv preprint arXiv:2101.00173},
  year={2021}
}
'''






