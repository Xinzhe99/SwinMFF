# SwinMFF
Official code for "SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network"

## Train
### Make dataset for training SwinMFF
1. Download [DUTS](https://pan.baidu.com/s/1XCCbFi-uNNXWlig0CNBoIA?pwd=cite)
2. Extract it to the project path
3. Run the following code to get the data set needed for training

`python .\make_dataset.py --mode='TR'`

`python .\make_dataset.py --mode='TE'`

### Start to train
`python .\train.py`
## Test
Download Weights in [Baidu](https://pan.baidu.com/s/15-5_TzVa-ZypyceiMSyMkg?pwd=cite) and put in the project path
### Lytro
`python .\predict.py --dataset_path='./assets/Lytro' --model_path='./checkpoint.ckpt' --is_gray=False`
### MFFW
`python .\predict.py --dataset_path='./assets/MFFW' --model_path='./checkpoint.ckpt' --is_gray=False`
### MFI-WHU
`python .\predict.py --dataset_path='./assets/MFI-WHU' --model_path='./checkpoint.ckpt' --is_gray=False`
### Others
`python .\predict.py --dataset_path='your path' --model_path='your path' --is_gray=False/True`
## Results of various methods
| Method     | Download link                                          |
|------------|--------------------------------------------------------|
| CNN        | [https://github.com/yuliu316316/CNN-Fusion](https://github.com/yuliu316316/CNN-Fusion) |
| ECNN       | [https://github.com/mostafaaminnaji/ECNN](https://github.com/mostafaaminnaji/ECNN) |
| SESF       | [https://github.com/Keep-Passion/SESF-Fuse](https://github.com/Keep-Passion/SESF-Fuse) |
| MFIF-GAN   | [https://github.com/ycwang-libra/MFIF-GAN](https://github.com/ycwang-libra/MFIF-GAN) |
| MSFIN      | [https://github.com/yuliu316316/MSFIN-Fusion](https://github.com/yuliu316316/MSFIN-Fusion) |
| ZMFF       | [https://github.com/junjun-jiang/ZMFF](https://github.com/junjun-jiang/ZMFF) |
| IFCNN-MAX  | [https://github.com/uzeful/IFCNN](https://github.com/uzeful/IFCNN) |
| U2Fusion   | [https://github.com/hanna-xu/U2Fusion](https://github.com/hanna-xu/U2Fusion) |
| SDNet      | [https://github.com/HaoZhang1018/SDNet](https://github.com/HaoZhang1018/SDNet) |
| MFF-GAN    | [https://github.com/HaoZhang1018/MFF-GAN](https://github.com/HaoZhang1018/MFF-GAN) |
| SwinFusion | [https://github.com/Linfeng-Tang/SwinFusion](https://github.com/Linfeng-Tang/SwinFusion) |
| FusionDiff | [https://github.com/lmn-ning/ImageFusion](https://github.com/lmn-ning/ImageFusion) |

Result of various learning-based methods compared can be download in [Baidu]([https://pan.baidu.com/s/15-5_TzVa-ZypyceiMSyMkg?pwd=cite](https://pan.baidu.com/s/1aDmgPnbUwElQ-t_4lQtEww?pwd=cite))

Includes traditional methods download in [https://github.com/yuliu316316/MFIF](https://github.com/yuliu316316/MFIF)

# If our work is helpful to you, please help cite our work
```
@article{xie2024swinmff,
  title={SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={The Visual Computer},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
```
