# SwinMFF
Official code for "SwinMFF: Revitalizing and Setting a Benchmark for End-to-End Multi-Focus Image Fusion"
![image](https://github.com/Xinzhe99/SwinMFF/assets/113503163/03ba666e-d66d-423e-9bc3-27642538aa41)
![image](https://github.com/Xinzhe99/SwinMFF/assets/113503163/4c14f8e9-7639-4ac5-8551-270bdd74c427)
![image](https://github.com/Xinzhe99/SwinMFF/assets/113503163/36f4f5fe-cedf-4b40-a7ff-7115dd91f5f2)

| Method      | *EI*     | *Q^{ab/f}* | *STD*    | *SF*     | *AVG*    | *MI*     | *EN*     | *VIF*    |
|-------------|----------|------------|----------|----------|----------|----------|----------|----------|
| DWT         | 70.7942  | 0.6850     | 57.2776  | 19.3342  | 6.8336   | **15.0872** | 7.5436   | 1.1114   |
| DTCWT       | 70.5666  | 0.6929     | 57.2315  | 19.3204  | 6.8134   | 15.0791  | 7.5396   | 1.1079   |
| NSCT        | 70.4289  | 0.6901     | 57.3601  | 19.2662  | 6.8027   | 15.0816  | **7.5408** | 1.1249   |
| GFF         | 70.5179  | 0.6998     | 57.4451  | 19.2947  | 6.8058   | 15.0716  | 7.5358   | 1.1277   |
| SR          | 70.2498  | 0.6944     | 57.3795  | 19.2819  | 6.7818   | 15.0650  | 7.5325   | 1.1208   |
| ASR         | 70.3342  | 0.6951     | 57.3616  | 19.2818  | 6.7897   | 15.0654  | 7.5327   | 1.1201   |
| MWGF        | 69.8052  | **0.7037** | 57.4136  | 19.1900  | 6.7273   | 15.0669  | 7.5334   | 1.1343   |
| ICA         | 68.3180  | 0.6766     | 56.9383  | 18.5968  | 6.6125   | 15.0655  | 7.5327   | 1.0708   |
| NSCT-SR     | 70.6705  | 0.6995     | 57.3924  | 19.3355  | 6.8213   | 15.0676  | 7.5338   | 1.1251   |
| Proposed    | **72.1691** | 0.6824 | **57.8876** | **19.7382** | **6.9511** | **15.0801** | **7.5400** | **1.1757** |
| SSSDI       | 70.7102  | 0.6966     | 57.4770  | 19.3567  | 6.8234   | 15.0668  | 7.5334   | 1.1309   |
| QUADTREE    | 70.8957  | 0.7027     | 57.5334  | 19.4163  | 6.8412   | 15.0684  | 7.5342   | 1.1368   |
| DSIFT       | 70.9808  | **0.7046** | 57.5319  | 19.4194  | 6.8493   | 15.0688  | 7.5344   | 1.1381   |
| SRCF        | **71.0810** | 0.7036 | **57.5394** | **19.4460** | **6.8607** | **15.0690** | **7.5345** | **1.1374** |
| GFDF        | 70.6258  | **0.7049** | 57.4973  | 19.3312  | 6.8145   | 15.0674  | 7.5337   | 1.1336   |
| BRW         | 70.6777  | 0.7040     | 57.5020  | 19.3433  | 6.8200   | 15.0675  | 7.5337   | 1.1336   |
| MISF        | 70.4148  | 0.6984     | 57.4437  | 19.2203  | 6.7945   | 15.0671  | 7.5335   | 1.1222   |
| Proposed    | **72.1691** | 0.6824 | **57.8876** | **19.7382** | **6.9511** | **15.0801** | **7.5400** | **1.1771** |


| Method       | Year | Journal/Conference | Network           | *EI*    | *Q^{ab/f}* | *STD*   | *SF*    | *AVG*   | *MI*    | *EN*    | *VIF*   |
|--------------|------|--------------------|-------------------|---------|------------|---------|---------|---------|---------|---------|---------|
| CNN          | 2017 | Information Fusion | CNN               | 70.3238 | 0.7019     | 57.4354 | 19.2295 | 6.7860  | 15.0663 | 7.5331  | 1.1255  |
| ECNN         | 2019 | Information fusion | CNN               | 70.7432 | 0.7030     | 57.5089 | 19.3837 | 6.8261  | 15.0675 | 7.5338  | 1.1337  |
| SESF         | 2020 | Neural. Comput. Appl. | CNN            | 70.9403 | 0.7031     | 57.5495 | 19.4158 | 6.8448  | 15.0696 | 7.5348  | 1.1395  |
| MFIF-GAN     | 2021 | SPIC               | GAN               | 71.0395 | 0.7029     | 57.5430 | 19.4370 | 6.8560  | 15.0690 | 7.5345  | 1.1393  |
| MSFIN        | 2021 | IEEE TIM           | CNN               | 71.0914 | 0.7045     | 57.5642 | 19.4438 | 6.8602  | 15.0695 | 7.5348  | 1.1420  |
| ZMFF         | 2023 | Information Fusion | DIP               | 70.8298 | 0.6635     | 57.0347 | 18.9707 | 6.8045  | 15.0735 | 7.5368  | 1.1331  |
| Proposed     | 2024 | Information Fusion | Transformer       | 72.1691 | 0.6824     | 57.8876 | 19.7382 | 6.9511  | 15.0801 | 7.5400  | 1.1757  |

| Method       | Year | Journal/Conference | Network           | *EI*    | *Q^{ab/f}* | *STD*   | *SF*    | *AVG*   | *MI*    | *EN*    | *VIF*   |
|--------------|------|--------------------|-------------------|---------|------------|---------|---------|---------|---------|---------|---------|
| IFCNN-MAX    | 2020 | Information Fusion | CNN               | 70.9193 | 0.6784     | 57.4896 | 19.3793 | 6.8463  | 15.0722 | 7.5361  | 1.1322  |
| U2Fusion     | 2020 | IEEE TPAMI         | CNN               | 59.8957 | 0.6190     | 51.9356 | 14.9334 | 5.6515  | 14.6153 | 7.3077  | 0.9882  |
| SDNet        | 2021 | IJCV               | CNN               | 60.3437 | 0.6441     | 55.2655 | 16.9252 | 5.8725  | 14.9332 | 7.4666  | 0.9281  |
| MFF-GAN      | 2021 | Information Fusion | GAN               | 66.0601 | 0.6222     | 55.1920 | 18.4022 | 6.4089  | 14.8153 | 7.4076  | 1.0084  |
| SwinFusion   | 2022 | IEEE/CAA JAS       | CNN & Transformer | 62.8130 | 0.6597     | 56.8142 | 16.6430 | 5.9862  | 15.0476 | 7.5238  | 1.0685  |
| FusionDiff   | 2024 | ESWA               | Diffusion Model   | 67.4911 | 0.6744     | 56.1372 | 18.8483 | 6.5325  | 14.9817 | 7.4909  | 1.0448  |
| Proposed     | 2024 | TBD | Transformer       | 72.1691 | 0.6824     | 57.8876 | 19.7382 | 6.9511  | 15.0801 | 7.5400  | 1.1771  |

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
Result of various learning-based methods compared can be download in [Baidu](https://pan.baidu.com/s/15-5_TzVa-ZypyceiMSyMkg?pwd=cite](https://pan.baidu.com/s/13pxQzkF1wXnJ1paZNFhwwg?pwd=cite)(Includes traditional methods download in [https://github.com/yuliu316316/MFIF](https://github.com/yuliu316316/MFIF)
# Acknowledgements
The research was supported by the Hainan Provincial Joint Project of Sanya Yazhou Bay Science and Technology City (No: 2021JJLH0079), Innovational Fund for Scientific and Technological Personnel of Hainan Province (NO. KJRC2023D19), and the Hainan Provincial Joint Project of Sanya Yazhou Bay Science and Technology City (No. 2021CXLH0020). Thanks for help by Hainan Provincial Observatory of Ecological Environment and Fishery Resource in Yazhou Bay. Also, we want to thank Chloe Alex Schaff for her contribution in polishing the article.

# If our work is helpful to you, please help cite our work
