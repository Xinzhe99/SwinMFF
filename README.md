# SwinMFF
Official code for "SwinMFF: Revitalizing and Setting a Benchmark for End-to-End Multi-Focus Image Fusion"
![image](https://github.com/Xinzhe99/SwinMFF/assets/113503163/03ba666e-d66d-423e-9bc3-27642538aa41)

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
1. Download Weights in [Baidu](https://pan.baidu.com/s/15-5_TzVa-ZypyceiMSyMkg?pwd=cite) and put in the project path
### Lytro
`python .\predict.py --dataset_path='./assets/Lytro' --model_path='./checkpoint.ckpt' --is_gray=False`
### MFFW
`python .\predict.py --dataset_path='./assets/MFFW' --model_path='./checkpoint.ckpt' --is_gray=False`
### MFI-WHU
`python .\predict.py --dataset_path='./assets/MFI-WHU' --model_path='./checkpoint.ckpt' --is_gray=False`
### Others
`python .\predict.py --dataset_path='your path' --model_path='your path' --is_gray=False/True`
# Acknowledgements
The research was supported by the Hainan Provincial Joint Project of Sanya Yazhou Bay Science and Technology City (No: 2021JJLH0079), Innovational Fund for Scientific and Technological Personnel of Hainan Province (NO. KJRC2023D19), and the Hainan Provincial Joint Project of Sanya Yazhou Bay Science and Technology City (No. 2021CXLH0020). Thanks for help by Hainan Provincial Observatory of Ecological Environment and Fishery Resource in Yazhou Bay. Also, we want to thank Chloe Alex Schaff for her contribution in polishing the article.

# If our work is helpful to you, please help cite our work
