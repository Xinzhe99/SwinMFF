# SwinMFF
Code for "SwinMFF: Pure Transformer for End-to-End  Multi-focus Image Fusion"
## Train
### Make dataset for training SwinMFF
1. Download [DUTS](https://pan.baidu.com/s/1XCCbFi-uNNXWlig0CNBoIA?pwd=cite)
2. Extract it to the project path
3. Run the following code to get the data set needed for training
    python .\make_dataset.py --mode='TR'
    python .\make_dataset.py --mode='TE'
### Start to train
    python .\train.py
## Test
1. Download Weights in [Baidu](https://pan.baidu.com/s/15-5_TzVa-ZypyceiMSyMkg?pwd=cite)
### Lytro
python .\predict.py --dataset_path='./assets/Lytro' --model_path='./checkpoint.ckpt' --is_gray=False
### MFFW
python .\predict.py --dataset_path='./assets/MFFW' --model_path='./checkpoint.ckpt' --is_gray=False
### MFI-WHU
python .\predict.py --dataset_path='./assets/MFI-WHU' --model_path='./checkpoint.ckpt' --is_gray=False
### Others
python .\predict.py --dataset_path='your path' --model_path='your path' --is_gray=False/True
# Acknowledgements
The research was supported by the Hainan Provincial Joint Project of Sanya Yazhou Bay Science and Technology City (No: 2021JJLH0079), Innovational Fund for Scientific and Technological Personnel of Hainan Province (NO. KJRC2023D19), and the Hainan Provincial Joint Project of Sanya Yazhou Bay Science and Technology City (No. 2021CXLH0020). Thanks for help by Hainan Provincial Observatory of Ecological Environment and Fishery Resource in Yazhou Bay. Also, we want to thank Chloe Alex Schaff for her contribution in polishing the article.

# If our work is helpful to you, please help cite our work
