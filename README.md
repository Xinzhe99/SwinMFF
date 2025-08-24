<div align="center">

# 🔬 SwinMFF

**Toward High-Fidelity End-to-End Multi-Focus Image Fusion via Swin Transformer-Based Network**

[![Paper](https://img.shields.io/badge/Paper-The%20Visual%20Computer-blue)](https://link.springer.com/article/10.1007/s00371-024-03637-3)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

*Official implementation of SwinMFF for multi-focus image fusion*

</div>

---

## 🚀 Quick Start

### 📊 Dataset Preparation

> **Step 1:** Download the DUTS dataset

[![DUTS Dataset](https://img.shields.io/badge/Download-DUTS%20Dataset-orange)](https://pan.baidu.com/s/1XCCbFi-uNNXWlig0CNBoIA?pwd=cite)

> **Step 2:** Extract the dataset to your project directory

> **Step 3:** Generate training and testing datasets

```bash
# Generate training dataset
python ./make_dataset.py --mode='TR'

# Generate testing dataset
python ./make_dataset.py --mode='TE'
```

### 🏋️ Training

Start training with the following command:

```bash
python ./train.py
```

> 💡 **Tip:** Make sure you have sufficient GPU memory for training. The model requires at least 8GB VRAM.
## 🧪 Testing & Evaluation

### 📥 Download Pre-trained Weights

[![Pre-trained Weights](https://img.shields.io/badge/Download-Pre--trained%20Weights-success)](https://pan.baidu.com/s/15-5_TzVa-ZypyceiMSyMkg?pwd=cite)

> Download the pre-trained weights and place them in your project directory

### 🎯 Inference on Different Datasets

| Dataset | Command | Description |
|---------|---------|-------------|
| **Lytro** | `python ./predict.py --dataset_path='./assets/Lytro' --model_path='./checkpoint.ckpt' --is_gray=False` | Light field camera dataset |
| **MFFW** | `python ./predict.py --dataset_path='./assets/MFFW' --model_path='./checkpoint.ckpt' --is_gray=False` | Multi-focus fusion dataset |
| **MFI-WHU** | `python ./predict.py --dataset_path='./assets/MFI-WHU' --model_path='./checkpoint.ckpt' --is_gray=False` | Wuhan University dataset |
| **Custom** | `python ./predict.py --dataset_path='your_path' --model_path='your_path' --is_gray=False/True` | Your own dataset |

### 📈 Benchmark Results

> 📊 **Comparison Results:** Download comprehensive comparison results with various learning-based methods

[![Benchmark Results](https://img.shields.io/badge/Download-Benchmark%20Results-purple)](https://pan.baidu.com/s/1aDmgPnbUwElQ-t_4lQtEww?pwd=cite)

> 🔗 **Traditional Methods:** For traditional method comparisons, visit the [MFIF repository](https://github.com/yuliu316316/MFIF)

---

## 📝 Citation

If you find our work helpful in your research, please consider citing our paper:

<details>
<summary>📋 <strong>BibTeX Citation</strong></summary>

```bibtex
@article{xie2024swinmff,
  title={SwinMFF: toward high-fidelity end-to-end multi-focus image fusion via swin transformer-based network},
  author={Xie, Xinzhe and Guo, Buyu and Li, Peiliang and He, Shuangyan and Zhou, Sangjun},
  journal={The Visual Computer},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
```

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

*Made with ❤️ by the SwinMFF Team*

</div>
