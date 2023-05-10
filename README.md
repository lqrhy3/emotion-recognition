<div align="center">

# Fast Emotion Recognition Neural Network for IoT Devices
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

<img src="demo/video_streaming.gif" alt="example" height="75%" width="75%" title="example">
</div>

## Description

PyTorch implementation of paper *"Fast Emotion Recognition Neural Network for IoT Devices"* by S. Mikhaylevskiy, V. Chernyavskiy and V. Pavlishen.

This project focused on developing an efficient system for recognizing human emotions from facial expressions, designed to run on edge devices, with the final optimized model weighing just 6MB and achieving 4 FPS on Raspberry Pi 3.
**[[Paper]](https://ieeexplore.ieee.org/abstract/document/9444517)**

## Installation and environment

#### Pip

```bash
# clone project
git clone https://github.com/lqrhy3/emotion-recognition.git
cd emotion-recognition

# [OPTIONAL] create conda environment
conda create -n emotion-recognition
conda activate emotion-recognition

# install requirements
pip install -r requirements.txt

# set PYTHONPATH variable
export PYTHONPATH=$PYHONPATH:/<path>/emotion-recognition
```

## Data
While the origins of the data remain unknown, both the training data and data for quantization are available on this repository. The dataset includes four categories of data: detection training data, detection calibration data, classification training data, and classification calibration data.

* Detection training data: [Link](data/detection/train_images)
* Detection calibration data: [Link](data/detection/calibration_images)
* Classification training data: [Link](data/classification/train_images)
* Classification calibration data: [Link](data/classification/calibration_images)

## Face Detection

```bash
python3 train.py --config <path to config (in 'configs' folder)> --debug <True for sanity checker, default False>
```

## Emotion Classification
  
## Quantization

## Emotion Recognition Pipeline

## Citation
If you find our code, data or paper is helpful, please consider citing:
```
@article{9444517,
  author={Mikhaylevskiy, S. and Chernyavskiy, V. and Pavlishen, V. and Romanova, I. and Solovyev, R.},
  booktitle={2021 International Seminar on Electron Devices Design and Production (SED)}, 
  title={Fast Emotion Recognition Neural Network for IoT Devices}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/SED51197.2021.9444517}
}
```

______________________________________________________________________

## Contributors 
This project was completed by [Stanislav Mikhaylevskiy](https://github.com/lqrhy3), 
[Victor Pavlishen](https://github.com/vspavl99) and [Vladimir Chernyavskiy](https://github.com/JJBT). If you have any questions or suggestions regarding this project, please feel free to contact us.
