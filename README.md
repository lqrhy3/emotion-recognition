<div align="center">

# Fast Emotion Recognition Neural Network for IoT Devices
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

<img src="demo/demo.gif" alt="example" height="75%" width="75%" title="example">
</div>

## Description

PyTorch implementation of paper *"Fast Emotion Recognition Neural Network for IoT Devices"* by S. Mikhaylevskiy, V. Chernyavskiy and V. Pavlishen.

This project focused on developing an efficient system for recognizing human emotions from facial expressions, designed to run on edge devices, with the final optimized models weighing just 6MB and achieving 4 FPS on Raspberry Pi 3. Pipeline consist of two separate models: one for face detection and the other for emotion classification. We could utilize one composite model, but...

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

Training face-detection model:
```bash
python3 training/detection_train.py [-h] [--num_epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LEARNING RATE] 
                           [--grid_size GRID_SIZE] [--num_bboxes NUM_BBOXES] [--val_split VAL_SPLIT] 
                           [--path_to_log PATH_TO_LOG] [--model MODEL] [--path_to_data PATH_TO_DATA]
```

Evaluating face-detection model:
```bash
python3 testing/detection_pipeline.py [-h] [--path_to_model] [--grid_size GRID_SIZE] [--num_bboxes NUM_BBOXES] 
                              [--image_size IMAGE_SIZE] [--detection_threshold DETECTION_THRESHOLD] [--device DEVICE]
```

## Emotion Classification

Training emotion-classification model:
```bash
python3 training/classification_train.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                                [--val_split VAL_SPLIT] [--path_to_log PATH_TO_LOG] [--comment COMMENT]
                                [--model {MiniXception,ConvNet}] [--path_to_data PATH_TO_DATA] [--emotions EMOTIONS_LIST]
```

Evaluating emotion-classification model:
```bash
python3 testing/classification_pipeline.py [-h] [--path_to_model PATH_TO_MODEL] [--image_size IMAGE_SIZE] 
                                   [--emotions EMOTIONS_LIST] [--device {cpu,cuda:0}]
```

## Emotion Recognition Pipeline

Running emotion-recognition pipeline:
```bash
python3 testing/detection&classification_pipeline.py [-h] [--path_to_detection_model PATH_TO_DETECTION_MODEL] 
                                                     [--path_to_classification_model PATH_TO_CLASSIFICATION_MODEL]
                                                     [--grid_size GRID_SIZE] [--num_bboxes NUM_BBOXES] [--detection_size DETECTION_SIZE]
                                                     [--classification_size CLASSIFICATION_SIZE] [--detection_threshold DETECTION_THRESHOLD]
                                                     [--emotions EMOTIONS_LIST] [--device {cpu,cuda:0}]
```

## Quantization
Running post-training quantization:
```bash
# detection model
python3 quantization/detection_post_training_quatization.py [-h] [--path_to_model PATH_TO_MODEL] [--path_to_data PATH_TO_DATA]
                                                            [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE] [--num_batches NUM_BATCHES] 
                                                            [--save_type SAVE_TYPE]
# classification model 
python3 quantization/classification_post_training_quatization.py [-h] [--path_to_model PATH_TO_MODEL] [--path_to_data PATH_TO_DATA]
                                                                 [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE] [--num_batches NUM_BATCHES] 
                                                                 [--emotions EMOTIONS_LIST] [--save_type SAVE_TYPE]
```

## Demo Application

Running Flask app for demonstrating pipeline:

```bash
cd video-streaming
sudo chmod 0777 app.sh
./app.sh
```
You should specify in `config.ini` path to quantized model

Application will be accessible at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

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
