## LF-SASR: Light Field Super-Resolution for Real-World Scenes via Adaptive Modeling

LF-SASR(Light Field Scenario-Aware Super Resolution) aims to bridge the gap between idealized synthetic datasets and challenging real-world environments, providing practical and efficient solutions for light field super-resolution tasks.

This project is forked from [LF-DMnet](https://github.com/yingqianwang/LF-DMnet) and is partly based on [EPIT](https://github.com/ZhengyuLiang24/EPIT/).

**THIS PROJECT IS UNDER CONSTRUCTION**

## Abstract:

LF-SASR is a research project aimed at advancing light field image super-resolution (SR) techniques tailored for real-world applications. Building upon the foundation of LF-DMNet, this project advances light field image super-resolution for real-world scenarios by introducing dual-case degradation modeling and a hybrid Transformer-CNN architecture.

## Preparation:

#### 1. Requirement:
* PyTorch 2.5.0, torchvision 0.20.0. The code is tested with python=3.12.7, cuda=12.7
* Matlab or Python for training/validation/test data generation.

#### 2. Datasets:
* We used the HCInew, HCIold and STFgantry datasets for training and validation. Please first download the aforementioned datasets via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place these datasets to the folder `../Datasets/`.
* We used the EPFL, INRIA and STFlytro datasets (which are developed by using Lytro cameras) to test the practical value of our method.

#### 3. Generating training/validation data:
* Run `GenerateDataForTraining.m` to generate training data. The generated data will be saved in `../Data/Train_MDSR_5x5/`.
* Please download the validation data via [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EgVU4b1ImNFMuchPObqZjLYBbI7zcfn_3tcM8bpXzphX5g) and place these data to the folder `../Data/Validation_MDSR_5x5/`.

## Train:
* Set the hyper-parameters in `parse_args()` if needed. We have provided our default settings in the released codes.
* Run `train.py` to perform network training.
* Checkpoint will be saved to `./log/`.

## Validation (synthetic degradation):
* Run `validation.py` to perform validation on each dataset.
* The metric scores will be printed on the screen.

## Test on your own LFs:
* Place the input LFs into `./input` (see the attached examples).
* Run `test.py` to perform SR. 
* The super-resolved LF images will be automatically saved to `./output`.

## Acknowledgement

* This project is part of the undergraduate thesis titled "Research on Super-Resolution Techniques for Light Field Images in Real-World Scenes" and is partly supported by the National Natural Science Foundation of China and the China Association for Science and Technology's Young Talent Program.
