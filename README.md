
# C3P: Cross-domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation

## Introduction
This is the official implementation for the paper, **"C3P: Cross-domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation"**, ECCV 2022. 


## C3P_code
***
 `CMUP_open` is the source for CMU dataset and `ITOP_open` is for ITOP dataset.   
 
## Installation
 ***
  Install the corresponding dependencies in the `requirement.txt`:
  
  ```python
    pip install requirement.txt
 ```   
  
  mkdir `model`, `output`, `data` for both `CMU_open` and `ITOP_open`.   
  Download preprocessed data https://drive.google.com/file/d/127P1g2SaaovZ_7gyVLh9rd-XXycXnf4y/view?usp=sharing to CMUP_open/data.   
  Download preprocessed data https://drive.google.com/file/d/1nEoD8qs-8XpSI7PRmMvE4Hc--H531mxC/view?usp=sharing to ITOP_open/data.   
  
## Test
  &ensp; Download model https://drive.google.com/file/d/1XE3M4h5Lf9OxVWSQwKoolDpd9xF4xUch/view?usp=sharing to ITOP_open/model.   
  &ensp; Download model https://drive.google.com/file/d/1XE3M4h5Lf9OxVWSQwKoolDpd9xF4xUch/view?usp=sharing to CMUP_open/model.   
  &ensp; Download model https://drive.google.com/file/d/1zjuVIOWQ_FSH4_pq0Acm7Hvjmbs8pUY4/view?usp=sharing to CMUP_open/model.   
  &ensp; **Test ITOP dataset**:   
```python
cd ITOP_open
python test.py   
```    
  &ensp; **Test CMU_Panoptic dataset**:   
```python
cd CMUP_open
python test.py   
```    
  &ensp;(you can test different models via modifing the corresponding code in `test.py`)   
  
## Train
```python
cd CMUP_open
python train.py   
``` 
   &ensp; the model and log file will be saved in output folder. 
 
 
If you find our work useful in your research or publication, please cite our work:
```
@inproceedings{C3P,
author = {Wu, Cunlin and Xiao, Yang and Zhang, Boshen and Zhang Mingyang and Cao, Zhiguo and Zhou Tianyi, Joey},
title = {C3P: Cross-domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
year = {2022}
}
```
