
# C3P: Cross-domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation

## Introduction
This is the official implementation for the paper, **"C3P: Cross-domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation"**, ECCV 2022. 


## C3P_code
***
 CMU is in CMUP_open.   
 ITOP is in ITOP_open.   
 
## Installation
 ***
  Install the corresponding dependencies in the requirement.txt.   
  mkdir 'model;output;data' in both CMU_open and ITOP_open.   
  download preprocessed data https://drive.google.com/file/d/127P1g2SaaovZ_7gyVLh9rd-XXycXnf4y/view?usp=sharing in CMUP_open/data.   
  download preprocessed data https://drive.google.com/file/d/1nEoD8qs-8XpSI7PRmMvE4Hc--H531mxC/view?usp=sharing in ITOP_open/data.   
  
## Test
  &ensp; download model https://drive.google.com/file/d/1XE3M4h5Lf9OxVWSQwKoolDpd9xF4xUch/view?usp=sharing in ITOP_open/model.   
  &ensp; download model https://drive.google.com/file/d/1XE3M4h5Lf9OxVWSQwKoolDpd9xF4xUch/view?usp=sharing in CMUP_open/model.   
  &ensp; download model https://drive.google.com/file/d/1zjuVIOWQ_FSH4_pq0Acm7Hvjmbs8pUY4/view?usp=sharing in CMUP_open/model.
  &ensp; to test ITOP dataset:   
    &ensp;&ensp; cd ITOP_open   
    &ensp;&ensp; python test.py   
  &ensp; to test CMU_Panoptic dataset:   
    &ensp;&ensp; cd CMUP_open   
    &ensp;&ensp; python test.py   
  &ensp;(change model to test need to modify the corresponding code in test.py)   
  
## Train
   &ensp; cd CMUP_open   
   &ensp; python train.py   
   &ensp; the model and log file will be saved in output   
 
 
If you find our work useful in your research or publication, please cite our work:
```
@inproceedings{C3P,
author = {Wu, Cunlin and Xiao, Yang and Zhang, Boshen and Zhang Mingyang and Cao, Zhiguo and Zhou Tianyi, Joey},
title = {C3P: Cross-domain Pose Prior Propagation for Weakly Supervised 3D Human Pose Estimation},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
year = {2022}
}
```
