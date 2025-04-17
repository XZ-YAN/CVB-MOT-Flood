# CVB-MOT-Flood
## Introduction
Intelligent flood scene understanding using computer vision-based multi-object tracking. To run the code, please download [output](https://www.alipan.com/s/UCtG2HghFdF) folder and put it into the root directory.    

## Optical Flow Tracking
The optical flow tracking algorithm is in MOT_KLT_mask/KLT_Mask.py. We developed this algorithm as an independent function that can be integrated into any tracking-by-detection model.

## Occlusion Handling Network
The structure of the Occlusion handling netowrk is in reid/model.py.  
To train this model, please run:  
  `$ python reid/train.py`  

## Multi-object Tracking
To test the trackers on videos, please run:  
  `$ python tracker.py`  
  `$ python tracker_StrongSORT.py`  
The visualization results are in the test/ directory.

## License
* The CMOT dataset is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/) to promote the open use of the dataset and future improvements.
* Without permission, the CMOT dataset should only be used for non-commercial scientific research purposes.  

## Citing the CMOT Dataset
To be updated.
