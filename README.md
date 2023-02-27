# UAED-Abnormal-Emotion-Detection
This repository contains the code for paper "UAED: Unsupervised Abnormal Emotion Detection Network Based on Wearable Mobile Device". The overall structure of the UAED is shown in the figure below.  

<img src="https://github.com/zjiaqi725/UAED-Abnormal-Emotion-Detection/blob/main/image/UAED.png" width="700" >  

## Implementation 
#### 1.Environment  
pytorch == 1.5.1  
torchvision == 0.6.1  
numpy == 1.21.5  
scipy == 1.4.1  
sklearn == 0.0  
#### 2.Dataset  
We evaluate the proposed model on four publicly available datasets: (1)[DREAMER](https://zenodo.org/record/546113/accessrequest) (2)[The Stress Recognition in Automobile Drivers database (DRIVEDB)](https://www.physionet.org/content/drivedb/1.0.0/) (3)[Mahnob-HCI-tagging database (MAHNOB-HCI)](https://mahnob-db.eu/hci-tagging/) (4)[Wearable Stress and Affect Detection (WESAD)](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/). Detailed information about the datasets is summarized in the Table below.
<img src="https://github.com/zjiaqi725/UAED-Abnormal-Emotion-Detection/blob/main/image/normal_samples.png" width="300" >  
#### 3.Pre-processing  
We only do simple data cleaning including removal of incomplete data and normalization to better understand the learning ability of UAED for feature representation.
We use a stacking operation to transform the original 1D time series into 3D samples, and the visualization from DREAMER is presented as follow.

<img src="https://github.com/zjiaqi725/UAED-Abnormal-Emotion-Detection/blob/main/image/normal_samples.png" width="300" >  
<img src="https://github.com/zjiaqi725/UAED-Abnormal-Emotion-Detection/blob/main/image/abnormal_samples.png" width="300" >  
The top two rows are normal samples and the bottom two rows are abnormal samples.  

#### 4.Train and Test the Model  
We write both training and evaluation process in the main.py, execute the following command to see the training and evaluation results.  
`python main.py`

## Performance  
We conduct the experiments under a nested cross-validation leave-one-subject-out (LOSO) procedure. Concretely, we leave the data from one subject at a time as the test set and the rest of the data as the training set, thus avoiding overfitting. Further, we perform a 10-fold cross-validation on the training set as an inner loop. 
UAED achieve better average results than a lot of existing traditional and deep methods. Here, we only show the F1 score performance, while the other results can be found in the original paper.

Methods  | DREAMER  | DRIVEDB | MAHNOB-HCI | WESAD | Average
 ---- | ----- | ------  | ------  | ------  | ------
 ISF  | 0.315 | 0.198 | 0.682 | 0.602 | 0.449
 OCSVM  | 0.128 | 0.012 | 0.556 | 0.658 | 0.339
 AE-LSTM  | 0.598 | 0.517 | 0.620 | 0.416 | 0.538
 DAGMM  | 0.412 | 0.210 | 0.665 | 0.523 | 0.453
 DSVDD  | 0.482 | 0.472 | 0.632 | 0.518| 0.526
 VAE-LSTM  | 0.436 | 0.564 | 0.606 | 0.594 | 0.550
 DOMI  | 0.564 | 0.552 | 0.692 | 0.659 | 0.617
 UAED(Pro.) | 0.546 | 0.612 | 0.726 | 0.789 | 0.668
