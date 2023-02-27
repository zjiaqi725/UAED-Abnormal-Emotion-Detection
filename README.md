# UAED-Abnormal-Emotion-Detection
The UAED model for Unsupervised Abnormal Emotion Detection
## Implementation 
#### 1.Environment  
pytorch == 1.5.1  
torchvision == 0.6.1  
numpy == 1.21.5  
scipy == 1.4.1  
sklearn == 0.0  

#### 2.Dataset  
Wearable Stress and Affect Detection (WESAD) is a multimodal dataset for stress and emotion detection. A RespiBAN professional sensor worn on the chest was used to acquire ECG signals from 15 participants at a sampling rate of 700 Hz with the aim of studying four different emotional states, namely neutral, stress, amusement, and meditation. In this experiment, we treated the stress state as abnormal and the rest of the affective states as normal.  
The datasets is available from (https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)  
#### 3.Pre-processing
we only do simple data cleaning including removal of incomplete data and normalization to better understand the learning ability of UAED for feature representation.   
We use a stacking operation to transform the original 1D time series into 3D samples, and the visualization is presented as follow.
<img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_pima_task1.jpg" width="300" ><img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_pima_task2.jpg" width="200" >  
<img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_pima_task1.jpg" width="300" ><img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_pima_task2.jpg" width="200" >  
#### 4.Train and Test the Model
We write both training and evaluation function in the main.py, execute the following command to see the training and evaluation results.

python main.py

## Performance
### Comparison of the Receiver Operating Characteristic (ROC) curve
The experiments are conducted under two task scenarios, Task I is a weakly supervised scenario using only normal samples for training (anomaly-free), and Task II is an unsupervised scenario where the training set is randomly mixed with a few anomalies.  
<img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_pima_task1.jpg" width="300" ><img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_pima_task2.jpg" width="300" >  
Task I and Task II on Pima  
<img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_thyroid_task1.jpg" width="300" ><img src="https://github.com/zjiaqi725/Aadae-anomaly-detection/blob/main/results/roccurve_thyroid_task2.jpg" width="300" >  
Task I and Task II on Thyroid 
### Special Thanks：
Our work is inspired by Gong’s work in [Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder (MemAE) for Unsupervised Anomaly Detection](https://donggong1.github.io/anomdec-memae)
