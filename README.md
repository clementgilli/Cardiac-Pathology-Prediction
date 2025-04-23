# Cardiac Pathology Prediction

This project tackles the automatic classification of cardiac pathologies from MRI using handcrafted features and machine learning models. It was developed as part of the IMA205 course for this [Kaggle challenge](https://www.kaggle.com/competitions/ima-205-challenge-2025/overview").

## Challenge
The task is to classify each patient into one of five categories:
1. Normal (NOR)
2. Myocardial Infarction (MINF)
3. Dilated Cardiomyopathy (DCM)
4. Hypertrophic Cardiomyopathy (HCM)
5. Abnormal Right Ventricle (ARV)

## How to Run

1. **Install dependencies**
```bash
pip install -r requirements.txt
```
2. **Run the notebook**

This will use the optimal hyperparameters I've found :
```python
TRAIN = False
```
