# Cardiac Pathology Prediction

This project tackles the automatic classification of cardiac pathologies from MRI using handcrafted features and machine learning models. It was developed as part of the IMA205 course for this [Kaggle challenge](https://www.kaggle.com/competitions/ima-205-challenge-2025).

## Challenge
The task is to classify each patient into one of five categories:
1. Normal (NOR)
2. Myocardial Infarction (MINF)
3. Dilated Cardiomyopathy (DCM)
4. Hypertrophic Cardiomyopathy (HCM)
5. Abnormal Right Ventricle (ARV)

## Files

All functions used in the notebook can be found in [utilities.py](./utilities.py), such as create_features_matrix or predictions. 
Functions used to calculate features can be found in [features.py](./features.py). 
MLP models can be found in [models.py](./models.py).
Finally, all save files can be found in the saves folder, such as the parameters of the various models (which can be seen in the .json files), or the feature matrices already calculated to save time.

## How to Run

1. **Install dependencies**
```bash
pip install -r requirements.txt
```
2. **Run the notebook**

You can fix `TRAIN = False` at the beginning of the [notebook](./worksheet.ipynb) : this  will use the optimal hyperparameters I've found.

You can also load the features matrix that I have already calculated.

⚠️ If you want to create the features matrix yourself, in the case of the test dataset, you must use the option `test_from_file=False` in the `load_dataset` function the first time you run the notebook : this will calculate and save the complete segmentation of the test dataset, and then if you want to reload the notebook, you can use the option `test_from_file=True` to save computing time.

You have access to all the documentation in the files.
