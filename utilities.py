import features as ft

import torchio as tio
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_subject(patient_id, category, weight, height, base_path="Dataset/Train"):
    """
    Create a torchio subject from a patient id and a base path
    
    Parameters
    ----------
    patient_id : str
    category : str
    weight : float
    height : float
    base_path : str
    
    Returns
    -------
    subject : tio.Subject
    """
    patient_path = os.path.join(base_path, patient_id)

    subject = tio.Subject(
        id = patient_id,
        ED=tio.ScalarImage(os.path.join(patient_path, f"{patient_id}_ED.nii")),
        ED_seg=tio.LabelMap(os.path.join(patient_path, f"{patient_id}_ED_seg.nii")),
        ES=tio.ScalarImage(os.path.join(patient_path, f"{patient_id}_ES.nii")),
        ES_seg=tio.LabelMap(os.path.join(patient_path, f"{patient_id}_ES_seg.nii")),
        category=category,
        weight=weight,
        height=height
    )
    return subject

def load_dataset(type,transform = True):
    """
    Load the dataset from the csv file and images
    
    Parameters
    ----------
    type : str
        'Train' or 'Test'
    
    Returns
    -------
    subjects : tio.SubjectsDataset
    """
    if type == "Train":
        df = pd.read_csv('Dataset/metaDataTrain.csv')
    elif type == "Test":
        df = pd.read_csv('Dataset/metaDataTest.csv')
    else:
        raise ValueError("type must be 'Train' or 'Test'")
    subjects = []

    for _, row in df.iterrows():
        patient_id = str(int(row["Id"])).zfill(3)
        
        if type == "Train":
            category = row["Category"]
        height = row["Height"]
        weight = row["Weight"]

        try:
            if type == "Train":
                subject = create_subject(patient_id, category, weight, height, "Dataset/Train")
            else:
                subject = create_subject(patient_id, -1, weight, height, "Dataset/Test")
            subjects.append(subject)
        except Exception as e:
            print(f"Error with patient {patient_id}: {e}")
    print(f"Loaded {len(subjects)} subjects")

    if transform:
        landmarks = {
        'ED': "saves/ED_landmarks.npy",
        'ES': "saves/ES_landmarks.npy"
        }
        histo_stand = tio.HistogramStandardization(landmarks)
        transform = tio.Compose([
        histo_stand,
        tio.RescaleIntensity(out_min_max=(0, 1)),
        ])
        return tio.SubjectsDataset(subjects, transform=transform)
    
    return tio.SubjectsDataset(subjects)

def create_landmarks_hist(ED_name = "ED_landmarks", ES_name = "ES_landmarks"):
    """
    Create the landmarks for the histogram standardization
    
    Parameters
    ----------
    ED_name : str
    ES_name : str
    
    Returns
    -------
    None
    """

    ED_paths = [f"Dataset/Train/{str(subject).zfill(3)}/{str(subject).zfill(3)}_ED.nii" for subject in range(1, 101)]
    ES_paths = [f"Dataset/Train/{str(subject).zfill(3)}/{str(subject).zfill(3)}_ES.nii" for subject in range(1, 101)]

    ED_landmarks_path = Path('ED_name.npy')
    ES_landmarks_path = Path('ES_name.npy')

    ED_landmarks = (
        ED_landmarks_path
        if ED_landmarks_path.is_file()
        else tio.HistogramStandardization.train(ED_paths)
    )
    ES_landmarks = (
        ES_landmarks_path
        if ES_landmarks_path.is_file()
        else tio.HistogramStandardization.train(ES_paths)
    )
    np.save(ED_landmarks_path, ED_landmarks)
    np.save(ES_landmarks_path, ES_landmarks)


def create_features_matrix(dataset, category=False):
    """
    Create a matrix of features from a dataset
    
    Parameters
    ----------
    dataset : tio.SubjectsDataset
    category : bool
        If True, the function will return the categories (for training)
        
    Returns
    -------
    X : np.array
    y : np.array
    """
    height = []
    weight = []
    imc = []
    category2 = []
    # add features here (à mieux faire c'est horrible là)
    volume_ED1 = []
    volume_ED2 = []
    volume_ED3 = []
    volume_ED4 = []
    volume_ES1 = []
    volume_ES2 = []
    volume_ES3 = []
    volume_ES4 = []
    
    for cpt, subject in enumerate(dataset):
        height.append(subject['height'])
        weight.append(subject['weight'])
        imc.append(subject['weight'] / (subject['height'] / 100) ** 2)
        if category:
            category2.append(int(subject['category']))
        
        voled = ft.volume_ED(cpt, dataset, test=not category)
        voles = ft.volume_ES(cpt, dataset, test=not category)
        
        volume_ED1.append(voled[0])
        volume_ED2.append(voled[1])
        volume_ED3.append(voled[2])
        volume_ED4.append(voled[3])
        volume_ES1.append(voles[0])
        volume_ES2.append(voles[1])
        volume_ES3.append(voles[2])
        volume_ES4.append(voles[3])
        
    if category:
        return np.array([height, weight, imc, volume_ED1, volume_ED2, volume_ED3, volume_ED4, volume_ES1, volume_ES2, volume_ES3, volume_ES4]).T, np.array(category2)
    return np.array([height, weight, imc, volume_ED1, volume_ED2, volume_ED3, volume_ED4, volume_ES1, volume_ES2, volume_ES3, volume_ES4]).T

def submission(clf, file_name="submission", plot=True):
    """
    Create a submission file for Kaggle from a classifier
    
    Parameters
    ----------
    clf : sklearn classifier
    file_name : str
    plot : bool
    
    Returns
    -------
    None
    """
    dataset_test = load_dataset("Test")
    X = create_features_matrix(dataset_test,category=False)
    ids = [subject['id'] for subject in dataset_test]
    y_pred = clf.predict(X)
    if plot:
        plt.figure(figsize=(6,5))
        plt.hist(y_pred, bins=range(0,6), align='left', rwidth=0.8)
        plt.xticks(range(0,5))
        plt.title("Predictions distribution")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.show()
    df = pd.DataFrame(list(zip(ids, y_pred)), columns=["Id","Category"])
    df.to_csv(file_name+".csv", index=False,sep=",")
    print(f"Submission file {file_name}.csv created")