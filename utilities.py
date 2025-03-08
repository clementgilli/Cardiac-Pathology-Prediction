import features as ft

import torchio as tio
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import copy
from tqdm import tqdm

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

def load_dataset(type,transform = True,test_from_file = False):
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
                if test_from_file:
                    subject = create_subject(patient_id, -1, weight, height, "Dataset/Test/Segmented")
                else:
                    subject = create_subject(patient_id, -1, weight, height, "Dataset/Test")
            subjects.append(subject)
        except Exception as e:
            print(f"Error with patient {patient_id}: {e}")

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
        dataset =  tio.SubjectsDataset(subjects, transform=transform)
    
    else:
        dataset = tio.SubjectsDataset(subjects)
    
    if type == "Test" and not test_from_file:
        for i in range(len(dataset)):
            index = i+101
            new_seg_ed = ft.left_ventricle_cavity_segmentation(i,"ED",dataset)
            new_seg_es = ft.left_ventricle_cavity_segmentation(i,"ES",dataset)
            new_seg_ed = torch.tensor(new_seg_ed).unsqueeze(0)
            new_seg_es = torch.tensor(new_seg_es).unsqueeze(0)
            modify_data(i,"ED_seg",new_seg_ed,dataset)
            modify_data(i,"ES_seg",new_seg_es,dataset)
            new_dir = Path('Dataset/Test/Segmented', str(index).zfill(3))
            new_dir.mkdir(parents=True, exist_ok=True)
            dataset[i]["ED_seg"].save(f"Dataset/Test/Segmented/{str(index).zfill(3)}/{str(index).zfill(3)}_ED_seg.nii")
            dataset[i]["ES_seg"].save(f"Dataset/Test/Segmented/{str(index).zfill(3)}/{str(index).zfill(3)}_ES_seg.nii")
            dataset[i]["ED"].save(f"Dataset/Test/Segmented/{str(index).zfill(3)}/{str(index).zfill(3)}_ED.nii")
            dataset[i]["ES"].save(f"Dataset/Test/Segmented/{str(index).zfill(3)}/{str(index).zfill(3)}_ES.nii")
    print(f"Loaded {len(subjects)} subjects")
    return dataset

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

def modify_data(subject_index, type, new_data, dataset):
    """"
    Modify the data of a attribute of a subject in a dataset
    
    Parameters
    ----------
    subject_index : int
    type : str
        'ED', 'ES', 'ED_seg', 'ES_seg'
    new_data : np.array
    dataset : tio.SubjectsDataset
    
    Returns
    -------
    None
    """
    subject = copy.deepcopy(dataset[subject_index])
    subject[type].set_data(new_data)
    dataset._subjects[subject_index] = subject

def create_features_matrix(dataset, category=False, dataframe=False):
    """
    Create a matrix of features from a dataset
    
    Parameters
    ----------
    dataset : tio.SubjectsDataset
    category : bool
        If True, the function will return the categories (for training)
        
    Returns
    -------
    X : pd.DataFrame
    y : np.array (if category=True)
    """
    print("Creating features matrix")
    
    feature_names = ["height", "weight", "imc"]
    features = {name: [] for name in feature_names}

    if category:
        categories = []
        
    for cpt, subject in enumerate(tqdm(dataset)):
        features["height"].append(subject["height"])
        features["weight"].append(subject["weight"])
        features["imc"].append(subject["weight"] / (subject["height"] / 100) ** 2)

        if category:
            categories.append(int(subject["category"]))

        voled = ft.volume_ED(cpt, dataset)
        voles = ft.volume_ES(cpt, dataset)

        for i in range(len(voled)):
            feature_name_ED = f"volume_ED{i+1}"
            feature_name_ES = f"volume_ES{i+1}"
            
            if feature_name_ED not in features:
                features[feature_name_ED] = []
                features[feature_name_ES] = []
            
            features[feature_name_ED].append(voled[i])
            features[feature_name_ES].append(voles[i])

        #if "new_feature" not in features:
        #    features["new_feature"] = []
        #features["new_feature"].append(ft.new_feature(cpt, dataset))
        
    X = pd.DataFrame(features)
    
    if not dataframe:
        X = X.to_numpy()

    if category:
        return X, np.array(categories)
    return X

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
    dataset_test = load_dataset("Test", test_from_file=True)
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