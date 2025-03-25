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

def load_dataset(type,test_from_file = False):
    """
    Load the dataset from the csv file and images
    
    Parameters
    ----------
    type : str
        'Train' or 'Test'
        
    test_from_file : bool
        If True, the test dataset test will be loaded from the segmented images (LV segmentation)
        
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
    
def filter_features(X, feature_importances, threshold=0.01):
    """
    Filter the features of a matrix
    
    Parameters
    ----------
    X : np.array
    feature_importances : np.array
    threshold : float
    
    Returns
    -------
    X_filtered : np.array
    removed_features : np.array
    """
    important_indices = np.where(feature_importances >= threshold)[0]
    removed_features = np.where(feature_importances < threshold)[0]
    X_filtered = X[:, important_indices]
    print(f"{len(removed_features)} features deleted on {X.shape[1]}")
    
    return X_filtered, removed_features

def submission(y_pred, file_name="submission", plot=True):
    """
    Create a submission file for Kaggle from a classifier
    
    Parameters
    ----------
    y_pred : np.array
    file_name : str
    plot : bool
    
    Returns
    -------
    None
    """
    #dataset_test = load_dataset("Test", test_from_file=True)
    #X = create_features_matrix(dataset_test,category=False,save=True)
    ids = np.arange(101,151)
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

def create_features_matrix(dataset, category=False, save=False):
    """
    Create a matrix of features from a dataset.
    
    Parameters
    ----------
    dataset : tio.SubjectsDataset
    category : bool
        If True, the function will return the categories (for training)
        
    Returns
    -------
    X : np.array
    y : np.array (if category=True)
    """
    print("Creating features matrix")

    N = len(dataset)
    NUM_FEATURES = 20
    
    X = np.zeros((N, NUM_FEATURES))
    
    if category:
        y = np.zeros(N)

    for cpt, subject in enumerate(tqdm(dataset)):
        
        voxel_size_mm = subject["ED"].spacing
        
        if category:
            y[cpt] = int(subject["category"])        

        vol_RV_ED, vol_MYO_ED, vol_LV_ED = ft.volume_ED(cpt, dataset, voxel_size_mm)
        vol_RV_ES, vol_MYO_ES, vol_LV_ES = ft.volume_ES(cpt, dataset, voxel_size_mm)
        
        X[cpt, 0:2] = vol_LV_ED, vol_RV_ED
        X[cpt, 2:5] = vol_LV_ES, vol_RV_ES, vol_MYO_ES
        X[cpt, 5] = vol_MYO_ED * 1.05
        X[cpt, 6] = ft.compute_EF(vol_LV_ED, vol_LV_ES)
        X[cpt, 7] = ft.compute_EF(vol_RV_ED, vol_RV_ES)
        X[cpt, 8] = vol_LV_ED / vol_RV_ED
        X[cpt, 9] = vol_LV_ES / vol_RV_ES
        X[cpt, 10] = vol_MYO_ES / vol_LV_ES
        X[cpt, 11] = (vol_MYO_ED*1.05) / vol_LV_ED
        X[cpt, 12:16] = ft.myocardial_wall_thickness(ft.get_ED_segmentation(cpt, dataset), voxel_size_mm)
        X[cpt, 16:20] = ft.myocardial_wall_thickness(ft.get_ES_segmentation(cpt, dataset), voxel_size_mm)
    
    if save:
        np.save("saves/features_Khened.npy", X)
    
    if category:
        return X, y
    return X