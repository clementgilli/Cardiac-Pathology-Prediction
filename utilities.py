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
    NUM_FEATURES = 47
    
    X = np.zeros((N, NUM_FEATURES))
    
    if category:
        y = np.zeros(N)

    for cpt, subject in enumerate(tqdm(dataset)):
        
        # =================== INSTANT FEATURES ===================
        
        # ------------------- 1 - Subject data -------------------
        height = subject["height"]
        weight = subject["weight"]
        imc = weight / (height / 100) ** 2
        body_surface = ft.body_surface(height, weight)

        X[cpt, 0:3] = [height, weight, imc]  # Columns 0,1,2

        if category:
            y[cpt] = int(subject["category"])

        # ------------------- 2 - Segmentation Extraction -------------------
        ED_seg = ft.get_ED_segmentation(cpt, dataset)
        ES_seg = ft.get_ES_segmentation(cpt, dataset)

        # ------------------- 3 - ED/ES Volume -------------------
        voled = ft.volume_ED(cpt, dataset)/body_surface 
        voles = ft.volume_ES(cpt, dataset)/body_surface
        X[cpt, 3:6] = voled  # Columns 3,4,5
        X[cpt, 6:9] = voles  # Columns 6,7,8

        # ------------------- 4 - LVM Thicknesses -------------------
        X[cpt, 9:13] = ft.compute_LVM_thickness(ED_seg)  # Max, Min, Mean, Std (col 9-12)
        X[cpt, 13:17] = ft.compute_LVM_thickness(ES_seg)  # Max, Min, Mean, Std (col 13-16)

        # ------------------- 5 - LVM and RVC Circularity -------------------
        X[cpt, 17] = ft.compute_circularity(ED_seg == 2)  # Circularité LVM ED
        X[cpt, 18] = ft.compute_circularity(ES_seg == 2)  # Circularité LVM ES
        X[cpt, 19] = ft.compute_circularity(ED_seg == 1)  # Circularité RVC ED
        X[cpt, 20] = ft.compute_circularity(ES_seg == 1)  # Circularité RVC ES

        # ------------------- 6 - LVM and RVC Circumference -------------------
        X[cpt, 21:23] = ft.compute_circumference(ED_seg == 2)  # Max & Mean Circum ED LVM
        X[cpt, 23:25] = ft.compute_circumference(ES_seg == 2)  # Max & Mean Circum ES LVM
        X[cpt, 25:27] = ft.compute_circumference(ED_seg == 1)  # Max & Mean Circum ED RVC
        X[cpt, 27:29] = ft.compute_circumference(ES_seg == 1)  # Max & Mean Circum ES RVC

        # ------------------- 7 - RVC apex -------------------
        X[cpt, 29:31] = ft.size_and_ratio_RVC_apex(ED_seg) # Size & Ratio ED
        X[cpt, 31:33] = ft.size_and_ratio_RVC_apex(ES_seg) # Size & Ratio ES
        
        # =================== DYNAMIC VOLUME FEATURES ===================
        vmax_RVC, vmax_LVM, vmax_LVC, vmin_RVC, vmin_LVC, vmin_LVM = ft.volume_min_max(cpt, dataset)
        X[cpt, 33:39] = [vmax_RVC, vmax_LVM, vmax_LVC, vmin_RVC, vmin_LVC, vmin_LVM]
        X[cpt, 39] = vmin_LVC/vmin_RVC
        X[cpt, 40] = vmin_LVM/vmin_LVC
        X[cpt, 41] = vmin_RVC/vmin_LVM
        
        X[cpt, 42] = voled[0]/voled[2]
        X[cpt, 43] = voles[1]/voles[2]
        
        X[cpt, 44] = ft.compute_EF(voled[0], voles[0])
        X[cpt, 45] = ft.compute_EF(voled[2], voles[2])
        X[cpt, 46] = ft.compute_EF(voled[1], voles[1])
    
    if save:
        np.save("saves/features.npy", X)
        np.save("saves/categories.npy", y)
        
    if category:
        return X, y
    return X

def submission(clf, file_name="submission", plot=True, pca=None):
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
    if pca is not None:
        X = pca.transform(X)
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