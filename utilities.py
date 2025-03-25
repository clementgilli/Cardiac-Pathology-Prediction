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
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV

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
    
def random_search_hyperparameters(clf, param_dist, X, y, n_splits=5, n_repeats=3):
    """
    Perform a random search for hyperparameters
    
    Parameters
    ----------
    clf : sklearn classifier
    param_dist : dict
    X : np.array
    y : np.array
    n_splits : int
    n_repeats : int
    
    Returns
    -------
    best_estimator : sklearn classifier
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    random_search = RandomizedSearchCV(
        clf, param_distributions=param_dist, 
        n_iter=50, cv=cv, scoring="accuracy", 
        n_jobs=-1, random_state=0, return_train_score=True
    )

    random_search.fit(X, y)

    print("Best params found :", random_search.best_params_)

    best_index = random_search.best_index_
    mean_val_score = random_search.cv_results_['mean_test_score'][best_index]
    std_val_score = random_search.cv_results_['std_test_score'][best_index]

    print(f"Validation Accuracy: {mean_val_score:.3f} ± {std_val_score:.3f}")

    mean_train_score = random_search.cv_results_['mean_train_score'][best_index]
    std_train_score = random_search.cv_results_['std_train_score'][best_index]

    print(f"Train Accuracy: {mean_train_score:.3f} ± {std_train_score:.3f}")
    return random_search.best_estimator_

def create_features_matrix_Khened(dataset, category=False, save=False):
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

def create_features_matrix_Isensee(dataset, category=False, save=False):
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
    NUM_FEATURES = 53
    
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
        voxel_size_mm = subject["ED"].spacing

        X[cpt, 0:3] = [height, weight, imc]  # Columns 0,1,2

        if category:
            y[cpt] = int(subject["category"])
        
        # ------------------- 2 - Segmentation Extraction -------------------
        ED_seg = ft.get_ED_segmentation(cpt, dataset)
        ES_seg = ft.get_ES_segmentation(cpt, dataset)

        # ------------------- 3 - ED/ES Volume -------------------
        voled = ft.volume_ED(cpt, dataset,voxel_size_mm)/body_surface 
        voles = ft.volume_ES(cpt, dataset,voxel_size_mm)/body_surface
        X[cpt, 3:6] = voled  # Columns 3,4,5
        X[cpt, 6:9] = voles  # Columns 6,7,8

        # ------------------- 4 - LVM Thicknesses -------------------
        X[cpt, 9:13] = ft.compute_LVM_thickness(ED_seg,voxel_size_mm)  # Max, Min, Mean, Std (col 9-12)
        X[cpt, 13:17] = ft.compute_LVM_thickness(ES_seg,voxel_size_mm)  # Max, Min, Mean, Std (col 13-16)

        # ------------------- 5 - LVM and RVC Circularity -------------------
        X[cpt, 17] = ft.compute_circularity(ED_seg == 2)  # Circularité LVM ED
        X[cpt, 18] = ft.compute_circularity(ES_seg == 2)  # Circularité LVM ES
        X[cpt, 19] = ft.compute_circularity(ED_seg == 1)  # Circularité RVC ED
        X[cpt, 20] = ft.compute_circularity(ES_seg == 1)  # Circularité RVC ES

        # ------------------- 6 - LVM and RVC Circumference -------------------
        X[cpt, 21:23] = ft.compute_circumference(ED_seg == 2,voxel_size_mm)  # Max & Mean Circum ED LVM
        X[cpt, 23:25] = ft.compute_circumference(ES_seg == 2,voxel_size_mm)  # Max & Mean Circum ES LVM
        X[cpt, 25:27] = ft.compute_circumference(ED_seg == 1,voxel_size_mm)  # Max & Mean Circum ED RVC
        X[cpt, 27:29] = ft.compute_circumference(ES_seg == 1,voxel_size_mm)  # Max & Mean Circum ES RVC

        # ------------------- 7 - RVC apex -------------------
        X[cpt, 29:31] = ft.size_and_ratio_RVC_apex(ED_seg,voxel_size_mm) # Size & Ratio ED
        X[cpt, 31:33] = ft.size_and_ratio_RVC_apex(ES_seg,voxel_size_mm) # Size & Ratio ES
        
        # =================== DYNAMIC VOLUME FEATURES ===================
        vmax_RVC, vmax_LVM, vmax_LVC, vmin_RVC, vmin_LVC, vmin_LVM = ft.volume_min_max(cpt, dataset, voxel_size_mm)
        X[cpt, 33:39] = [vmax_RVC, vmax_LVM, vmax_LVC, vmin_RVC, vmin_LVC, vmin_LVM]
        X[cpt, 39] = vmin_LVC/vmin_RVC
        X[cpt, 40] = vmin_LVM/vmin_LVC
        X[cpt, 41] = vmin_RVC/vmin_LVM
        
        X[cpt, 42] = voled[0]/voled[2]
        X[cpt, 43] = voles[1]/voles[2]
        
        X[cpt, 44] = ft.compute_EF(voled[0], voles[0])
        X[cpt, 45] = ft.compute_EF(voled[2], voles[2])
        X[cpt, 46] = ft.compute_EF(voled[1], voles[1])
        
        X[cpt, 47] = voles[0]/voles[2]
        X[cpt, 48] = voled[1]/voled[2]
        
        X[cpt, 49:51] = ft.compute_LVM_thickness2(ED_seg,voxel_size_mm)
        X[cpt, 51:53] = ft.compute_LVM_thickness2(ES_seg,voxel_size_mm)
        
    
    if save:
        np.save("saves/features_Isensee.npy", X)
        np.save("saves/categories.npy", y)
        
    if category:
        return X, y
    return X



def create_features_matrix_Wolterink(dataset, category=False, save=False):
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
    NUM_FEATURES = 27
    
    X = np.zeros((N, NUM_FEATURES))
    
    if category:
        y = np.zeros(N)

    for cpt, subject in enumerate(tqdm(dataset)):
        
        # =================== INSTANT FEATURES ===================
        ED_seg = ft.get_ED_segmentation(cpt, dataset)
        ES_seg = ft.get_ES_segmentation(cpt, dataset)
        # ------------------- 1 - Subject data -------------------
        height = subject["height"]
        weight = subject["weight"]
        body_surface = ft.body_surface(height, weight)
        voxel_size_mm = subject["ED"].spacing

        X[cpt, 0:2] = [height, weight]  # Columns 0,1,2

        if category:
            y[cpt] = int(subject["category"])
            
        # ------------------- 3 - ED/ES Volume -------------------
        voled = ft.volume_ED(cpt, dataset,voxel_size_mm)/body_surface
        voles = ft.volume_ES(cpt, dataset,voxel_size_mm)/body_surface
        X[cpt, 2:5] = voled  # Columns 3,4,5
        X[cpt, 5:8] = voles  # Columns 6,7,8
        
        X[cpt, 8] = ft.compute_EF(voled[0], voles[0])
        X[cpt, 9] = ft.compute_EF(voled[2], voles[2])
        
        X[cpt, 10] = voled[0]/voled[2]
        X[cpt, 11] = voles[0]/voles[2]
        X[cpt, 12] = voled[1]/voled[2]
        X[cpt, 13] = voles[1]/voles[2]
        
        vmax_RVC, vmax_LVM, vmax_LVC, vmin_RVC, vmin_LVC, vmin_LVM = ft.volume_min_max(cpt, dataset,voxel_size_mm)
        X[cpt, 14] = vmin_LVC/vmin_RVC
        X[cpt, 15] = vmin_LVM/vmin_LVC
        X[cpt, 16] = vmin_RVC/vmin_LVM
        
        X[cpt, 17] = ft.compute_circularity(ED_seg == 2)
        X[cpt, 18] = ft.compute_circularity(ES_seg == 2)
        
        X[cpt, 19:21] = ft.size_and_ratio_RVC_apex(ED_seg,voxel_size_mm) # Size & Ratio ED
        X[cpt, 21:23] = ft.size_and_ratio_RVC_apex(ES_seg,voxel_size_mm) # Size & Ratio ES
        
        maxi,_,mean,_ = ft.compute_LVM_thickness(ED_seg,voxel_size_mm)
        X[cpt, 23] = maxi
        X[cpt, 24] = mean
        maxi,_,mean,_ = ft.compute_LVM_thickness(ES_seg,voxel_size_mm)
        X[cpt, 25] = maxi
        X[cpt, 26] = mean
        
    
    if save:
        np.save("saves/features_Wolterink.npy", X)
        np.save("saves/categories.npy", y)
        
    if category:
        return X, y
    return X