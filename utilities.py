import features as ft
from models import MLP

import torchio as tio
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import copy
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

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

def objective_mlp_opti(trial, X, y):
    """
    Objective function for Optuna optimization of MLPClassifier.
    
    Parameters
    ----------
    trial : optuna.Trial
    X : np.array
        Features matrix
    y : np.array
        Labels (categories)
        
    Returns
    -------
    float
        Mean accuracy score of the MLPClassifier on the cross-validation folds.
    """
    hidden_options = ["100-100", "100-100-100"]
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
    solver = trial.suggest_categorical("solver", ["adam", "lbfgs"])
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    learning_rate = trial.suggest_categorical("learning_rate", ["constant", "adaptive"])
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
    early_stopping = trial.suggest_categorical("early_stopping", [True, False])
    
    hidden_key = trial.suggest_categorical("hidden_layer_sizes", hidden_options)
    hidden_layer_sizes = tuple(map(int, hidden_key.split("-")))

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        solver=solver,
        activation=activation,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        early_stopping=early_stopping,
        max_iter=1000,
        random_state=0
    )

    scores = cross_val_score(
        clf,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        scoring='accuracy'
    )
    return scores.mean()

def objective_expert(trial, X_expert, y_expert):
        """
        Objective function for Optuna optimization of MLPClassifier for the expert model.
        
        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        X_expert : np.array
            Features matrix for the expert model
        y_expert : np.array
            Labels (categories) for the expert model
        
        Returns
        -------
        float
            Mean accuracy score of the MLPClassifier on the cross-validation folds.
        """
        hidden_dim = trial.suggest_int("hidden_dim", 16, 128)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.3, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        val_scores = []

        for train_idx, val_idx in skf.split(X_expert, y_expert):
            X_train, X_val = X_expert[train_idx], X_expert[val_idx]
            y_train, y_val = y_expert[train_idx], y_expert[val_idx]

            model = MLP(
                input_dim=X_expert.shape[1],
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                epochs=800
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            val_scores.append(acc)

        return np.mean(val_scores)

def stage1(X, y, clfs, voting='hard'):
    """
    Create a voting classifier from a list of classifiers and fit it to the data.
    
    Parameters
    ----------
    X : np.array
        Features matrix
    y : np.array
        Labels (categories)
    clfs : list
        List of classifiers
    voting : str
        Voting method ('hard' or 'soft')
        
    Returns
    -------
    eclf : VotingClassifier
        Voting classifier fitted to the data
    """
    eclf = VotingClassifier(estimators=clfs, voting=voting)
    eclf = eclf.fit(X, y)
    return eclf

def stage2(X_test, y_pred, mlp_expert_fitted):
    """
    Create a second stage classifier to refine the predictions of the first stage classifier.
    
    Parameters
    ----------
    X_test : np.array
        Features matrix of the test set
    y_pred : np.array
        Predictions of the first stage classifier
    mlp_expert_fitted : MLPClassifier
        MLP classifier fitted to the data
    Returns
    -------
    y_pred : np.array
        Refined predictions
    """
    round2 = []
    for i,pred in enumerate(y_pred):
        if pred == 1 or pred == 2:
            round2.append(i)
    if len(round2) == 0:
        return y_pred
    Xround2 = X_test[round2][:,16:20]
    
    y_pred2 = mlp_expert_fitted.predict(Xround2)
    y_pred[round2] = y_pred2
    return y_pred

def predictions(X, y, X_test, clfs, mlp_expert_fitted=None, voting='hard'):
    """
    Predict the labels of the test set using a two-stage classifier.
    
    Parameters
    ----------
    X : np.array
        Features matrix of the training set
    y : np.array
        Labels (categories) of the training set
    X_test : np.array
        Features matrix of the test set
    clfs : list
        List of classifiers for the first stage
    mlp_expert_fitted : MLPClassifier
        MLP classifier fitted to the data for the second stage (if None, no second stage)
    voting : str
        Voting method for the first stage classifier ('hard' or 'soft')
    Returns
    -------
    y_pred : np.array
        Predicted labels of the test set
    """
    clf = stage1(X, y, clfs, voting)
    y_pred = clf.predict(X_test)
    if mlp_expert_fitted is None:
        return y_pred
    y_pred = stage2(X_test, y_pred, mlp_expert_fitted)
    return y_pred