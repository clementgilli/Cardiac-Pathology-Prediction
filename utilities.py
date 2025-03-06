import torchio as tio
import pandas as pd
import os
import numpy as np

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

def load_dataset(type):
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
        patient_id = f"{int(row['Id']):03d}"
        if type == "Train":
            category = row["Category"]
        height = row["Height"]
        weight = row["Weight"]

        try:
            if type == "Train":
                subject = create_subject(patient_id, category, weight, height, "Dataset/Train")
            else:
                subject = create_subject(patient_id, None, weight, height, "Dataset/Test")
            subjects.append(subject)
        except Exception as e:
            print(f"Erreur avec le patient {patient_id}: {e}")
    print(f"Loaded {len(subjects)} subjects")
    return tio.SubjectsDataset(subjects)

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
    #add here features
    category2 = []
    for subject in dataset:
        height.append(subject['height'])
        weight.append(subject['weight'])
        imc.append(subject['weight'] / (subject['height'] / 100) ** 2)
        if category:
            category2.append(int(subject['category']))
    if category:
        return np.array([height, weight, imc]).T, np.array(category2)
    return np.array([height, weight, imc]).T

def submission(clf, file_name="submission"):
    """
    Create a submission file for Kaggle from a classifier
    
    Parameters
    ----------
    clf : sklearn classifier
    file_name : str
    
    Returns
    -------
    None
    """
    dataset_test = load_dataset("Test")
    X = create_features_matrix(dataset_test,category=False)
    ids = [subject['id'] for subject in dataset_test]
    y_pred = clf.predict(X)
    df = pd.DataFrame(list(zip(ids, y_pred)), columns=["Id","Category"])
    df.to_csv(file_name+".csv", index=False,sep=",")
    print(f"Submission file {file_name}.csv created")