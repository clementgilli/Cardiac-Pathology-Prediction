import torchio as tio
import pandas as pd
import os
import numpy as np

def create_subject(patient_id, category, weight, height, base_path="Dataset/Train"):
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
    height = []
    weight = []
    imc = []
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

def classifier_to_submission(clf, dataset, file_name="submission"):
    X = create_features_matrix(dataset,category=False)
    ids = [subject['id'] for subject in dataset]
    y_pred = clf.predict(X)
    df = pd.DataFrame(list(zip(ids, y_pred)), columns=["Id","Category"])
    df.to_csv(file_name+".csv", index=False,sep=",")
    
def submission(clf, file_name = "submission"):
    dataset_test = load_dataset("Test")
    classifier_to_submission(clf, dataset_test, file_name)