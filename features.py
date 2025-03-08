from utilities import *

import numpy as np
from skimage.segmentation import flood_fill

def get_ES(index, dataset):
    """
    Get the end-systolic image of a subject
    
    Parameters
    ----------
    index : int
    dataset : tio.SubjectsDataset
    
    Returns
    -------
    np.array
        Values between 0 and 255
    """
    return dataset[index]["ES"].data.numpy().squeeze(0)

def get_ED(index, dataset):
    """
    Get the end-diastolic image of a subject
    
    Parameters
    ----------
    index : int
    dataset : tio.SubjectsDataset
    
    Returns
    -------
    np.array 
        Values between 0 and 255
    """
    return dataset[index]["ED"].data.numpy().squeeze(0)

def get_ES_segmentation(index, dataset):
    """
    Get the end-systolic segmentation of a subject
    
    Parameters
    ----------
    index : int
    dataset : tio.SubjectsDataset
    
    Returns
    -------
    np.array 
        Values between 0 and 4:
        0 - Background
        1 - Right ventricle cavity
        2 - Myocardium
        3 - Left ventricle cavity
    """
    return dataset[index]["ES_seg"].data.numpy().squeeze(0)

def get_ED_segmentation(index, dataset):
    """
    Get the end-diastolic segmentation of a subject
    
    Parameters
    ----------
    index : int
    dataset : tio.SubjectsDataset
    
    Returns
    -------
    np.array 
        Values between 0 and 4:
        0 - Background
        1 - Right ventricle cavity
        2 - Myocardium
        3 - Left ventricle cavity
    """
    return dataset[index]["ED_seg"].data.numpy().squeeze(0)

def left_ventricle_cavity_segmentation(id, time, dataset):
    """
    Get the segmentation of the left ventricle cavity (for test set)
    
    Parameters
    ----------
    id : int
    time : str
        "ED" or "ES"
    dataset : tio.SubjectsDataset
    
    Returns
    -------
    np.array
        Values between 0 and 4:
        0 - Background
        1 - Right ventricle cavity
        2 - Myocardium
        3 - Left ventricle cavity
    """
    if time == "ED":
        seg = get_ED_segmentation(id, dataset)
    elif time == "ES":
        seg = get_ES_segmentation(id, dataset)
    
    for slices in range(seg.shape[2]):
        slice_seg = seg[:, :, slices].copy()

        if np.any(slice_seg != 0):  
            slice_seg = flood_fill(slice_seg, (0, 0), new_value=4)

            slice_seg[slice_seg == 0] = 3
            slice_seg[slice_seg == 4] = 0

            seg[:, :, slices] = slice_seg

    return seg

def volume_ED(id, dataset):
    """
    Get the volume of each label in the end-diastolic segmentation
    
    Parameters
    ----------
    id : int
    dataset : tio.SubjectsDataset
    
    Returns
    -------
    dict
        The volume of each label
    """
        
    return dataset[id]["ED_seg"].count_labels()

def volume_ES(id, dataset):
    """
    Get the volume of each label in the end-systolic segmentation
    
    Parameters
    ----------
    id : int
    dataset : tio.SubjectsDataset
        
    Returns
    -------
    dict
        The volume of each label
    """
        
    return dataset[id]["ES_seg"].count_labels()