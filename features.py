from utilities import *

import numpy as np
from skimage.segmentation import flood_fill
from skimage.feature import canny

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

def volume_ED(id, dataset, voxel_size_mm):
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
    dic = dataset[id]["ED_seg"].count_labels()
    dic.pop(0,None)
    return np.array(list(dic.values()),dtype=np.float32) * np.prod(voxel_size_mm)

def volume_ES(id, dataset, voxel_size_mm):
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
    dic = dataset[id]["ES_seg"].count_labels()
    dic.pop(0,None)
    return np.array(list(dic.values()),dtype=np.float32) * np.prod(voxel_size_mm)

def body_surface(height, weight):
    """
    Compute the body surface of a subject (Mosteller formula)
    
    Parameters
    ----------
    height : float
        In cm
    weight : float
        In kg
        
    Returns
    -------
    float
        In m^2
    """
    return np.sqrt(height * weight / 3600)
    
def compute_EF(volume_ED, volume_ES):
    """
    Compute the ejection fraction of a subject
    
    Parameters
    ----------
    volume_ED : float
        Volume of the left ventricle at end-diastole
    volume_ES : float
        Volume of the left ventricle at end-systole
        
    Returns
    -------
    float
        In percentage
    """
    if volume_ED == 0:
        return 0

    return (volume_ED - volume_ES / volume_ED) * 100.0

def find_min_dists(edges_ext, edges_int):
    """
    Find the minimum distance between the edges of the myocardium
    
    Parameters
    ----------
    edges_ext : np.array
        External edges of the myocardium
    edges_int : np.array
        Internal edges of the myocardium
        
    Returns
    -------
    np.array
        The minimum distances
    """
    min_dists = []
    int_coords = np.argwhere(edges_int)
    ext_coords = np.argwhere(edges_ext)
    for i in range(int_coords.shape[0]):
        min_dist = np.min(np.linalg.norm(ext_coords - int_coords[i], axis=1))
        min_dists.append(min_dist)
    return np.array(min_dists)

def myocardial_wall_thickness(seg_img, voxel_size_mm):
    """
    Compute the myocardial wall thickness of a subject
    
    Parameters
    ----------
    seg_img : np.array
        Segmentation of the myocardium
    voxel_size_mm : tuple
        Voxel size in mm
        
    Returns
    -------
    float
        Maximum mean distance between the edges of the myocardium
    float
        Standard deviation of the mean distances
    float
        Mean standard deviation of the distances
    float
        Standard deviation of the standard deviations
    """
    means, stds = [], []
    for i in range(seg_img.shape[2]):
        slic = seg_img[:,:,i]
        ext = (slic==2) + (slic==3)
        inte = slic==3
        if np.sum(ext) == 0 or np.sum(inte) == 0:
            continue
        edges_ext = canny(ext, sigma=1)
        edges_int = canny(inte, sigma=1)
        dists = find_min_dists(edges_ext, edges_int)
        dists = dists * voxel_size_mm[0]
        means.append(np.mean(dists))
        stds.append(np.std(dists))
    means = np.array(means)
    stds = np.array(stds)
    return np.max(means), np.std(means), np.mean(stds), np.std(stds)