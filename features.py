from utilities import *

import numpy as np
from skimage.segmentation import flood_fill
from scipy.ndimage import distance_transform_edt
from skimage import measure

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
    dic = dataset[id]["ED_seg"].count_labels()
    dic.pop(0,None)
    return dic

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
    dic = dataset[id]["ES_seg"].count_labels()
    dic.pop(0,None)
    return dic

def body_surface(height, weight):
    return np.sqrt(height * weight / 3600)

def compute_LVM_slice_thickness(slice_segmentation):
    LVM_mask = (slice_segmentation == 2).astype(np.uint8)

    if np.sum(LVM_mask) == 0:
        return 0

    distance_map = distance_transform_edt(LVM_mask)
    distance_map = distance_map[distance_map > 0]
    #max_thickness = np.max(distance_map) * 2
    #min_thickness = np.min(distance_map) * 2
    mean_thickness = np.mean(distance_map) * 2
    #std_thickness = np.std(distance_map) * 2
    return mean_thickness
    #return max_thickness #, min_thickness, mean_thickness, std_thickness

def compute_LVM_thickness(segmented_img):
    max_thicknesses = [compute_LVM_slice_thickness(segmented_img[:, :, i]) for i in range(segmented_img.shape[2])]
    return (
        np.max(max_thicknesses),  
        np.min(max_thicknesses),
        np.mean(max_thicknesses),
        np.std(max_thicknesses)   
    )
    
def compute_circularity_slice(slice_mask):
    
    mask = (slice_mask > 0).astype(np.uint8)

    contours = measure.find_contours(mask, level=0.5)

    if len(contours) == 0:
        return 0  
    
    external_contour = max(contours, key=len)

    perimeter = np.sum(np.sqrt(np.sum(np.diff(external_contour, axis=0) ** 2, axis=1)))

    area = np.sum(mask)

    if perimeter == 0:
        return 0 

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    return circularity

def compute_circularity(segmented_img):
    circularities = [
        compute_circularity_slice(segmented_img[:, :, i])
        for i in range(segmented_img.shape[2])
    ]

    circularities = [c for c in circularities if c > 0]

    if len(circularities) == 0:
        return 0

    return np.mean(circularities)

def compute_circumference_slice(slice_mask):
    contours = measure.find_contours(slice_mask, level=0.5)

    if len(contours) == 0:
        return 0

    longest_contour = max(contours, key=len)

    perimeter = np.sum(np.sqrt(np.sum(np.diff(longest_contour, axis=0) ** 2, axis=1)))

    return perimeter

def compute_circumference(segmented_img):
    circumferences = [
        compute_circumference_slice(segmented_img[:, :, i])
        for i in range(segmented_img.shape[2])
    ]

    circumferences = [c for c in circumferences if c > 0]

    if len(circumferences) == 0:
        return 0, 0

    return np.max(circumferences), np.mean(circumferences)

def get_apex_LVM_slice(segmented_img):
    for i in range(segmented_img.shape[2]-1,0,-1):
        LVM_mask = segmented_img[:,:,i] == 2
        if np.sum(LVM_mask) != 0:
            return i
    print("Error in get_apex_LVM_slice")
    
def size_and_ratio_RVC_apex(segmented_img):
    index = get_apex_LVM_slice(segmented_img)
    slice = segmented_img[:,:,index]
    LVM_mask = slice == 2
    RVC_mask = slice == 1
    size_LVM = np.sum(LVM_mask)
    size_RVC = np.sum(RVC_mask)
    return size_RVC, size_RVC/size_LVM