from utilities import *

import numpy as np
from skimage.segmentation import flood_fill
from scipy.ndimage import distance_transform_edt
from skimage import measure
from scipy.spatial import cKDTree
from skimage.feature import canny
from skimage.morphology import closing

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

def compute_LVM_slice_thickness(slice_segmentation, voxel_size_mm):
    LVM_mask = (slice_segmentation == 2).astype(np.uint8)

    if np.sum(LVM_mask) == 0:
        return 0
    
    distance_map = distance_transform_edt(LVM_mask) * voxel_size_mm[0]
    distance_map = distance_map[distance_map > 0]
    mean_thickness = np.mean(distance_map) * 2
    return mean_thickness

def compute_LVM_thickness(segmented_img, voxel_size_mm):
    max_thicknesses = [compute_LVM_slice_thickness(segmented_img[:, :, i], voxel_size_mm) for i in range(segmented_img.shape[2])]
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

def compute_circumference_slice(slice_mask, voxel_size_mm):
    contours = measure.find_contours(slice_mask, level=0.5)

    if len(contours) == 0:
        return 0

    longest_contour = max(contours, key=len)

    perimeter = np.sum(np.sqrt(np.sum(np.diff(longest_contour, axis=0) ** 2, axis=1)))

    return perimeter * voxel_size_mm[0]

def compute_circumference(segmented_img, voxel_size_mm):
    circumferences = [
        compute_circumference_slice(segmented_img[:, :, i],voxel_size_mm)
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
    
def size_and_ratio_RVC_apex(segmented_img, voxel_size_mm):
    index = get_apex_LVM_slice(segmented_img)
    slice = segmented_img[:,:,index]
    LVC_mask = slice == 3
    RVC_mask = slice == 1
    size_LVC = np.sum(LVC_mask) * np.prod(voxel_size_mm)
    size_RVC = np.sum(RVC_mask) * np.prod(voxel_size_mm)
    if size_LVC == 0:
        return 0
    return size_RVC, size_RVC/size_LVC

def volume_min_max(id, dataset, voxel_size_mm):
    volumes_ED = volume_ED(id, dataset, voxel_size_mm)
    volumes_ES = volume_ES(id, dataset, voxel_size_mm)
    RVC_volumes= (volumes_ED[0], volumes_ES[0])
    LVM_volumes = (volumes_ED[1], volumes_ES[1])
    LVC_volumes = (volumes_ED[2], volumes_ES[2])
    max_RVC = max(RVC_volumes)
    max_LVM = max(LVM_volumes)
    max_LVC = max(LVC_volumes)
    min_RVC = min(RVC_volumes)
    min_LVC = min(LVC_volumes)
    argmin_LVC = np.argmin(LVC_volumes)
    min_LVM = LVM_volumes[argmin_LVC]
    return max_RVC, max_LVM, max_LVC, min_RVC, min_LVC, min_LVM
    
def compute_EF(volume_ED, volume_ES):
    if volume_ED == 0:
        return 0

    return (volume_ED - volume_ES / volume_ED) * 100.0

def find_min_dists(edges_ext, edges_int):
    min_dists = []
    int_coords = np.argwhere(edges_int)
    ext_coords = np.argwhere(edges_ext)
    for i in range(int_coords.shape[0]):
        min_dist = np.min(np.linalg.norm(ext_coords - int_coords[i], axis=1))
        min_dists.append(min_dist)
    return np.array(min_dists)

def myocardial_wall_thickness(seg_img, voxel_size_mm):
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