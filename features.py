from utilities import *

from skimage.segmentation import flood_fill
from skimage import measure, morphology

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
    
    if time == "ED":
        seg = get_ED_segmentation(id, dataset)
    elif time == "ES":
        seg = get_ES_segmentation(id, dataset)
    
    binary = seg == 2
    
    for slices in range(binary.shape[2]):
        
        label_image = measure.label(binary)
        regions = measure.regionprops(label_image)

        start_point = None
        for region in regions:
            start_point = region.centroid
            break
        
        start_point = (int(start_point[0]), int(start_point[1]))
        if seg[start_point[0], start_point[1], slices] == 2:
            break
        seg[:,:,slices] = flood_fill(seg[:,:,slices], start_point, new_value=3)
        
    return seg

def volume_ED(id, dataset, test=False):
    if test:
        seg = left_ventricle_cavity_segmentation(id, "ED" ,dataset)
        dic = {0:0,1:0, 2:0, 3:0}
        for i in range(4):
            dic[i] = (seg == i).sum()
        return dic
        
    return dataset[id]["ED_seg"].count_labels()

def volume_ES(id, dataset, test=False):
    if test:
        seg = left_ventricle_cavity_segmentation(id, "ES" ,dataset)
        dic = {0:0,1:0, 2:0, 3:0}
        for i in range(4):
            dic[i] = (seg == i).sum()
        return dic
        
    return dataset[id]["ES_seg"].count_labels()