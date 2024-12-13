#%% Imports -------------------------------------------------------------------

import os
import warnings
import numpy as np
os.environ['NO_ALBUMENTATIONS_UPDATE'] = "1" # Don't know if it works
import albumentations as A
from joblib import Parallel, delayed 

# bdtools
from bdtools.mask import get_edt
from bdtools.norm import norm_gcn, norm_pct
from bdtools.patch import extract_patches

# Skimage
from skimage.segmentation import find_boundaries 

#%% Function: get_paths() -----------------------------------------------------

def get_paths(
        rootpath, 
        ext=".tif", 
        tags_in=[], 
        tags_out=[], 
        subfolders=False, 
        ):
    
    """     
    Retrieve file paths with specific extensions and tag criteria from a 
    directory. The search can include subfolders if specified.
    
    Parameters
    ----------
    rootpath : str or pathlib.Path
        Path to the target directory where files are located.
        
    ext : str, default=".tif"
        File extension to filter files by (e.g., ".tif" or ".jpg").
        
    tags_in : list of str, optional
        List of tags (substrings) that must be present in the file path
        for it to be included.
        
    tags_out : list of str, optional
        List of tags (substrings) that must not be present in the file path
        for it to be included.
        
    subfolders : bool, default=False
        If True, search will include all subdirectories within `rootpath`. 
        If False, search will be limited to the specified `rootpath` 
        directory only.
        
    Returns
    -------  
    selected_paths : list of pathlib.Path
        A list of file paths that match the specified extension and 
        tag criteria.
        
    """
    
    if subfolders:
        paths = list(rootpath.rglob(f"*{ext}"))
    else:
        paths = list(rootpath.glob(f"*{ext}"))
        
    selected_paths = []
    for path in paths:
        if tags_in:
            check_tags_in = all(tag in str(path) for tag in tags_in)
        else:
            check_tags_in = True
        if tags_out:
            check_tags_out = not any(tag in str(path) for tag in tags_out)
        else:
            check_tags_out = True
        if check_tags_in and check_tags_out:
            selected_paths.append(path)

    return selected_paths

#%% Function: preprocess() ----------------------------------------------------
   
def preprocess(
        imgs, msks=None,
        img_norm="global",
        msk_type="normal", 
        patch_size=256, 
        patch_overlap=0,
        ):
    
    """ 
    Preprocess images and masks for training or prediction procedures.
    
    If msks=None, only images will be preprocessed.
    Images and masks will be splitted into patches.
    
    Parameters
    ----------
    imgs : 2D ndarray or list of 2D ndarrays (int or float)
        Input image(s).
        
    msks : 2D ndarray or list of 2D ndarrays (bool or int), optional, default=None 
        Input corresponding mask(s).
        If None, only images will be preprocessed.
        
    img_norm : str, default="global"
        - "global" : 0 to 1 normalization considering the full stack.
        - "image"  : 0 to 1 normalization per image.
        
    msk_type : str, default="normal"
        - "normal" : No changes.
        - "edt"    : Euclidean distance transform of binary/labeled objects.
        - "bounds" : Boundaries of binary/labeled objects.

    patch_size : int, default=256
        Size of extracted patches.
        Should be int > 0 and multiple of 2.
    
    patch_overlap : int, default=0
        Overlap between patches.
        Should be int, from 0 to patch_size - 1.
        
    Returns
    -------  
    imgs : 3D ndarray (float32)
        Preprocessed images.
        
    msks : 3D ndarray (float32), optional
        Preprocessed masks.
        
    """
    
    valid_types = ["normal", "edt", "bounds"]
    if msk_type not in valid_types:
        raise ValueError(
            f"Invalid value for msk_type: '{msk_type}'."
            f" Expected one of {valid_types}."
            )

    valid_norms = ["global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
        
    if patch_size <= 0 or patch_size % 2 != 0:
        raise ValueError(
            f"Invalid value for patch_size: '{patch_size}'."
            f" Should be int > 0 and multiple of 2."
            )

    if patch_overlap < 0 or patch_overlap >= patch_size:
        raise ValueError(
            f"Invalid value for patch_overlap: '{patch_overlap}'."
            f" Should be int, from 0 to patch_size - 1."
            )

    # Nested function(s) ------------------------------------------------------

    def normalize(arr, sample_fraction=0.1):
        arr = norm_gcn(arr, sample_fraction=sample_fraction)
        arr = norm_pct(arr, sample_fraction=sample_fraction)
        return arr      
            
    def _preprocess(img, msk=None):

        if msks is None:
            
            img = np.array(img).squeeze()
            
            img = extract_patches(img, patch_size, patch_overlap)
                 
            return img
            
        else:
            
            img = np.array(img).squeeze()
            msk = np.array(msk).squeeze()
            
            if msk_type == "normal":
                msk = msk > 0
            elif msk_type == "edt":
                msk = get_edt(msk, normalize="object", parallel=False)
            elif msk_type == "bounds":
                msk = find_boundaries(msk)           
            
            img = extract_patches(img, patch_size, patch_overlap)
            msk = extract_patches(msk, patch_size, patch_overlap)
                
            return img, msk
    
    # Execute -----------------------------------------------------------------        
       
    # Normalize images
    if img_norm == "global":
        imgs = normalize(imgs)
    if img_norm == "image":
        if isinstance(imgs, np.ndarray) and imgs.ndim == 2: 
            imgs = normalize(imgs)
        else:
            imgs = [normalize(img) for img in imgs]
    
    # Preprocess
    if msks is None:
        
        if isinstance(imgs, np.ndarray):           
            if imgs.ndim == 2: imgs = [imgs]
            elif imgs.ndim == 3: imgs = list(imgs)
        
        if len(imgs) > 1:
               
            outputs = Parallel(n_jobs=-1)(
                delayed(_preprocess)(img)
                for img in imgs
                )
            imgs = [data for data in outputs]
            imgs = np.stack([arr for sublist in imgs for arr in sublist])
                
        else:
            
            imgs = _preprocess(imgs)
            imgs = np.stack(imgs)
        
        imgs = imgs.astype("float32")
        
        return imgs
    
    else:
        
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 2: imgs = [imgs]
            elif imgs.ndim == 3: imgs = list(imgs)
        if isinstance(msks, np.ndarray):
            if msks.ndim == 2: msks = [msks]
            elif msks.ndim == 3: msks = list(msks)
        
        if len(imgs) > 1:
            
            outputs = Parallel(n_jobs=-1)(
                delayed(_preprocess)(img, msk)
                for img, msk in zip(imgs, msks)
                )
            imgs = [data[0] for data in outputs]
            msks = [data[1] for data in outputs]
            imgs = np.stack([arr for sublist in imgs for arr in sublist])
            msks = np.stack([arr for sublist in msks for arr in sublist])
            
        else:
            
            imgs, msks = _preprocess(imgs, msks)
            imgs = np.stack(imgs)
            msks = np.stack(msks)
            
        imgs = imgs.astype("float32")
        msks = msks.astype("float32")
        
        return imgs, msks
    
#%% Function: augment() -------------------------------------------------------

def augment(imgs, msks, iterations):
      
    """
    Augment images and masks using random transformations.
    
    The following transformation are applied:
        
        - vertical flip (p = 0.5)      
        - horizontal flip (p = 0.5)
        - rotate 90° (p = 0.5)
        - transpose (p = 0.5)
        - distord (p = 0.5)
    
    The same transformation is applied to an image and its correponding mask.
    Transformation can be tuned by modifying the `operations` variable.
    The function is based on the `albumentations` library.
    https://albumentations.ai/

    Parameters
    ----------
    imgs : 3D ndarray (float)
        Input image(s).
        
    msks : 3D ndarray (float) 
        Input corresponding mask(s).
        
    iterations : int
        The number of augmented samples to generate.
    
    Returns
    -------
    imgs : 3D ndarray (float)
        Augmented image(s).
        
    msks : 3D ndarray (float) 
        Augmented corresponding mask(s).
    
    """
    
    if iterations <= imgs.shape[0]:
        warnings.warn(f"iterations ({iterations}) is less than n of images")
        
    # Nested function(s) ------------------------------------------------------
    
    def _augment(imgs, msks, operations):      
        idx = np.random.randint(0, len(imgs) - 1)
        outputs = operations(image=imgs[idx,...], mask=msks[idx,...])
        return outputs["image"], outputs["mask"]
    
    # Execute -----------------------------------------------------------------
    
    operations = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.GridDistortion(p=0.5),
        ])
    
    outputs = Parallel(n_jobs=-1)(
        delayed(_augment)(imgs, msks, operations)
        for i in range(iterations)
        )
    imgs = np.stack([data[0] for data in outputs])
    msks = np.stack([data[1] for data in outputs])
    
    return imgs, msks