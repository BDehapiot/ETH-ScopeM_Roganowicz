#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from numba import njit
import segmentation_models as sm

# Functions
from bdmodel.functions import preprocess

# scipy
from scipy.ndimage import distance_transform_edt

#%% Function : merge_patches() ------------------------------------------------

@njit
def merge_patches_2d_numba(patches, patch_edt, arr, edt, y0s, x0s, size):
    count = 0
    ny0 = y0s.shape[0]
    nx0 = x0s.shape[0]
    for i0 in range(ny0):
        y0 = y0s[i0]
        for j0 in range(nx0):
            x0 = x0s[j0]
            for i in range(size):
                for j in range(size):
                    y_idx = y0 + i
                    x_idx = x0 + j
                    if patch_edt[i, j] > edt[y_idx, x_idx]:
                        edt[y_idx, x_idx] = patch_edt[i, j]
                        arr[y_idx, x_idx] = patches[count, i, j]
            count += 1

def merge_patches(patches, shape, overlap):
    
    """ 
    Reassemble a 2D or 3D ndarray from extract_patches().
    
    The shape of the original array and the overlap between patches used with
    extract_patches() must be provided to instruct the reassembly process. 
    When merging patches with overlap, priority is given to the central regions
    of the overlapping patches.
    
    Parameters
    ----------
    patches : list of ndarrays
        List containing extracted patches.
        
    shape : tuple of int
        Shape of the original ndarray.
        
    overlap : int
        Overlap between patches (Must be between 0 and size - 1).
                
    Returns
    -------
    arr : 2D or 3D ndarray
        Reassembled array.
    
    """
    
    def get_patch_edt(patch_shape):
        edt_temp = np.ones(patch_shape, dtype=float)
        edt_temp[:, 0] = 0
        edt_temp[:, -1] = 0
        edt_temp[0, :] = 0
        edt_temp[-1, :] = 0
        return distance_transform_edt(edt_temp) + 1

    # Get size & dimensions 
    size = patches[0].shape[0]
    if len(shape) == 2:
        nT = 1
        nY, nX = shape
    elif len(shape) == 3:
        nT, nY, nX = shape
    else:
        raise ValueError("shape must be 2D or 3D")
    nPatch = len(patches) // nT

    # Get patch edt
    patch_edt = get_patch_edt(patches[0].shape)
    
    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    # Initialize arrays
    y0s_arr = np.array(y0s, dtype=np.int64)
    x0s_arr = np.array(x0s, dtype=np.int64)

    # Merge patches (2D)
    if len(shape) == 2:
        out_shape = (nY + yPad, nX + xPad)
        arr_out = np.zeros(out_shape, dtype=patches[0].dtype)
        edt_out = np.zeros(out_shape, dtype=patch_edt.dtype)
        patches_array = np.stack(patches)
        merge_patches_2d_numba(patches_array, patch_edt, arr_out, edt_out,
                               y0s_arr, x0s_arr, size)
        
        return arr_out[yPad1:yPad1 + nY, xPad1:xPad1 + nX]

    # Merge patches (3D)
    elif len(shape) == 3:
        patches_array = np.stack(patches).reshape(nT, nPatch, size, size)
        merged_slices = []
        for t in range(nT):
            out_shape = (nY + yPad, nX + xPad)
            arr_out = np.zeros(out_shape, dtype=patches_array.dtype)
            edt_out = np.zeros(out_shape, dtype=patch_edt.dtype)
            merge_patches_2d_numba(patches_array[t], patch_edt, arr_out, edt_out,
                                   y0s_arr, x0s_arr, size)
            merged_slice = arr_out[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
            merged_slices.append(merged_slice)
        
        return np.stack(merged_slices)

#%% Function : predict() ------------------------------------------------------

def predict(
        imgs, 
        model_path, 
        img_norm="global",
        patch_overlap=0,
        ):

    valid_norms = ["global", "image"]
    if img_norm not in valid_norms:
        raise ValueError(
            f"Invalid value for img_norm: '{img_norm}'."
            f" Expected one of {valid_norms}."
            )
        
    # Nested function(s) ------------------------------------------------------
        
    # Execute -----------------------------------------------------------------
    
    # Load report
    with open(str(model_path / "report.pkl"), "rb") as f:
        report = pickle.load(f)
    
    # Load model
    model = sm.Unet(
        report["backbone"], 
        input_shape=(None, None, 1), 
        classes=1, 
        activation="sigmoid", 
        encoder_weights=None,
        )
    
    # Load weights
    model.load_weights(model_path / "weights.h5") 
       
    # Preprocess
    patches = preprocess(
        imgs, msks=None, 
        img_norm=img_norm,
        patch_size=report["patch_size"], 
        patch_overlap=patch_overlap,
        )

    # Predict
    prds = model.predict(patches).squeeze(axis=-1)
    prds = merge_patches(prds, imgs.shape, patch_overlap)
    
    return prds