#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from pathlib import Path

# Functions
from functions import open_images

# bdmodel
from bdmodel.predict import predict

# Skimage
from skimage.feature import peak_local_max

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
model_path = Path(Path.cwd(), "model_nuclei_edt")
img_name = "Plate_01-01.czi"

# Parameters
rS = tuple(np.arange(100, 150))  # number of extracted scene per czi file
# rS = 600
rf = 0.25 # rescaling factor
size = int(512 * rf)

#%% Function(s) ---------------------------------------------------------------

def get_lmax(img, min_dist=5, min_prom=0.5, return_mask=False):
    lmax = peak_local_max(
        img, min_distance=min_dist, threshold_abs=min_prom, 
        exclude_border=False, 
        )
    if return_mask:
        lmax_msk = np.zeros_like(img)
        lmax_msk[(lmax[:, 0], lmax[:, 1])] = True
        return lmax, lmax_msk
    return lmax

def segment():
    pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    from skimage.measure import label
    from skimage.filters import gaussian
    # from skimage.feature import blob_dog
    from skimage.segmentation import expand_labels
    from skimage.morphology import (
        disk, remove_small_objects, 
        binary_erosion,
        binary_dilation, 
        binary_closing, 
        binary_opening, 
        )
    
    t0 = time.time()
    
    # -------------------------------------------------------------------------
    
    # Paths
    img_path = list(data_path.rglob(f"*{img_name}"))[0]
        
    # Open data
    C1s = open_images(str(img_path), rS=rS, rC=0, rf=rf)
    C2s = open_images(str(img_path), rS=rS, rC=1, rf=rf, norm=False)
    
    # # Predict
    # prds = predict(
    #     C1s, model_path, img_norm="global", patch_overlap=size // 2)
    
    # -------------------------------------------------------------------------
    
    dogs, dog_msks, lmax_msks = [], [], []
    for img in C2s:
        
        gbl1 = gaussian(img, sigma=1) # parameter
        gbl2 = gaussian(img, sigma=8) # parameter
        dog = (gbl1 - gbl2) / gbl2
        dog_msk = dog > 1 # parameter
        dog_msk = remove_small_objects(dog_msk, min_size=32) # parameter
        dog_lbl = label(dog_msk)
        
        for lbl in np.unique(dog_lbl)[1:]:
            print(lbl)
                
        # lmax, lmax_msk = get_lmax(
        #     dog, min_dist=20, min_prom=0.25, return_mask=True)
        # lmax_msk = binary_dilation(lmax_msk, footprint=disk(3))
        # lmax_msks.append(lmax_msk)
        
        dogs.append(dog)
        dog_msks.append(dog_msk)

    # -------------------------------------------------------------------------

    # Display
    viewer = napari.Viewer()
    viewer.add_image(np.stack(C1s), blending="additive", contrast_limits=[0, 2])
    viewer.add_image(np.stack(C2s), blending="additive")
    viewer.add_image(np.stack(dogs), blending="additive")
    viewer.add_image(np.stack(dog_msks), blending="additive")
    # viewer.add_image(np.stack(lmax_msks), blending="additive", colormap="green")

    t1 = time.time()    
    print(f"execute : {t1 - t0:.5f}")

    pass

#%%

    # -------------------------------------------------------------------------
    
    # if prds.ndim == 2:
    #     prds = gaussian(prds, sigma=1)
    # if prds.ndim == 3:
    #     prds = gaussian(prds, sigma=(0, 1, 1))
        
    # lmax_dist, lmax_prom = 4, 0.6
    # lmax = peak_local_max(
    #     prds, exclude_border=False,
    #     min_distance=lmax_dist, threshold_abs=lmax_prom,
    #     )
    # pMask = prds > 0.2
    # pMax = np.zeros_like(prds, dtype=int)
    # pMax[(lmax[:, 0], lmax[:, 1])] = 1
    # pMax = binary_dilation(pMax)
    # nLabels = label(pMax)
    # nLabels = expand_labels(nLabels, distance=6) 
    # nLabels[pMask == 0] = 0
    # nLabels = remove_small_objects(label(nLabels), min_size=32)
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(pMax)
    # viewer.add_image(prds, colormap="magenta", opacity=0.5, visible=False)
    # viewer.add_image(C1_imgs, blending="additive")
    # viewer.add_labels(nLabels)
    
    # -------------------------------------------------------------------------

    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(C2_imgs, blending="additive")
    
    # -------------------------------------------------------------------------
