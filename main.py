#%% Imports -------------------------------------------------------------------

import cv2
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
rS = tuple(np.arange(400, 600))  # number of extracted scene per czi file
# rS = 185
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
    
    from skimage.filters import gaussian
    from skimage.draw import rectangle_perimeter
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed, find_boundaries
    from skimage.morphology import remove_small_objects, binary_dilation
    
    from scipy.ndimage import distance_transform_edt
    
    lmax_dist, lmax_prom = 4, 0.6
    
    # -------------------------------------------------------------------------
    
    t0 = time.time()
    
    # -------------------------------------------------------------------------
    
    # Paths
    img_path = list(data_path.rglob(f"*{img_name}"))[0]
        
    # Open data
    C1s = open_images(str(img_path), rS=rS, rC=0, rf=rf)
    C2s = open_images(str(img_path), rS=rS, rC=1, rf=rf, norm=False)
    if C1s.ndim == 2:
        C1s, C2s = [C1s], [C2s]
    elif C1s.ndim == 3:
        C1s, C2s = list(C1s), list(C2s)
    
    # Predict
    prds = predict(
        np.stack(C1s), model_path, img_norm="global", patch_overlap=size // 2)
    
#%%
    
    displays = []
    C1_msks, C1_lbls, C1_edts = [], [], []
    C2_dogs, C2_msks, C2_lbls, C2_dsps = [], [], [], []
    for C1, C2, prd in zip(C1s, C2s, prds):
        
        # Detect C1 nuclei
        lmax = peak_local_max(
            gaussian(prd, sigma=1), exclude_border=False,
            min_distance=lmax_dist, threshold_abs=lmax_prom,
            )
        lmax_msk = np.zeros_like(C1, dtype=int)
        lmax_msk[(lmax[:, 0], lmax[:, 1])] = True
        lmax_lbl = label(lmax_msk)
        C1_lbl = watershed(-prd, lmax_lbl, mask=prd > 0.1)
        C1_msk = C1_lbl > 0
        C1_edt = distance_transform_edt(np.invert(C1_msk)).astype("float32")
        C1_out = find_boundaries(C1_lbl, mode="outer")
        
        # Append
        C1_msks.append(C1_msk)
        C1_lbls.append(C1_lbl)
        C1_edts.append(C1_edt)
        
        # Detect C2 objects
        gbl1 = gaussian(C2, sigma=1) # parameter
        gbl2 = gaussian(C2, sigma=8) # parameter
        C2_dog = (gbl1 - gbl2) / gbl2
        C2_msk = C2_dog > 1 # parameter
        C2_msk = remove_small_objects(C2_msk, min_size=32) # parameter
        C2_lbl = label(C2_msk)
        
        C2_dsp = np.zeros_like(C2_msk, dtype="uint8")
        for props in regionprops(C2_lbl, intensity_image=C2):
            
            lbl = props.label
            y = int(props.centroid[0])  
            x = int(props.centroid[1])
            mean_int = np.mean(C2[C2_lbl == lbl])
            mean_edt = np.mean(C1_edt[C2_lbl == lbl])
            area = props.area
            
            if mean_int < 30000 or mean_edt > 20: # parameters !!!
                C2_msk[C2_lbl == lbl] = False
        
            else:
                
                # Draw object rectangles
                rr, cc = rectangle_perimeter(
                    (y - 25, x - 25), extent=(50, 50), shape=C2_dsp.shape)
                C2_dsp[rr, cc] = 255
                
                # Draw object texts
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    C2_dsp, f"{mean_int:.2e}", 
                    (x + 30, y - 18), # depend on resolution !!!
                    font, 0.33, 128, 1, cv2.LINE_AA
                    ) 
                cv2.putText(
                    C2_dsp, f"{area:.2e}", 
                    (x + 30, y - 5), # depend on resolution !!!
                    font, 0.33, 128, 1, cv2.LINE_AA
                    ) 
        C2_lbl = label(C2_msk)
        C2_out = binary_dilation(C2_msk) ^ C2_msk
        C2_dsp += (C2_out * 128).astype("uint8") 
        
        # Append
        C2_dogs.append(C2_dog)
        C2_msks.append(C2_msk)
        C2_lbls.append(C2_lbl)
        C2_dsps.append(C2_dsp)
        
        # Display
        display = np.maximum(C2_dsp, (C1_out * 32).astype("uint8") )
                
        # Append
        displays.append(display)       

    # # -------------------------------------------------------------------------

    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(np.stack(C1s), blending="additive")
    # viewer.add_image(np.stack(C2s), blending="additive")
    # viewer.add_labels(np.stack(C1_lbls), blending="additive")
    # # viewer.add_image(np.stack(C1_edts), blending="additive")
    # # viewer.add_image(np.stack(C2_dogs), blending="additive")
    # # viewer.add_labels(np.stack(C2_lbls), blending="additive")
    # # viewer.add_image(np.stack(C2_dsps), blending="additive")
    # viewer.add_image(np.stack(displays), blending="additive")

    t1 = time.time()    
    print(f"execute : {t1 - t0:.5f}")
