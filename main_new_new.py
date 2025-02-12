#%% Imports -------------------------------------------------------------------

import cv2
import time
import shutil
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# czitools
from czitools import extract_metadata, extract_data

# bdmodel
from bdmodel.predict import predict

# skimage
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.draw import rectangle_perimeter
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import (
    remove_small_objects, binary_dilation, skeletonize)

# scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Fixed parameters
rf = 0.25 # rescaling factor
size = int(512 * rf) # patch size

# Parameters
rS = tuple(np.arange(200, 300))
batch_size = 500
patch_overlap = size // 8
save = "all" # "all", "images" or "csv"

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
model_path = Path(Path.cwd(), "model_nuclei_edt")
czi_paths = list(data_path.rglob("*.czi"))

#%% Function : process_image() ------------------------------------------------

def process_images(
        czi_path,
        rS="all",
        batch_size=500,
        patch_overlap=16,
        ):
    
    # Nested function(s) ------------------------------------------------------          

    def _process_images(i, C1, C2, prd):
        
        # Parameters
        lmax_dist, lmax_prom = 4, 0.6
        C2_dog_sigma1 = 1 
        C2_dog_sigma2 = 8
        C2_dog_thresh = 1
        C2_min_size = 32
        C2_min_mean_int = 30000
        C2_min_mean_edt = 20
        
        # Initialize
        scn_well = metadata["scn_well"][i]
        scn_pos = metadata["scn_pos"][i]
        img_name = f"{czi_path.stem}_{scn_well}-{scn_pos:03d}"
        
        # Detect C1 nuclei
        lmax = peak_local_max(
            gaussian(prd, sigma=1), exclude_border=False,
            min_distance=lmax_dist, threshold_abs=lmax_prom, # parameters
            )
        lmax_msk = np.zeros_like(C1, dtype=int)
        lmax_msk[(lmax[:, 0], lmax[:, 1])] = True
        lmax_lbl = label(lmax_msk)
        C1_lbl = watershed(-prd, lmax_lbl, mask=prd > 0.1)
        C1_msk = C1_lbl > 0
        C1_edt = distance_transform_edt(np.invert(C1_msk))
        C1_out = skeletonize(find_boundaries(C1_lbl, mode="outer"))
        
        # Detect C2 objects
        gbl1 = gaussian(C2, sigma=C2_dog_sigma1) # parameter
        gbl2 = gaussian(C2, sigma=C2_dog_sigma2) # parameter
        C2_dog = (gbl1 - gbl2) / gbl2
        C2_msk = C2_dog > C2_dog_thresh # parameter
        C2_msk = remove_small_objects(C2_msk, min_size=C2_min_size) # parameter
        C2_lbl = label(C2_msk)
        
        # Results
        result = {
            "experiment"  : czi_path.stem,
            "scn_well"    : scn_well,
            "scn_pos"     : scn_pos,
            "C2_areas"    : [],
            "C2_mean_int" : [],
            "C2_mean_edt" : [],
            }
        
        # Make display
        display = np.zeros_like(C2_msk, dtype=int)
        for props in regionprops(C2_lbl, intensity_image=C2):
            
            lbl = props.label
            y = int(props.centroid[0])  
            x = int(props.centroid[1])
            mean_int = np.mean(C2[C2_lbl == lbl])
            mean_edt = np.mean(C1_edt[C2_lbl == lbl])
            area = props.area
            
            if mean_int < C2_min_mean_int or mean_edt > C2_min_mean_edt: # parameters !!!
                C2_msk[C2_lbl == lbl] = False
        
            else:
                
                # Update result #1
                result["C2_areas"].append(area)
                result["C2_mean_int"].append(mean_int)
                result["C2_mean_edt"].append(mean_edt)
                
                # Draw object rectangles
                rr, cc = rectangle_perimeter(
                    (y - 25, x - 25), extent=(50, 50), shape=display.shape)
                display[rr, cc] = 255
                
                # Draw object texts
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(
                    display, f"{mean_int:.2e}", 
                    (x + 30, y - 16), # depend on resolution !!!
                    font, 0.5, 192, 1, cv2.LINE_AA
                    ) 
                cv2.putText(
                    display, f"{area:.0f}", 
                    (x + 30, y), # depend on resolution !!!
                    font, 0.5, 192, 1, cv2.LINE_AA
                    ) 
        
        C2_lbl = label(C2_msk)
        C2_out = binary_dilation(C2_msk) ^ C2_msk
        
        # Update result #2
        result["C1_count"] = np.max(C1_lbl)
        result["C2_count"] = np.max(C2_lbl)
                
        # Merge display
        display += (C2_out * 128)
        display = np.maximum(display, (C1_out * 64))
        merged_display = np.stack((
            (C1 / 257).astype("uint8"), 
            (C2 / 257).astype("uint8"), 
            display.astype("uint8"),
            ), axis=0)
                
        # Save predictions
        io.imsave(
            exp_path / (img_name + "_predictions.tif"),
            prd.astype("float32"), check_contrast=False
            )
        
        # Save labels
        io.imsave(
            exp_path / (img_name + "_C1_labels.tif"),
            C1_lbl.astype("uint16"), check_contrast=False
            )
        io.imsave(
            exp_path / (img_name + "_C2_labels.tif"),
            C2_lbl.astype("uint16"), check_contrast=False
            )
        
        # Save display
        val_range = np.arange(256, dtype='uint8')
        lut_gray = np.stack([val_range, val_range, val_range])
        lut_green = np.zeros((3, 256), dtype='uint8')
        lut_green[1, :] = val_range
        lut_magenta = np.zeros((3, 256), dtype='uint8')
        lut_magenta[[0,2],:] = np.arange(256, dtype='uint8')
        io.imsave(
            exp_path / (img_name + "_display.tif"),
            merged_display,
            check_contrast=False,
            imagej=True,
            metadata={
                'axes': 'CYX', 
                'mode': 'composite',
                'LUTs': [lut_magenta, lut_green, lut_gray],
                },
            photometric='minisblack',
            planarconfig='contig',
            )
        
        return C1_lbl, C2_lbl, display, result

    # Execute -----------------------------------------------------------------
    
    global metadata
    
    # Initialize
    metadata = extract_metadata(czi_path)
    if rS == "all": rS = np.arange(0, metadata["nS"])
    exp_path = Path(czi_path.parent / czi_path.stem)
    if exp_path.exists():
        for item in exp_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        exp_path.mkdir(parents=True, exist_ok=True)

    # Extract images
    t0 = time.time()
    print(f"extract {czi_path.name}", end=" ", flush=True)
    _, C1s = extract_data(czi_path, rS=rS, rC=0, zoom=rf)
    _, C2s = extract_data(czi_path, rS=rS, rC=1, zoom=rf)
    C1s = [C1.squeeze() for C1 in C1s]
    C2s = [C2.squeeze() for C2 in C2s]
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Predict images
    print(f"predict {czi_path.name}")
    prds = []
    if batch_size > len(C1s): batch_size = len(C1s)
    for i in range(0, len(C1s), batch_size):
        prds.append(predict(
            np.stack(C1s[i:i + batch_size]), model_path, 
            img_norm="global", patch_overlap=patch_overlap
            ))
    prds = [p for prd in prds for p in prd]
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # result images
    t0 = time.time()
    print(f"result {czi_path.name}", end=" ", flush=True)
    outputs = Parallel(n_jobs=-1)(
        delayed(_process_images)(i, C1, C2, prd) 
        for (i, C1, C2, prd) in zip(rS, C1s, C2s, prds)
        )
    C1_lbls  = [data[0] for data in outputs]
    C2_lbls  = [data[1] for data in outputs]
    displays = [data[2] for data in outputs]
    results = [data[3] for data in outputs]
    
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")

    return C1s, C2s, prds, C1_lbls, C2_lbls, displays, results

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Process images
    C1s, C2s, prds, C1_lbls, C2_lbls, displays, results = process_images(
        czi_paths[0],
        rS=rS,
        batch_size=batch_size,
        patch_overlap=patch_overlap,
        )
    
    df = pd.DataFrame(results)
        
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(
    #     np.stack(C1s), 
    #     blending="additive", colormap="magenta"
    #     )
    # viewer.add_image(
    #     np.stack(C2s), 
    #     blending="additive", colormap="green"
    #     )
    # viewer.add_image(
    #     np.stack([data[2] for data in outputs]),  
    #     blending="additive", colormap="gray"
    #     )

    