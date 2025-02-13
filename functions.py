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

# bdtools
from bdtools.norm import norm_gcn, norm_pct

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

#%% Function : open_images() --------------------------------------------------

'''
- legacy function to open images for extracting training images
'''

def open_images(path, rS="all", rC=0, rf=1, norm=True):
    
    # Open data
    metadata, imgs = extract_data(
        path, rS=rS, rT='all', rZ='all', rC=rC, zoom=rf)
    
    # Normalize & format
    if norm:
        imgs = norm_pct(
            norm_gcn(imgs, sample_fraction=0.001), sample_fraction=0.001)
    imgs = np.stack(imgs).squeeze()
    
    return imgs    

#%% Function : process_images() -----------------------------------------------

def process_images(
        czi_path,
        rS="all",
        batch_size=500,
        patch_overlap=16,
        C2_min_mean_int=30000,
        C2_min_mean_edt=20,
        ):
    
    # Nested function(s) ------------------------------------------------------          

    def _process_images(i, C1, C2, prd):
        
        # Parameters
        lmax_dist, lmax_prom = 4, 0.6
        C2_dog_sigma1 = 1 
        C2_dog_sigma2 = 8
        C2_dog_thresh = 1
        C2_min_size = 32
        
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
            "plate"       : czi_path.stem,
            "well"        : scn_well,
            "position"    : scn_pos,
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
                result["C2_areas"].append(area.astype(int))
                result["C2_mean_int"].append(mean_int.astype(int))
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
        C1_count = np.max(C1_lbl)
        C2_count = np.max(C2_lbl)
        if C1_count == 0:
            C1C2_ratio = np.nan
        else:
            C1C2_ratio = (C2_count / C1_count)
        result["C1_count"  ] = C1_count
        result["C2_count"  ] = C2_count
        result["C2C1_ratio"] = C1C2_ratio
                
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
        
        return result

    # Execute -----------------------------------------------------------------

    # Fixed parameters
    rf = 0.25
    model_path = Path(Path.cwd(), "model_nuclei_edt")
    
    # Initialize
    metadata = extract_metadata(czi_path)
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
    
    # Process images
    t0 = time.time()
    print(f"process {czi_path.name}", end=" ", flush=True)
    if rS == "all": rS = np.arange(0, metadata["nS"])
    results = Parallel(n_jobs=-1)(
        delayed(_process_images)(i, C1, C2, prd) 
        for (i, C1, C2, prd) in zip(rS, C1s, C2s, prds)
        )    
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Format & save results
    results = pd.DataFrame(results)
    results = results[[
        "plate", "well", "position", 
        "C1_count", "C2_count", "C2C1_ratio",
        "C2_areas", "C2_mean_int", "C2_mean_edt",
        ]]
    results.to_csv(
        exp_path / (czi_path.stem + "_results.csv"), index=False)

#%% Function : display_images() -----------------------------------------------

def display_images(czi_path):
       
    # Initialize
    exp_path = Path(czi_path.parent / czi_path.stem)
    display_paths = list(exp_path.glob("*_display.tif"))
    
    # Load images
    displays = []
    for display_path in display_paths:
        displays.append(io.imread(display_path))
    C1s = [data[..., 0] for data in displays]
    C2s = [data[..., 1] for data in displays]
    C3s = [data[..., 2] for data in displays]
    
    # Viewer
    viewer = napari.Viewer()
    viewer.add_image(
        np.stack(C1s), name="C1",
        blending="additive", colormap="magenta"
        )
    viewer.add_image(
        np.stack(C2s), name="C2",
        blending="additive", colormap="green"
        )
    viewer.add_image(
        np.stack(C3s), name="display",
        blending="additive", colormap="gray"
        )

#%% Function : cond_convert() -------------------------------------------------

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pass