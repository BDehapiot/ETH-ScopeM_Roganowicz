#%% Imports -------------------------------------------------------------------

import cv2
import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import concurrent.futures
from joblib import Parallel, delayed 

# czitools
from czitools import save_tiff

# bdmodel
from bdmodel.predict import predict

# skimage
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.draw import rectangle_perimeter
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import remove_small_objects, binary_dilation

# scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Auto parameters
rf = 0.25 # rescaling factor
size = int(512 * rf) # patch size

# Parameters
rS = "all"
extract_overwrite = False
predict_overwrite = False
process_overwrite = False

predict_batch_size = 500
predict_patch_overlap = size // 8

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
model_path = Path(Path.cwd(), "model_nuclei_edt")
czi_paths = list(data_path.rglob("*.czi"))

#%% Function : extract_images() -----------------------------------------------

def extract_images(czi_paths, rS="all", rf=1, overwrite=False):
    
    # Paths
    for czi_path in czi_paths:
        exp_path = Path(czi_path.parent / czi_path.stem)
        
        # Extract & save images
        if overwrite or not exp_path.exists():
            t0 = time.time()
            print(f"extracting {czi_path.name}", end=" ", flush=True)
            save_tiff(czi_path, rS=rS, zoom=rf)
            t1 = time.time()
            print(f"({t1 - t0:.3f}s)")
            
#%% Function : predict_images() -----------------------------------------------
            
def predict_images(
        czi_paths, model_path, 
        img_norm="global", patch_overlap=16, overwrite=False, batch_size=50,
        ):
    
    # Paths
    for czi_path in czi_paths:
        exp_path = Path(czi_path.parent / czi_path.stem)
        img_paths = list(exp_path.glob("*.tif"))
        
        # Extract images
        C1s, save_paths = [], []
        for img_path in img_paths:
            if "prd" not in img_path.name:
                save_path = Path(exp_path, (img_path.stem + "_prd.tif"))
                if overwrite or not save_path.exists():
                    C1s.append(io.imread(img_path)[0, ...])
                    save_paths.append(save_path)
        
        # Predict
        if C1s:
            t0 = time.time()
            print(f"predict {czi_path.name}")
            prds = []
            for i in range(0, len(C1s), batch_size):
                prds.append(predict(
                    np.stack(C1s[i:i + batch_size]), model_path, 
                    img_norm=img_norm, patch_overlap=patch_overlap
                    ))
            prds = [p for prd in prds for p in prd]
            t1 = time.time()
            print(f"({t1 - t0:.3f}s)")
            
            # Save
            for prd, save_path in zip(prds, save_paths): 
                io.imsave(
                    save_path, prd.astype("float32"), check_contrast=False)

#%% Function : process_images() -----------------------------------------------

def process_images(czi_paths, overwrite=False):
    
    # Nested function(s) ------------------------------------------------------
    
    def load_images(prd_path):
        img_path = exp_path / prd_path.name.replace("_prd", "")
        img = io.imread(img_path)
        prd = io.imread(prd_path)
        return img[0, ...], img[1, ...], prd
    
    def quantify_images(C1, C2, prd):
        
        # Parameters
        lmax_dist, lmax_prom = 4, 0.6
        C2_dog_sigma1 = 1 
        C2_dog_sigma2 = 8
        C2_dog_thresh = 1
        C2_min_size = 32
        C2_min_mean_int = 30000
        C2_min_mean_edt = 20
        
        # Detect C1 nuclei
        lmax = peak_local_max(
            gaussian(prd, sigma=1), exclude_border=False,
            min_distance=lmax_dist, threshold_abs=lmax_prom, # parameters
            )
        lmax_msk = np.zeros_like(C1, dtype=int)
        lmax_msk[(lmax[:, 0], lmax[:, 1])] = True
        lmax_lbl = label(lmax_msk)
        C1_lbl = watershed(-prd, lmax_lbl, mask=prd > 0.1).astype("uint16")
        C1_msk = C1_lbl > 0
        C1_edt = distance_transform_edt(np.invert(C1_msk)).astype("float32")
        C1_out = find_boundaries(C1_lbl, mode="outer")
        
        # Detect C2 objects
        gbl1 = gaussian(C2, sigma=C2_dog_sigma1) # parameter
        gbl2 = gaussian(C2, sigma=C2_dog_sigma2) # parameter
        C2_dog = (gbl1 - gbl2) / gbl2
        C2_msk = C2_dog > C2_dog_thresh # parameter
        C2_msk = remove_small_objects(C2_msk, min_size=C2_min_size) # parameter
        C2_lbl = label(C2_msk)
        
        # Make display
        C2_dsp = np.zeros_like(C2_msk, dtype="uint8")
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
                
        C2_lbl = label(C2_msk).astype("uint16")
        C2_out = binary_dilation(C2_msk) ^ C2_msk
        C2_dsp += (C2_out * 128).astype("uint8") 
        display = np.maximum(C2_dsp, (C1_out * 32).astype("uint8") )
    
        return C1_lbl, C2_lbl, display
    
    # Execute -----------------------------------------------------------------
    
    # Paths
    # for czi_path in czi_paths:
    czi_path = czi_paths[0]
    exp_path = Path(czi_path.parent / czi_path.stem)
    prd_paths = list(exp_path.glob("*_prd.tif"))
    
    # Open images
    t0 = time.time()
    print(f"Loading {czi_path.name}", end=" ", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(load_images, prd_paths))
    C1s, C2s, prds = zip(*results)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Quantify images
    t0 = time.time()
    print(f"Quantifying {czi_path.name}", end=" ", flush=True)
    outputs = Parallel(n_jobs=-1)(
        delayed(quantify_images)(C1, C2, prd) 
        for C1, C2, prd in zip(C1s, C2s, prds)
        )
    C1_lbls = [data[0] for data in outputs]
    C2_lbls = [data[1] for data in outputs]
    displays = [data[2] for data in outputs]
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Saving images
    t0 = time.time()
    print(f"Saving {czi_path.name}", end=" ", flush=True)
    for i, (C1_lbl, C2_lbl, display) in enumerate(zip(C1_lbls, C2_lbls, displays)):
        C1_lbl_path = exp_path / prd_paths[i].name.replace("_prd", "_C1lbl")
        C2_lbl_path = exp_path / prd_paths[i].name.replace("_prd", "_C2lbl")
        display_path = exp_path / prd_paths[i].name.replace("_prd", "_display")
        io.imsave(C1_lbl_path, C1_lbl, check_contrast=False)
        io.imsave(C2_lbl_path, C2_lbl, check_contrast=False)
        io.imsave(display_path, display, check_contrast=False)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
        
    return outputs

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Extract images
    extract_images(czi_paths, rS=rS, rf=rf, overwrite=extract_overwrite)      
    
    # Predict images
    predict_images(
        czi_paths, model_path, 
        img_norm="global", 
        patch_overlap=size // 8, 
        overwrite=predict_overwrite,
        batch_size=predict_batch_size,
        )
    
    # Process images
    # outputs = process_images(czi_paths, overwrite=process_overwrite)   

#%% ---------------------------------------------------------------------------
 
def display_images(czi_path):
    
    # Nested function(s) ------------------------------------------------------
    
    def load_images(prd_path):
        img_path = exp_path / prd_path.name.replace("_prd", "")
        img = io.imread(img_path)
        prd = io.imread(prd_path)
        return img[0, ...], img[1, ...], prd
    
    # Execute -----------------------------------------------------------------
    
    
    

# # Display
# viewer = napari.Viewer()
# viewer.add_image(np.stack(C1), blending="additive")
# viewer.add_image(np.stack(C2), blending="additive")
# viewer.add_labels(np.stack(C1_lbl), blending="additive")
# # viewer.add_image(np.stack(C1_edts), blending="additive")
# # viewer.add_image(np.stack(C2_dogs), blending="additive")
# # viewer.add_labels(np.stack(C2_lbls), blending="additive")
# # viewer.add_image(np.stack(C2_dsps), blending="additive")
# viewer.add_image(np.stack(display), blending="additive")
    