#%% Imports -------------------------------------------------------------------

import cv2
import time
import napari
import numpy as np
from skimage import io
from pathlib import Path

# czitools
from czitools import save_tiff

# bdmodel
from bdmodel.predict import predict

#%% Inputs --------------------------------------------------------------------

# Auto parameters
rf = 0.25 # rescaling factor
size = int(512 * rf) # patch size

# Parameters
rS = "all"
extract_overwrite = False
predict_overwrite = False
predict_batch_size = 500
predict_patch_overlap = size // 8

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
model_path = Path(Path.cwd(), "model_nuclei_edt")
czi_paths = list(data_path.rglob("*.czi"))

#%% Function(s) : procedures --------------------------------------------------

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
            
#%% Function(s) : postprocessing ----------------------------------------------
            


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
    
    # -------------------------------------------------------------------------
    
    czi_path = czi_paths[0]
    exp_path = Path(czi_path.parent / czi_path.stem)
    prd_paths = list(exp_path.glob("*_prd.tif"))
    
    C1s, C2s, prds = [], [], []
    for prd_path in prd_paths:
        img_path = Path(exp_path, prd_path.name.replace("_prd", ""))
        img = io.imread(img_path)
        prd = io.imread(prd_path)
        C1s.append(img[0, ...])
        C2s.append(img[0, ...])
        prds.append(prd)
    

    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(prds)    
    
    pass