#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# Functions
from functions import open_images

# czitools
from czitools import extract_metadata

# bdtools
from bdtools.patch import extract_patches
from bdtools.mask import get_edt

# bdmodel
from bdmodel.predict import predict

# skimage
from skimage.measure import label
from skimage.morphology import disk
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.feature import peak_local_max
from skimage.filters.rank import modal, median
from skimage.segmentation import watershed, find_boundaries

# scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Parameters
nS = 50   # number of extracted scene per czi file
rf = 0.5 # rescaling factor
size = int(512 * rf)
np.random.seed(41)
lmax_dist, lmax_prom = 4, 0.6

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
model_path = Path(Path.cwd(), "model_nuclei_edt")
train_path = Path(Path.cwd(), "data", "train")
stock_path = Path(Path.cwd(), "data", "train", "stock")
save_path = Path(Path.cwd(), "report")
czi_paths = list(data_path.rglob("*.czi"))

# Stock
stock_paths = list(stock_path.glob("*.tif"))
stock_names = [path.name for path in stock_paths]
raw_names = [name for name in stock_names if "mask" not in name] 
msk_names = [name for name in stock_names if "mask" in name]                  

#%% Function(s) ---------------------------------------------------------------

def rescale_msk(msk):
    msk = rescale(msk, 2, order=0)
    msk = modal(msk, footprint=disk(2))
    return msk

def rescale_prd(prd):
    prd = rescale(prd, 2, order=0)
    prd = median(prd, footprint=disk(2))
    prd = gaussian(prd, sigma=1)
    return prd

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

def get_prd_mask(raw, prd):
    lmax = peak_local_max(
        gaussian(prd, sigma=1), exclude_border=False,
        min_distance=lmax_dist, threshold_abs=lmax_prom,
        )
    lmax_msk = np.zeros_like(raw, dtype=int)
    lmax_msk[(lmax[:, 0], lmax[:, 1])] = True
    lmax_lbl = label(lmax_msk)
    prd_msk = watershed(-prd, lmax_lbl, mask=prd > 0.1) # Parameter
    return prd_msk

def extract_report():
       
    for path in czi_paths:

        metadata = extract_metadata(str(path))
        sIdxs = tuple(np.random.randint(0, high=metadata["nS"], size=nS))
        imgs = open_images(path, rS=sIdxs, rC=0, rf=rf)
        for i, img in enumerate(imgs):
            patches = extract_patches(img, size, 0)
            pIdx = np.random.randint(0, high=len(patches))
            patch = patches[pIdx]
            
            name = f"{path.stem}_scene-{sIdxs[i]:04d}.tif"
            
            if name in raw_names:
                
                msk_name = name.replace(".tif", "_mask.tif")
                edt_name = name.replace(".tif", "_mask_edt.tif")
                edn_name = name.replace(".tif", "_mask_edn.tif")
                
                msk = io.imread(stock_path / msk_name)
                msk = rescale_msk(msk)
                edt = get_edt(msk)
                edn = get_edt(msk, normalize="object")

                if np.max(msk) > 0:
                    io.imsave(
                        Path(save_path, name), patch.astype("float32"), 
                        check_contrast=False,
                        )  
                    io.imsave(
                        Path(save_path, msk_name), msk, 
                        check_contrast=False,
                        ) 
                    io.imsave(
                        Path(save_path, edt_name), edt, 
                        check_contrast=False,
                        )  
                    io.imsave(
                        Path(save_path, edn_name), edn, 
                        check_contrast=False,
                        ) 

def predict_report():
    
    global raws, prds, prd_msks
        
    raws = [io.imread(stock_path / name) for name in raw_names]
    
    prds = predict(
        np.stack(raws),
        model_path,
        img_norm="global",
        patch_overlap=0,
        )
    
    prd_msks = [get_prd_mask(raw, prd) for raw, prd in zip(raws, prds)]

    save_names = [path.name for path in list(save_path.glob("*.tif"))]

    for i, name in enumerate(raw_names):
        
        if name in save_names:
            
            print(name)
        
            prd = rescale_prd(prds[i])
            prd_msk = rescale_msk(prd_msks[i])
            
            io.imsave(
                save_path / name.replace(".tif", "_pred.tif"),
                prd.astype("float32"), check_contrast=False,
                )   
            io.imsave(
                save_path / name.replace(".tif", "_pred_mask.tif"),
                prd_msk.astype("uint8"), check_contrast=False,
                )  
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    extract_report()
    predict_report()
        
#%%
   
    # msk = io.imread(save_path / "Plate_01-01_scene-0243_mask.tif")
    # edt = get_edt(msk)
    # edt_norm = get_edt(msk, normalize="object")
    
    # raw = raws[1]
    # prd = prds[1]
    # prd_msk = get_prd_mask(raw, prd)
    # prd_msk = rescale_msk(prd_msk)
    # prd_msk = prd_msk.astype("uint8")
        
    # Display
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(edt)
    # viewer.add_image(edt_norm)
    # viewer.add_image(prd)
    # viewer.add_image(prd_msk)
     
