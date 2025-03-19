#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path

# Functions
from functions import open_images

# czitools
from czitools import extract_metadata

# bdtools
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

# Parameters
nS = 50   # number of extracted scene per czi file
rf = 0.25 # rescaling factor
size = int(512 * rf)
np.random.seed(41)

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Roganowicz\data")
train_path = Path(Path.cwd(), "data", "train")
czi_paths = list(data_path.rglob("*.czi"))

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    t0 = time.time()
    
    for path in czi_paths:

        metadata = extract_metadata(str(path))
        sIdxs = tuple(np.random.randint(0, high=metadata["nS"], size=nS))
        imgs = open_images(path, rS=sIdxs, rC=0, rf=rf)
        for i, img in enumerate(imgs):
            patches = extract_patches(img, size, 0)
            pIdx = np.random.randint(0, high=len(patches))
            patch = patches[pIdx]
            io.imsave(
                Path(train_path, f"{path.stem}_scene-{sIdxs[i]:04d}.tif"),
                patch.astype("float32"), check_contrast=False
                )        
        
    t1 = time.time()
    print(f"runtime : {t1 - t0:.5f}s")    