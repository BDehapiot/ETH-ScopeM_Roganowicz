#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# Functions
from functions import open_images

# bdtools
from bdtools.mask import get_edt
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

# Parameters
rS = 200  # number of extracted scene per czi file

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
model_path = Path(Path.cwd(), "model_nuclei_edt")
save_path = Path(Path.cwd(), "report_bis")
img_name = "Plate_01-01.czi"

#%% Function(s) ---------------------------------------------------------------

def extract(rf):
        
    # Initialize
    size = int(512 * rf)
    path = list(data_path.rglob(f"*{img_name}"))[0]
    
    # Open data
    img = open_images(str(path), rS=rS, rC=0, rf=rf)

    # Get patches
    pchs = extract_patches(img, size, 0)
    
    # Ger edt
    edts = []
    for pch in pchs:
        edt = get_edt()

    # Save
    for p, pch in enumerate(pchs):
        name = f"{path.stem}_scene-{rS:04d}_patch-{p:02d}_rf-{rf:.1f}.tif"
        io.imsave(
            Path(save_path, name), pch.astype("float32"), 
            check_contrast=False,
            )   
    
    return pchs

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    pchs0 = extract(1)
    pchs1 = extract(0.5)

    # # Display
    # import napari
    # viewer = napari.Viewer()
    # viewer.add_image(np.stack(pchs0))
