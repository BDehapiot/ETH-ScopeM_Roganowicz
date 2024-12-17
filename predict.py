#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path

# Functions
from functions import open_images

# bdmodel
from bdmodel.predict import predict

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
model_path = Path(Path.cwd(), "model_nuclei_edt")
img_name = "Plate_01-01.czi"

# Parameters
rS = tuple(np.arange(50, 100))  # number of extracted scene per czi file
rf = 0.25 # rescaling factor
size = int(512 * rf)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    img_path = list(data_path.rglob(f"*{img_name}"))[0]
        
    # Open data
    imgs = open_images(str(img_path), rS=rS, rC=0, rf=rf)
    
    # Predict
    prds = predict(
        imgs,
        model_path,
        img_norm="global",
        patch_overlap=size // 2,
        )
    
    # Display
    import napari
    viewer = napari.Viewer()
    viewer.add_image(imgs)
    viewer.add_image(prds)