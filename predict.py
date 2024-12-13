#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path

# bdmodel
from bdmodel.predict import predict

#%% Inputs --------------------------------------------------------------------

# Paths
img_path = Path(Path.cwd(), "data", "240611-12_2 merged_pix(13.771)_00.tif")
model_path = Path(Path.cwd(), "model_mass")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Open data
    img = io.imread(img_path)
    
    # Predict
    prds = predict(        
        img,
        model_path,
        img_norm="global",
        patch_overlap=0,
        )
    
    # Display
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_image(prds)