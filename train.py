#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path

# bdmodels
from bdmodel.train import Train
from bdmodel.functions import get_paths

#%% Inputs --------------------------------------------------------------------    

# Paths
train_path = Path(Path.cwd(), "data", "train")

# Parameters
ext = ".tif"
msk_name = "_mask"
tags_in = [msk_name]
save_name = "nuclei_edt"
    
#%% Execute -------------------------------------------------------------------
    
if __name__ == "__main__":

    # Open data
    imgs, msks = [], []
    msk_paths = get_paths(train_path, ext=ext, tags_in=tags_in, subfolders=True)
    for path in msk_paths:
        imgs.append(io.imread(str(path).replace(msk_name, "")))
        msks.append(io.imread(path))
            
    # Train
    train = Train(
        imgs, msks,
        save_name=save_name,
        save_path=Path.cwd(),
        msk_type="bounds",
        img_norm="global",
        patch_size=128,
        patch_overlap=0,
        nAugment=500,
        backbone="resnet18",
        epochs=200,
        batch_size=4,
        validation_split=0.2,
        learning_rate=0.0005,
        patience=30,
        weights_path="",
        # weights_path=Path(Path.cwd(), save_name, "weights.h5"),
        )