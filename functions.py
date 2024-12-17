#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path

# czitools
from czitools import extract_data

# bdtools
from bdtools.norm import norm_gcn, norm_pct

#%% Function(s) ---------------------------------------------------------------

def open_images(path, rS="all", rC=0, rf=1, norm=True):
    
    # Open
    metadata, imgs = extract_data(
        str(path), rS=rS, rT='all', rZ='all', rC=rC, zoom=rf)
    
    # Normalize & format
    if norm:
        imgs = norm_pct(norm_gcn(imgs))
    imgs = np.stack(imgs).squeeze()
    
    return imgs

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Paths
    data_path = Path("D:\local_Roganowicz\data")
    czi_paths = list(data_path.rglob("*.czi"))
    
    # Parameters
    rf = 0.25
    rS, rC = (0, 10, 50, 100, 500, 1000), 0
    
    # Open image(s)
    imgs = open_images(czi_paths[0], rS=rS, rC=rC, rf=rf)