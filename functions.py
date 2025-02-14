#%% Imports -------------------------------------------------------------------

import numpy as np

# czitools
from czitools import extract_data

# bdtools
from bdtools.norm import norm_gcn, norm_pct

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

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    pass