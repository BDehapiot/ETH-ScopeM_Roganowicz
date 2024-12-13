#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from pathlib import Path

# czitools
from czitools import extract_metadata, extract_data

#%% Comments : ----------------------------------------------------------------

'''
- add option to open a given scene in extract data
'''

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Roganowicz\data")
czi_paths = list(data_path.rglob("*.czi"))

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    t0 = time.time()
    
    path = str(czi_paths[1])
    metadata, C1 = extract_data(path, rT='all', rZ='all', rC=0, zoom=0.25)
        
    t1 = time.time()
    print(f"runtime : {t1 - t0:.5f}s")
    
    # Display
    import napari
    viewer = napari.Viewer()
    viewer.add_image(np.stack(C1))

    