#%% Imports -------------------------------------------------------------------

import ast
import numpy as np
import pandas as pd
from pathlib import Path

# functions
from functions import process_images, display_images

#%% Inputs --------------------------------------------------------------------

# Procedure
process = False
display = False
display_idx = 0 

# Process parameters
# rS = "all"
# rS = tuple(np.arange(0, 100))
batch_size = 500
patch_overlap = 16
C2_min_mean_int = 30000
C2_min_mean_edt = 20

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
czi_paths = list(data_path.rglob("*.czi"))

#%% Function : analyse() ------------------------------------------------------

def analyse(data_path):
        
    def convert_mapping(mResults):
        
        plate_mapping = {
            'Plate_01-01': 'Control 0.1% DMSO',
            'Plate_02-01': 'Norfloxacin 0.002 µg/ml',
            'Plate_03-01': 'Norfloxacin 0.008 µg/ml',
            'Plate_04-01': 'Norfloxacin 0.032 µg/ml',
            }

        well_mapping = {
            'A01': 'parental', 'A02': 'ΔfimH', 'A03': 'ΔmotA', 
            'A04': 'ΔmotB'   , 'A05': 'ΔfliA', 'A06': 'ΔfliC',
            'B01': 'ΔcsgA'   , 'B02': 'ΔcsgB', 'B03': 'Δkps' ,
            'B04': 'Δneu'    , 'B05': 'Δgspl', 'B06': 'ΔyhjM',    
            'C01': 'Δfiu'    , 'C02': 'ΔnikA', 'C03': 'Δyhjk', 
            'C04': 'ΔsfaA'   , 'C05': 'ΔyeaP', 'C06': 'ΔyagX',
            'D01': 'ΔluxS'   , 'D02': 'ΔcheA', 'D03': 'Δtsr' , 
            'D04': 'ΔrfaQ'   , 'D05': 'ΔyaiW', 'D06': 'ΔompW',
            }

        mResults['plate'] = mResults['plate'].replace(plate_mapping)
        mResults['well' ] = mResults['well' ].replace(well_mapping)   
            
    global mResults
    
    # Initialize
    csv_paths = list(data_path.rglob("*.csv"))
    
    # Load csv
    mResults = []
    for csv_path in csv_paths:
        mResults.append(pd.read_csv(csv_path))
    mResults = pd.concat(mResults, ignore_index=True)
    mResults['C2_areas'] = mResults['C2_areas'].apply(ast.literal_eval)
    mResults['C2_mean_int'] = mResults['C2_mean_int'].apply(ast.literal_eval)
    mResults['C2_mean_edt'] = mResults['C2_mean_edt'].apply(ast.literal_eval)
    
    # Convert mapping
    # convert_mapping(mResults)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    # Process images
    if process:
        for czi_path in czi_paths:    
            process_images(
                czi_path, rS=rS,
                batch_size=batch_size,
                patch_overlap=patch_overlap,
                C2_min_mean_int=C2_min_mean_int,
                C2_min_mean_edt=C2_min_mean_edt,
                )

    # Display image
    if display:
        display_images(czi_paths[display_idx])
        
    # # Analyse
    # analyse(data_path)
    
#%%
    
    # def format_list(df, key):
    #     lst = df[key].explode().tolist()
    #     lst = [e for e in lst if not np.isnan(e)]
    #     return lst
        
    # cResults = {
    #     "plate"           : [],
    #     "well"            : [],
    #     "C1_count"        : [],
    #     "C2_count"        : [],
    #     "C2C1_ratio"      : [],
    #     "C2_areas_avg"    : [],
    #     "C2_mean_int_avg" : [],
    #     "C2_mean_edt_avg" : [],
    #     }

    # conds = mResults[['plate', 'well']].drop_duplicates().reset_index(drop=True)
    # for index, row in conds.iterrows():        
        
    #     df = mResults[
    #         (mResults['plate'] == row['plate']) &
    #         (mResults['well' ] == row['well' ])
    #         ] 
        
    #     C1_count = df["C1_count"].sum()
    #     C2_count = df["C2_count"].sum()
    #     C2C1_count = C2_count / C1_count
    #     C2_areas = format_list(df, 'C2_areas')
    #     C2_mean_int = format_list(df, 'C2_mean_int')
    #     C2_mean_edt = format_list(df, 'C2_mean_edt')
        
    #     cResults['plate'          ].append(row['plate'])
    #     cResults['well'           ].append(row['well' ])
    #     cResults['C1_count'       ].append(C1_count)
    #     cResults['C2_count'       ].append(C2_count)
    #     cResults['C2C1_ratio'     ].append(C2C1_count)
    #     cResults['C2_areas_avg'   ].append(np.mean(C2_areas))
    #     cResults['C2_mean_int_avg'].append(np.mean(C2_mean_int))
    #     cResults['C2_mean_edt_avg'].append(np.mean(C2_mean_edt))

    # cResults = pd.DataFrame(cResults)        
        