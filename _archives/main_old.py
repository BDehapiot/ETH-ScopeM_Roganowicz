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
rS = "all"
# rS = tuple(np.arange(0, 3168, 20))
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
        
    def format_list(df, key):
        lst = df[key].explode().tolist()
        lst = [e for e in lst if not np.isnan(e)]
        return lst
    
    def get_stats(data):
        if data:
            avg = np.nanmean(data)
            std = np.nanstd(data)
            sem = std  / np.sqrt(len(data))
        else:
            avg, std, sem = np.nan, np.nan, np.nan
        return avg, std, sem
            
    
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
            
    global mResults, aResults
    
    # Initialize
    csv_paths = list(data_path.rglob("*_results.csv"))
    
    # Load & merge results (mResults)
    mResults = []
    for csv_path in csv_paths:
        mResults.append(pd.read_csv(csv_path))
    mResults = pd.concat(mResults, ignore_index=True)
    mResults['C2_areas'] = mResults['C2_areas'].apply(ast.literal_eval)
    mResults['C2_mean_int'] = mResults['C2_mean_int'].apply(ast.literal_eval)
    mResults['C2_mean_edt'] = mResults['C2_mean_edt'].apply(ast.literal_eval)
    
    # Avg. results (aResults)
    
    aResults = {
        "plate"           : [], "well"            : [],
        "C1_count_cum"    : [], "C2_count_cum"    : [],
        "C2C1_ratio_avg"  : [], "C2C1_ratio_std"  : [], "C2C1_ratio_sem"  : [],
        "C2_areas_avg"    : [], "C2_areas_std"    : [], "C2_areas_sem"    : [],
        "C2_mean_int_avg" : [], "C2_mean_int_std" : [], "C2_mean_int_sem" : [],
        "C2_mean_edt_avg" : [], "C2_mean_edt_std" : [], "C2_mean_edt_sem" : [],
        }

    conds = mResults[['plate', 'well']].drop_duplicates().reset_index(drop=True)
    for index, row in conds.iterrows():        
        
        df = mResults[
            (mResults['plate'] == row['plate']) &
            (mResults['well' ] == row['well' ])
            ] 
        
        C2_areas = format_list(df, 'C2_areas')
        C2_mean_int = format_list(df, 'C2_mean_int')
        C2_mean_edt = format_list(df, 'C2_mean_edt')
        C1_count_cum = np.sum(df["C1_count"])
        C2_count_cum = np.sum(df["C2_count"])
        C1C2_ratio_stats = get_stats(list(df["C2C1_ratio"]))
        C2_areas_stats = get_stats(C2_areas)
        C2_mean_int_stats = get_stats(C2_mean_int)
        C2_mean_edt_stats = get_stats(C2_mean_edt)
            
        aResults['plate'          ].append(row['plate'])
        aResults['well'           ].append(row['well' ])
        aResults['C1_count_cum'   ].append(C1_count_cum)
        aResults['C2_count_cum'   ].append(C2_count_cum)
        aResults['C2C1_ratio_avg' ].append(C1C2_ratio_stats[0])
        aResults['C2C1_ratio_std' ].append(C1C2_ratio_stats[1])
        aResults['C2C1_ratio_sem' ].append(C1C2_ratio_stats[2])
        aResults['C2_areas_avg'   ].append(C2_areas_stats[0])
        aResults['C2_areas_std'   ].append(C2_areas_stats[1])
        aResults['C2_areas_sem'   ].append(C2_areas_stats[2])
        aResults['C2_mean_int_avg'].append(C2_mean_int_stats[0])
        aResults['C2_mean_int_std'].append(C2_mean_int_stats[1])
        aResults['C2_mean_int_sem'].append(C2_mean_int_stats[2])
        aResults['C2_mean_edt_avg'].append(C2_mean_edt_stats[0])
        aResults['C2_mean_edt_std'].append(C2_mean_edt_stats[1])
        aResults['C2_mean_edt_sem'].append(C2_mean_edt_stats[2])

    aResults = pd.DataFrame(aResults)    

    # Convert mapping
    convert_mapping(mResults)
    convert_mapping(aResults)
    
    # Save 
    mResults.to_csv(csv_paths[0].parent.parent / "mResults.csv", index=False)
    aResults.to_csv(csv_paths[0].parent.parent / "aResults.csv", index=False)

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
        
    # Analyse
    analyse(data_path)
    
#%% Plot ----------------------------------------------------------------------

import matplotlib.pyplot as plt

# Parameters
data = "C2C1_ratio"

# Initialize
plates = np.unique(aResults["plate"])
nPlates = len(plates)

fig, axes = plt.subplots(
    nPlates, 1, figsize=(8, 3 * nPlates))

for ax, plate in zip(axes, plates):
    
    # Initialize
    df = aResults[aResults['plate'] == plate]
    wells = df["well"]
    avg = np.array(df[f"{data}_avg"])
    sem = np.array(df[f"{data}_sem"])
    x = np.arange(len(wells))
    avg[np.isnan(avg)] = 0

    # Plot
    ax.bar(
        x, avg, 
        yerr=[np.zeros_like(sem), sem], capsize=5,
        color="skyblue", alpha=0.7, label=data,
        )

    # Formatting
    ax.set_ylim(0, 0.006)
    ax.set_xticks(x)
    ax.set_xticklabels(wells, rotation=90)  # Rotate labels if many
    ax.set_ylabel(data)
    ax.set_title(plate)
    ax.legend(loc="upper right")

# Adjust layout
plt.tight_layout()
plt.show()
