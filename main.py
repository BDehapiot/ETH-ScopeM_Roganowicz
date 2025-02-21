#%% Imports -------------------------------------------------------------------

import cv2
import ast
import time
import shutil
import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import concurrent.futures
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# czitools
from czitools import extract_metadata, extract_data

# bdtools
from bdtools.norm import norm_gcn, norm_pct

# bdmodel
from bdmodel.predict import predict

# skimage
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.draw import rectangle_perimeter
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import (
    remove_small_objects, binary_dilation, skeletonize)

# scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Procedure
preprocess = False
process = False
analyse = False
plot = False
display = False
display_idx = 1

# Process parameters
rS = "all"
# rS = tuple(np.arange(0, 3168, 10))
batch_size = 500
patch_overlap = 16
C2_min_area = 32
C2_min_mean_int = 15000
C2_min_mean_edt = 20

#%% Initialize ----------------------------------------------------------------

# Paths
data_path = Path("D:\local_Roganowicz\data")
czi_paths = list(data_path.rglob("*.czi"))

#%% Function : preprocess_images() --------------------------------------------

def preprocess_images(
        czi_path,
        rS="all",
        batch_size=500,
        patch_overlap=16,
        ):
    
    # Fixed parameters
    rf = 0.25
    model_path = Path(Path.cwd(), "model_nuclei_edt")
    
    # Initialize
    metadata = extract_metadata(czi_path)
    exp_path = Path(czi_path.parent / czi_path.stem)
    if exp_path.exists():
        for item in exp_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        exp_path.mkdir(parents=True, exist_ok=True)
    
    # Extract images
    t0 = time.time()
    print(f"extract {czi_path.name}", end=" ", flush=True)
    _, C1s = extract_data(czi_path, rS=rS, rC=0, zoom=rf)
    _, C2s = extract_data(czi_path, rS=rS, rC=1, zoom=rf)
    C1s = [C1.squeeze() for C1 in C1s]
    C2s = [C2.squeeze() for C2 in C2s]
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Predict images
    t0 = time.time()
    print(f"predict {czi_path.name}")
    prds = []
    if batch_size > len(C1s): batch_size = len(C1s)
    for i in range(0, len(C1s), batch_size):
        prds.append(predict(
            np.stack(C1s[i:i + batch_size]), model_path, 
            img_norm="global", patch_overlap=patch_overlap
            ))
    prds = [p for prd in prds for p in prd]
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Save images
    t0 = time.time()
    print(f"save {czi_path.name}", end=" ", flush=True)
    if rS == "all": rS = np.arange(0, metadata["nS"])
    for i, C1, C2, prd in zip(rS, C1s, C2s, prds):
        well = metadata["scn_well"][i]
        position = metadata["scn_pos"][i]
        img_name = f"{czi_path.stem}_{i:04d}_{well}-{position}"
        io.imsave(
            exp_path / (img_name + "_C1.tif"),
            C1.astype("uint16"), check_contrast=False)
        io.imsave(
            exp_path / (img_name + "_C2.tif"),
            C2.astype("uint16"), check_contrast=False)
        io.imsave(
            exp_path / (img_name + "_predictions.tif"), 
            prd, check_contrast=False)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")   

#%% Function : process_images() -----------------------------------------------

def process_images(
        czi_path,
        C2_min_area=32,
        C2_min_mean_int=30000,
        C2_min_mean_edt=20,
        ):
    
    # Nested function(s) ------------------------------------------------------          

    def load_images(img_path):
        C1 =  io.imread(str(img_path) + "_C1.tif")
        C2 =  io.imread(str(img_path) + "_C2.tif")
        prd = io.imread(str(img_path) + "_predictions.tif")
        return C1, C2, prd

    def _process_images(i, C1, C2, prd):
        
        # Parameters
        lmax_dist, lmax_prom = 4, 0.6
        C2_dog_sigma1 = 1 
        C2_dog_sigma2 = 8
        C2_dog_thresh = 1

        # Initialize
        well = metadata["scn_well"][i]
        position = metadata["scn_pos"][i]
        img_name = f"{czi_path.stem}_{i:04d}_{well}-{position}"
        
        # Detect C1 nuclei
        lmax = peak_local_max(
            gaussian(prd, sigma=1), exclude_border=False,
            min_distance=lmax_dist, threshold_abs=lmax_prom, # parameters
            )
        lmax_msk = np.zeros_like(C1, dtype=int)
        lmax_msk[(lmax[:, 0], lmax[:, 1])] = True
        lmax_lbl = label(lmax_msk)
        C1_lbl = watershed(-prd, lmax_lbl, mask=prd > 0.1)
        C1_msk = C1_lbl > 0
        C1_edt = distance_transform_edt(np.invert(C1_msk))
        C1_out = skeletonize(find_boundaries(C1_lbl, mode="outer"))
        
        # Detect C2 objects
        gbl1 = gaussian(C2, sigma=C2_dog_sigma1) # parameter
        gbl2 = gaussian(C2, sigma=C2_dog_sigma2) # parameter
        C2_dog = (gbl1 - gbl2) / gbl2
        C2_msk = C2_dog > C2_dog_thresh # parameter
        C2_msk = remove_small_objects(C2_msk, min_size=C2_min_area // 2) # parameter
        C2_lbl = label(C2_msk)
        C2_out = binary_dilation(C2_msk) ^ C2_msk
        
        # ///
        C2_msk_valid = C2_msk.copy()        
        C2_lbl_valid = C2_lbl.copy() 
        # ///
        
        # Results
        result = {
            "plate"       : czi_path.stem,
            "well"        : well,
            "position"    : position,
            "C2_areas"    : [],
            "C2_mean_int" : [],
            "C2_mean_edt" : [],
            }
        
        # Make display
        display = np.zeros_like(C2_msk, dtype=int)
        for props in regionprops(C2_lbl, intensity_image=C2):
            
            lbl = props.label
            y = int(props.centroid[0])  
            x = int(props.centroid[1])
            mean_int = np.mean(C2[C2_lbl == lbl])
            mean_edt = np.mean(C1_edt[C2_lbl == lbl])
            area = props.area

            if (area < C2_min_area or
                mean_int < C2_min_mean_int or 
                mean_edt > C2_min_mean_edt ): # parameters !!!                
                
                isvalid = False
                C2_msk_valid[C2_lbl_valid == lbl] = False
        
            else:
                
                isvalid = True
                
                # Update result #1
                result["C2_areas"].append(area.astype(int))
                result["C2_mean_int"].append(mean_int.astype(int))
                result["C2_mean_edt"].append(mean_edt)
                
                # Draw object rectangles (only valid objects)
                rr, cc = rectangle_perimeter(
                    (y - 25, x - 25), extent=(50, 50), shape=display.shape)
                display[rr, cc] = 255
                
            # Draw object texts
            text_int = 192 if isvalid else 96
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                display, f"{mean_int:.2e}", 
                (x + 15, y - 16), # depend on resolution !!!
                font, 0.5, text_int, 1, cv2.LINE_AA
                ) 
            cv2.putText(
                display, f"{area:.0f}", 
                (x + 15, y), # depend on resolution !!!
                font, 0.5, text_int, 1, cv2.LINE_AA
                ) 
        
        C2_lbl_valid = label(C2_msk_valid)

        # Update result #2
        C1_count = np.max(C1_lbl)
        C2_count = np.max(C2_lbl_valid)
        if C1_count == 0:
            C1C2_ratio = np.nan
        else:
            C1C2_ratio = (C2_count / C1_count)
        result["C1_count"  ] = C1_count
        result["C2_count"  ] = C2_count
        result["C2C1_ratio"] = C1C2_ratio
                
        # Merge display
        display += (C2_out * 128)
        display = np.maximum(display, (C1_out * 64))
        merged_display = np.stack((
            (C1 / 257).astype("uint8"), 
            (C2 / 257).astype("uint8"), 
            display.astype("uint8"),
            ), axis=0)
                
        # Save predictions
        io.imsave(
            exp_path / (img_name + "_predictions.tif"),
            prd.astype("float32"), check_contrast=False
            )
        
        # Save labels
        io.imsave(
            exp_path / (img_name + "_C1_labels.tif"),
            C1_lbl.astype("uint16"), check_contrast=False
            )
        io.imsave(
            exp_path / (img_name + "_C2_labels.tif"),
            C2_lbl_valid.astype("uint16"), check_contrast=False
            )
        
        # Save display
        val_range = np.arange(256, dtype='uint8')
        lut_gray = np.stack([val_range, val_range, val_range])
        lut_green = np.zeros((3, 256), dtype='uint8')
        lut_green[1, :] = val_range
        lut_magenta = np.zeros((3, 256), dtype='uint8')
        lut_magenta[[0,2],:] = np.arange(256, dtype='uint8')
        io.imsave(
            exp_path / (img_name + "_display.tif"),
            merged_display,
            check_contrast=False,
            imagej=True,
            metadata={
                'axes': 'CYX', 
                'mode': 'composite',
                'LUTs': [lut_magenta, lut_green, lut_gray],
                },
            photometric='minisblack',
            planarconfig='contig',
            )
        
        return result

    # Execute -----------------------------------------------------------------
    
    # Initialize
    metadata = extract_metadata(czi_path)
    exp_path = Path(czi_path.parent / czi_path.stem)
    img_paths = [
        Path(str(path).replace("_C1.tif", "")) 
        for path in exp_path.glob("*_C1.tif")
        ]
    img_names = [path.stem for path in img_paths]
    rS = [int(name.split("_")[2]) for name in img_names]

    # Load images
    t0 = time.time()
    print(f"load {czi_path.name}", end=" ", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        imports = list(executor.map(load_images, img_paths))
    C1s, C2s, prds = zip(*imports)
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Process images
    t0 = time.time()
    print(f"process {czi_path.name}", end=" ", flush=True)
    results = Parallel(n_jobs=-1)(
        delayed(_process_images)(i, C1, C2, prd) 
        for (i, C1, C2, prd) in zip(rS, C1s, C2s, prds)
        )    
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Format & save results
    results = pd.DataFrame(results)
    results = results[[
        "plate", "well", "position", 
        "C1_count", "C2_count", "C2C1_ratio",
        "C2_areas", "C2_mean_int", "C2_mean_edt",
        ]]
    results.to_csv(
        exp_path / (czi_path.stem + "_results.csv"), index=False)

#%% Function : analyse_results() ----------------------------------------------

def analyse_results(data_path):
        
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

#%% Function : plot_results() -------------------------------------------------

def plot_results(aResults):
    
    # Initialize
    data = ["C2C1_ratio", "C2_areas", "C2_mean_int"]
    plates = np.unique(aResults["plate"])
    nPlates = len(plates)

    for dat in data:

        fig, axes = plt.subplots(nPlates, 1, figsize=(8, 3 * nPlates))
        
        for ax, plate in zip(axes, plates):
            
            # Initialize
            df = aResults[aResults['plate'] == plate]
            wells = df["well"]
            avg = np.array(df[f"{dat}_avg"])
            sem = np.array(df[f"{dat}_sem"])
            x = np.arange(len(wells))
            avg[np.isnan(avg)] = 0
        
            # Plot
            ax.bar(
                x, avg, 
                yerr=sem, capsize=5,
                color="lightgray", alpha=1, label=dat,
                )
        
            # Formatting
            if dat == "C2C1_ratio"  : ax.set_ylim(0, 0.02)
            if dat == "C2_areas"    : ax.set_ylim(0, 300)
            if dat == "C2_mean_int" : ax.set_ylim(10000, 60000)
            ax.set_xticks(x)
            ax.set_xticklabels(wells, rotation=90)
            ax.set_ylabel(dat)
            ax.set_title(plate)
            ax.legend(loc="upper right")
        
        # Save
        plt.tight_layout()
        plt.savefig(czi_paths[0].parent / f"histograms_{dat}.png", format="png")
        plt.show()

#%% Function : display_images() -----------------------------------------------

def display_images(czi_path):
       
    def load_images(display_path):
        return io.imread(display_path)
    
    # Initialize
    exp_path = Path(czi_path.parent / czi_path.stem)
    display_paths = list(exp_path.glob("*_display.tif"))
    
    # Load images
    t0 = time.time()
    print(f"load displays {czi_path.name}", end=" ", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        displays = list(executor.map(load_images, display_paths))
    C1s = [data[..., 0] for data in displays]
    C2s = [data[..., 1] for data in displays]
    C3s = [data[..., 2] for data in displays]
    t1 = time.time()
    print(f"({t1 - t0:.3f}s)")
    
    # Viewer
    viewer = napari.Viewer()
    viewer.add_image(
        np.stack(C1s), name="C1",
        blending="additive", colormap="magenta"
        )
    viewer.add_image(
        np.stack(C2s), name="C2",
        blending="additive", colormap="green"
        )
    viewer.add_image(
        np.stack(C3s), name="display",
        blending="additive", colormap="gray"
        )

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for czi_path in czi_paths:
        if preprocess:
            preprocess_images(
                czi_path, rS=rS,
                batch_size=batch_size,
                patch_overlap=patch_overlap,
                )
        if process:
            process_images(
                czi_path,
                C2_min_area=C2_min_area,
                C2_min_mean_int=C2_min_mean_int,
                C2_min_mean_edt=C2_min_mean_edt,
                )
            
    if analyse:
        analyse_results(data_path)
        
    if plot:
        aResults = pd.read_csv(list(data_path.rglob("*aResults.csv"))[0])
        plot_results(aResults)
        
    if display:
        display_images(czi_paths[display_idx])
        
#%% 

    # czi_path = czi_paths[0]

    # def load_images(img_path):
    #     C1 =  io.imread(str(img_path) + "_C1.tif")
    #     C2 =  io.imread(str(img_path) + "_C2.tif")
    #     prd = io.imread(str(img_path) + "_predictions.tif")
    #     return C1, C2, prd

    # # Initialize
    # metadata = extract_metadata(czi_path)
    # exp_path = Path(czi_path.parent / czi_path.stem)
    # img_paths = [
    #     Path(str(path).replace("_C1.tif", "")) 
    #     for path in exp_path.glob("*_C1.tif")
    #     ]
    # img_names = [path.stem for path in img_paths]
    # rS = [int(name.split("_")[2]) for name in img_names]

    # # Load images
    # t0 = time.time()
    # print(f"load {czi_path.name}", end=" ", flush=True)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #     imports = list(executor.map(load_images, img_paths))
    # C1s, C2s, prds = zip(*imports)
    # t1 = time.time()
    # print(f"({t1 - t0:.3f}s)")
    
    # # Detect C2 objects -------------------------------------------------------
    
    # C2_dog_sigma1 = 1 
    # C2_dog_sigma2 = 8
    # C2_dog_thresh = 1
        
    # C2_lbls, C2_outs = [], []
    
    # for C2 in C2s:
        
    #     gbl1 = gaussian(C2, sigma=C2_dog_sigma1) # parameter
    #     gbl2 = gaussian(C2, sigma=C2_dog_sigma2) # parameter
    #     C2_dog = (gbl1 - gbl2) / gbl2
    #     C2_msk = C2_dog > C2_dog_thresh # parameter
    #     C2_msk = remove_small_objects(C2_msk, min_size=C2_min_area // 2) # parameter
    #     C2_lbl = label(C2_msk)
    #     C2_out = binary_dilation(C2_msk) ^ C2_msk
        
    #     C2_lbls.append(C2_lbl)
    #     C2_outs.append(C2_out)
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(np.stack(C2s), colormap="green")
    # viewer.add_image(np.stack(C2_outs), blending="additive")
    # viewer.add_labels(np.stack(C2_lbls)) 