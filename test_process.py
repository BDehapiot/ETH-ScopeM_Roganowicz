#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# skimage
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation

# Qt
from qtpy.QtWidgets import (
    QWidget, QPushButton, QRadioButton, QLabel, QLineEdit,
    QGroupBox, QVBoxLayout, QHBoxLayout
    )

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\\local_Roganowicz\\data\\")
exp = "2025-03_mutants_norfloxacin"
# exp = "2025-04_mutants_nitrofurantoin"

parameters = {
    
    # Paths
    "plate_idx" : 11,
    "scene_idx" : 745,
    
    # Parameters
    "C2_min_area"     : 32, # Default : 32
    "C2_min_mean_int" : 24, # Default : 24
    "C2_dog_sigma1"   : 1,  # Default : 1
    "C2_dog_sigma2"   : 8,  # Default : 8
    "C2_dog_thresh"   : 1,  # Default : 1
    
    }

#%% Class : Process -----------------------------------------------------------

class Process:
    
    def __init__(self, data_path, exp, parameters):
    
        # Fetch    
        self.data_path = data_path
        self.exp = exp
        self.parameters = parameters

        # Run
        self.load()
        self.process()
    
    def load(self):
        
        # Fetch
        data_path = self.data_path / self.exp
        plate_idx = self.parameters["plate_idx"]
        scene_idx = self.parameters["scene_idx"]
        
        # Initialize
        plate_paths = [
            p for p in data_path.iterdir() 
            if p.is_dir() and p.name.startswith("p")
            ]
        for path in plate_paths[plate_idx].iterdir():
            if path.name.endswith("C2.tif"):
                if int(path.stem.split("_")[5]) == scene_idx:
                    C2_path = path
                    
        # Load
        self.C2 = io.imread(C2_path)
                
    def process(self):
        
        # Fetch 
        C2_min_area     = self.parameters["C2_min_area"]
        C2_min_mean_int = self.parameters["C2_min_mean_int"]
        C2_dog_sigma1   = self.parameters["C2_dog_sigma1"]
        C2_dog_sigma2   = self.parameters["C2_dog_sigma2"]
        C2_dog_thresh   = self.parameters["C2_dog_thresh"]

        # Detect C2 objects
        gbl1 = gaussian(self.C2, sigma=C2_dog_sigma1)
        gbl2 = gaussian(self.C2, sigma=C2_dog_sigma2)
        C2_dog = (gbl1 - gbl2) / gbl2
        C2_msk = C2_dog > C2_dog_thresh
        C2_out = binary_dilation(C2_msk) ^ C2_msk
        C2_lbl = label(C2_msk)
        
        # Filter C2 objects
        C2_msk_valid = C2_msk.copy()        
        for props in regionprops(C2_lbl, intensity_image=self.C2):
            lbl = props.label
            mean_int = np.mean(self.C2[C2_lbl == lbl])
            area = props.area
            if (area < C2_min_area) or (mean_int < C2_min_mean_int):
                C2_msk_valid[C2_lbl == lbl] = 0       
        C2_out_valid = binary_dilation(C2_msk_valid) ^ C2_msk_valid
        C2_out = C2_out ^ C2_out_valid
        
        # Append instance
        self.gbl1 = gbl1
        self.gbl2 = gbl2
        self.C2_dog = C2_dog
        self.C2_msk = C2_msk 
        self.C2_out = C2_out 
        self.C2_msk_valid = C2_msk_valid
        self.C2_out_valid = C2_out_valid

#%% Class : Display -----------------------------------------------------------

class Display:
    
    def __init__(self, process):
        
        # Fetch    
        self.process = process
        self.parameters = process.parameters
        
        # Execute
        self.initialize()
        
    def initialize(self):
        
        self.viewer = napari.Viewer()
        
        # Create "display" menu
        self.dsp_group_box = QGroupBox("Display")
        dsp_group_layout = QHBoxLayout()
        self.rad_raw = QRadioButton("raw")
        self.rad_dog = QRadioButton("dog")
        self.rad_raw.setChecked(True)
        dsp_group_layout.addWidget(self.rad_raw)
        dsp_group_layout.addWidget(self.rad_dog)
        self.dsp_group_box.setLayout(dsp_group_layout)
        self.rad_raw.toggled.connect(
            lambda checked: self.show_raw() if checked else None)
        self.rad_dog.toggled.connect(
            lambda checked: self.show_dog() if checked else None)
        
        # Create "select" menu
        self.slc_group_box = QGroupBox("Select")
        slc_group_layout = QVBoxLayout()
        self.slc_plate_idx = QLineEdit()
        self.slc_plate_idx.setText(f"{self.parameters['plate_idx']}")
        self.slc_scene_idx = QLineEdit()
        self.slc_scene_idx.setText(f"{self.parameters['scene_idx']}")
        slc_group_layout.addWidget(QLabel("plate_idx : "))
        slc_group_layout.addWidget(self.slc_plate_idx)
        slc_group_layout.addWidget(QLabel("scene_idx : "))
        slc_group_layout.addWidget(self.slc_scene_idx)
        self.slc_group_box.setLayout(slc_group_layout)

        # Create "parameter" menu
        self.prm_group_box = QGroupBox("Parameters")
        prm_group_layout = QVBoxLayout()
        self.prm_min_area = QLineEdit()
        self.prm_min_area.setText(f"{self.parameters['C2_min_area']}")
        self.prm_min_mean_int = QLineEdit()
        self.prm_min_mean_int.setText(f"{self.parameters['C2_min_mean_int']}")
        self.prm_dog_sigma1 = QLineEdit()
        self.prm_dog_sigma1.setText(f"{self.parameters['C2_dog_sigma1']}")
        self.prm_dog_sigma2 = QLineEdit()
        self.prm_dog_sigma2.setText(f"{self.parameters['C2_dog_sigma2']}")
        self.prm_dog_thresh = QLineEdit()
        self.prm_dog_thresh.setText(f"{self.parameters['C2_dog_thresh']}")
        prm_group_layout.addWidget(QLabel("min_area : "))
        prm_group_layout.addWidget(self.prm_min_area)
        prm_group_layout.addWidget(QLabel("min_mean_int : "))
        prm_group_layout.addWidget(self.prm_min_mean_int)
        prm_group_layout.addWidget(QLabel("dog_sigma1 : "))
        prm_group_layout.addWidget(self.prm_dog_sigma1)
        prm_group_layout.addWidget(QLabel("dog_sigma2 : "))
        prm_group_layout.addWidget(self.prm_dog_sigma2)
        prm_group_layout.addWidget(QLabel("dog_thresh : "))
        prm_group_layout.addWidget(self.prm_dog_thresh)
        self.prm_group_box.setLayout(prm_group_layout)
        
        # Create "update" menu
        self.upd_group_box = QGroupBox("Update")
        upd_group_layout = QVBoxLayout()
        self.upd_update = QPushButton("update")
        upd_group_layout.addWidget(self.upd_update)
        self.upd_group_box.setLayout(upd_group_layout)
        self.upd_update.clicked.connect(self.update)
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.dsp_group_box)
        self.layout.addWidget(self.slc_group_box)
        self.layout.addWidget(self.prm_group_box)
        self.layout.addWidget(self.upd_group_box)

        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Painter") 

        # Add raw images
        self.viewer.add_image(
            self.process.C2_out_valid, name="C2_out_valid",
            blending="additive", colormap="green", opacity=0.5,
            visible=1,
            )
        self.viewer.add_image(
            self.process.C2_out, name="C2_out",
            blending="additive", colormap="magenta", opacity=0.5,
            visible=1,
            )
        self.viewer.add_image(
            self.process.C2, name="C2", 
            blending="additive", gamma=0.5,
            visible=1, colormap="gray", 
            )
        
        # Add dog images
        self.viewer.add_image(
            self.process.gbl1, name="gbl1", 
            blending="additive",
            visible=0, colormap="gray", 
            )
        self.viewer.add_image(
            self.process.gbl2, name="gbl2", 
            blending="additive",
            visible=0, colormap="gray", 
            )
        self.viewer.add_image(
            self.process.C2_dog, name="C2_dog", 
            blending="additive",
            visible=0, colormap="gray", 
            )
        
    def show_raw(self):
        for name in self.viewer.layers:
            name = str(name)
            if name in ["gbl1", "gbl2", "C2_dog"]:
                self.viewer.layers[name].visible = 0
            else:
                self.viewer.layers[name].visible = 1
    
    def show_dog(self):
        for name in self.viewer.layers:
            name = str(name)
            if name in ["gbl1", "gbl2", "C2"]:
                self.viewer.layers[name].visible = 0
            else:
                self.viewer.layers[name].visible = 1
        
    def update_layers(self):
        self.viewer.layers["C2_out_valid"].data = self.process.C2_out_valid
        self.viewer.layers["C2_out"].data = self.process.C2_out
        self.viewer.layers["C2"].data = self.process.C2
        self.viewer.layers["gbl1"].data = self.process.gbl1
        self.viewer.layers["gbl2"].data = self.process.gbl2
        self.viewer.layers["C2_dog"].data = self.process.C2_dog
        
    def update(self):
        self.parameters["plate_idx"] = int(self.slc_plate_idx.text())
        self.parameters["scene_idx"] = int(self.slc_scene_idx.text())
        self.parameters["C2_min_area"] = int(self.prm_min_area.text())
        self.parameters["C2_min_mean_int"] = int(self.prm_min_mean_int.text())
        self.parameters["C2_dog_sigma1"] = int(self.prm_dog_sigma1.text())
        self.parameters["C2_dog_sigma2"] = int(self.prm_dog_sigma2.text())
        self.parameters["C2_dog_thresh"] = float(self.prm_dog_thresh.text())
        self.process = Process(data_path, exp, self.parameters)
        self.update_layers()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    process = Process(data_path, exp, parameters)
    display = Display(process)
