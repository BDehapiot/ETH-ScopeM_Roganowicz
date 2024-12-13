#%% Imports -------------------------------------------------------------------

import pickle
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import segmentation_models as sm

# Functions
from bdmodel.functions import preprocess, augment

# Tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint
    )

# Matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%% Comments ------------------------------------------------------------------

'''
'''

#%% Function(s) ---------------------------------------------------------------

def split_idx(n, validation_split=0.2):
    val_n = int(n * validation_split)
    trn_n = n - val_n
    idx = np.arange(n)
    np.random.shuffle(idx)
    trn_idx = idx[:trn_n]
    val_idx = idx[trn_n:]
    return trn_idx, val_idx

def save_val_prds(imgs, msks, prds, save_path):

    plt.ioff() # turn off inline plot
    
    for i in range(imgs.shape[0]):

        # Initialize
        fig, (ax0, ax1, ax2) = plt.subplots(
            nrows=1, ncols=3, figsize=(15, 5))
        cmap0, cmap1, cmap2 = cm.gray, cm.plasma, cm.plasma
        shrink = 0.75

        # Plot img
        ax0.imshow(imgs[i], cmap=cmap0)
        ax0.set_title("image")
        ax0.set_xlabel("pixels")
        ax0.set_ylabel("pixels")
        fig.colorbar(
            cm.ScalarMappable(cmap=cmap0), ax=ax0, shrink=shrink)

        # Plot msk
        ax1.imshow(msks[i], cmap=cmap1)
        ax1.set_title("mask")
        ax1.set_xlabel("pixels")
        fig.colorbar(
            cm.ScalarMappable(cmap=cmap1), ax=ax1, shrink=shrink)
        
        # Plot prd
        ax2.imshow(prds[i], cmap=cmap2)
        ax2.set_title("prediction")
        ax2.set_xlabel("pixels")
        fig.colorbar(
            cm.ScalarMappable(cmap=cmap2), ax=ax2, shrink=shrink)
        
        plt.tight_layout()
        
        # Save
        Path(save_path, "val_prds").mkdir(exist_ok=True)
        plt.savefig(save_path / "val_prds" / f"expl_{i:02d}.png")
        plt.close(fig)

#%% Class: Train() ------------------------------------------------------------

class Train:
       
    def __init__(
            self, 
            imgs, msks,
            save_name="",
            save_path=Path.cwd(),
            msk_type="normal",
            img_norm="global",
            patch_size=128,
            patch_overlap=32,
            nAugment=0,
            backbone="resnet18",
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            learning_rate=0.001,
            patience=20,
            weights_path="",
            ):
        
        self.imgs = imgs
        self.msks = msks
        self.save_name = save_name
        self.save_path = save_path
        self.msk_type = msk_type
        self.img_norm = img_norm
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.nAugment = nAugment
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.patience = patience
        self.weights_path = weights_path
        
        # Model name
        self.date = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        if not self.save_name:
            self.save_name = f"model_{self.date}"
        else:
            self.save_name = f"model_{self.save_name}"

        # Save path
        self.save_path = Path(Path.cwd(), self.save_name)
        self.backup_path = Path(Path.cwd(), f"{self.save_name}_backup")
        if self.save_path.exists():
            if self.weights_path and self.weights_path.exists():
                if self.backup_path.exists():
                    shutil.rmtree(self.backup_path)
                shutil.copytree(self.save_path, self.backup_path)
            shutil.rmtree(self.save_path)
        self.save_path.mkdir(exist_ok=True)
        
        # Preprocess
        self.imgs, self.msks = preprocess(
            self.imgs, self.msks,
            msk_type=self.msk_type, 
            img_norm=self.img_norm,
            patch_size=self.patch_size, 
            patch_overlap=self.patch_overlap,
            )
        self.nImg = self.imgs.shape[0]
        
        # Augment
        if self.nAugment > 0:
            self.imgs, self.msks = augment(
                self.imgs, self.msks, self.nAugment,
                )
            
        # Split indexes
        self.trn_idx, self.val_idx = split_idx(
            self.imgs.shape[0], validation_split=self.validation_split) 

        # Train
        self.setup()
        self.train()
        self.save()
        
    # Train -------------------------------------------------------------------
        
    def setup(self):
        
        # Model
        
        self.model = sm.Unet(
            self.backbone, 
            input_shape=(None, None, 1), 
            classes=1, 
            activation="sigmoid", 
            encoder_weights=None,
            )
        
        if self.weights_path:
            self.model.load_weights(
                Path(Path.cwd(), f"{self.save_name}_backup", "weights.h5"))
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy", 
            metrics=["mse"],
            )
        
        # Checkpoint
        self.checkpoint = ModelCheckpoint(
            filepath=Path(self.save_path, "weights.h5"),
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True
            )
        
        # Callbacks
        self.customCallback = CustomCallback(self)
        self.callbacks = [
            EarlyStopping(patience=self.patience, monitor='val_loss'),
            self.checkpoint, self.customCallback
            ]
    
    def train(self):
        
        self.history = self.model.fit(
            x=self.imgs[self.trn_idx], 
            y=self.msks[self.trn_idx],
            validation_data=(
                self.imgs[self.val_idx],
                self.msks[self.val_idx]
                ),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            verbose=0,
            ) 
        
    def save(self):      
        
        # Report
        self.customCallback.fig.savefig(
            Path(self.save_path , "report_plot.png"))
        # plt.close(self.customCallback.fig)
        
        idx = np.argmin(self.history.history["val_loss"])
        self.report = {
            
            # Parameters
            "date"             : self.date,
            "save_name"        : self.save_name,
            "save_path"        : self.save_path,
            "msk_type"         : self.msk_type,
            "img_norm"         : self.img_norm,
            "patch_size"       : self.patch_size,
            "patch_overlap"    : self.patch_overlap,
            "img/patches"      : self.nImg,
            "augmentation"     : self.nAugment,
            "backbone"         : self.backbone,
            "epochs"           : self.epochs,
            "batch_size"       : self.batch_size,
            "validation_split" : self.validation_split,
            "learning_rate"    : self.learning_rate,
            "patience"         : self.patience,
            
            # Results
            "best_epoch"       : idx,
            "best_val_loss"    : self.history.history["val_loss"][idx], 
            
            } 
                
        with open(str(Path(self.save_path, "report.txt")), "w") as f:
            for key, value in self.report.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
                    
        with open(Path(self.save_path) / "report.pkl", "wb") as f:
            pickle.dump(self.report, f)

        # History
        self.history_df = pd.DataFrame(self.history.history)
        self.history_df = self.history_df.round(5)
        self.history_df.index.name = 'Epoch'
        self.history_df.to_csv(Path(self.save_path, "history.csv"))
                    
        # Validation predictions
        nPrds = 50
        val_imgs = self.imgs[self.val_idx[:nPrds]]
        val_msks = self.msks[self.val_idx[:nPrds]]
        val_prds = np.stack(self.model.predict(val_imgs).squeeze())
        save_val_prds(val_imgs, val_msks, val_prds, self.save_path)

#%% Class: CustomCallback -----------------------------------------------------

class CustomCallback(Callback):
    
    def __init__(self, train):
        
        super(CustomCallback, self).__init__()
        self.train = train
        self.trn_loss, self.val_loss = [], []
        self.trn_mse, self.val_mse = [], []
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.35)
        self.axsub = None
        plt.rcParams["font.family"] = "Consolas"
        plt.rcParams["font.size"] = 12
        plt.ion()

        # Add stop button
        self.stop_training = False
        axbutton = plt.axes([0.80, 0.075, 0.1, 0.075])
        self.stop_button = Button(axbutton, 'Stop')
        def stop_training(event):
            self.stop_training = True
        self.stop_button.on_clicked(stop_training)

    def on_epoch_end(self, epoch, logs=None):
        
        # Get loss and mse values
        trn_loss = logs["loss"]
        val_loss = logs.get("val_loss")
        trn_mse = logs["mse"]
        val_mse = logs.get("val_mse")
        self.trn_loss.append(trn_loss)
        self.val_loss.append(val_loss)
        self.trn_mse.append(trn_mse)
        self.val_mse.append(val_mse)

        # Main plot -----------------------------------------------------------
        
        self.ax.clear()
        self.ax.plot(
            range(1, epoch + 2), self.trn_loss, "y", label="training loss")
        self.ax.plot(
            range(1, epoch + 2), self.val_loss, "r", label="validation loss")
        self.ax.set_title(f"{self.train.save_name}")
        self.ax.set_xlabel("epochs")
        self.ax.set_ylabel("loss")
        self.ax.legend(
            loc="upper right", bbox_to_anchor=(1, -0.1), borderaxespad=0.)
                
        # Subplot -------------------------------------------------------------

        if self.axsub is not None:
            self.axsub.clear()
        else:
            self.axsub = inset_axes(
                self.ax, width="50%", height="50%", loc="upper right")
        self.axsub.plot(
            range(1, epoch + 2), self.trn_loss, "y", label="training loss")
        self.axsub.plot(
            range(1, epoch + 2), self.val_loss, "r", label="validation loss")
        self.axsub.set_xlabel("epochs")
        self.axsub.set_ylabel("loss")
        
        n = 10 # dynamic y axis
        if len(self.val_loss) < n: 
            trn_loss_avg = np.mean(self.trn_loss)
            val_loss_avg = np.mean(self.val_loss)
        else:
            trn_loss_avg = np.mean(self.trn_loss[-n:])
            val_loss_avg = np.mean(self.val_loss[-n:])
        y_min = np.minimum(trn_loss_avg, val_loss_avg) * 0.75
        y_max = np.maximum(trn_loss_avg, val_loss_avg) * 1.25
        if y_min > np.minimum(trn_loss, val_loss):
            y_min = np.minimum(trn_loss, val_loss) * 0.75
        if y_max < np.maximum(trn_loss, val_loss):
            y_max = np.maximum(trn_loss, val_loss) * 1.25
        self.axsub.set_ylim(y_min, y_max)
                       
        # Info ----------------------------------------------------------------
        
        info_path = (
            
            f"date : {self.train.date}\n"
            f"save_name : {self.train.save_name}\n"
            f"save_path : {self.train.save_path}\n"
            
            ) 
        
        info_parameters = (
            
            f"Parameters\n"
            f"----------\n"
            f"msk_type         : {self.train.msk_type}\n"
            f"img_norm         : {self.train.img_norm}\n"
            f"patch_size       : {self.train.patch_size}\n"
            f"patch_overlap    : {self.train.patch_overlap}\n"
            f"img/patches      : {self.train.nImg}\n"
            f"augmentation     : {self.train.nAugment}\n"
            f"backbone         : {self.train.backbone}\n"
            f"batch_size       : {self.train.batch_size}\n"
            f"validation_split : {self.train.validation_split}\n"
            f"learning_rate    : {self.train.learning_rate}\n"

            )
                
        info_monitoring = (

            f"Monitoring\n"
            f"----------\n"
            f"epoch    : {epoch + 1} / {self.train.epochs} ({np.argmin(self.val_loss) + 1})\n"
            f"trn_loss : {logs['loss']:.4f}\n"
            f"val_loss : {logs['val_loss']:.4f} ({np.min(self.val_loss):.4f})\n"
            f"trn_mse  : {logs['loss']:.4f}\n"
            f"val_mse  : {logs['val_mse']:.4f}\n"
            f"patience : {epoch - np.argmin(self.val_loss)} / {self.train.patience}\n"
            
            )
                
        self.ax.text(
            0.00, -0.1, info_path,  
            transform=self.ax.transAxes, 
            ha="left", va="top", color="black",
            )
                
        self.ax.text(
            0.00, -0.2, info_parameters,  
            transform=self.ax.transAxes, 
            ha="left", va="top", color="black",
            )
       
        self.ax.text(
            0.35, -0.2, info_monitoring,  
            transform=self.ax.transAxes, 
            ha="left", va="top", color="black",
            )
        
        # Draw ----------------------------------------------------------------

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1) 

        # Exit ----------------------------------------------------------------

        if self.stop_training:
            self.model.stop_training = True
            print("Training stopped")