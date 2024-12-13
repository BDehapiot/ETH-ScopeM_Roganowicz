#%% Imports -------------------------------------------------------------------

from pathlib import Path

# bdmodels
from bdmodel.annotate import Annotate

#%% Inputs --------------------------------------------------------------------

# Paths
train_path = Path(Path.cwd(), "data", "train_tissue")

# Parameters
randomize = True
# np.random.seed(42)

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    annotate = Annotate(train_path, randomize=randomize)
