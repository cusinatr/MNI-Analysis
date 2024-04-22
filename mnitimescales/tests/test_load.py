from mnitimescales.load import Load
from pathlib import Path

load_class = Load(mat_path=Path("F:\\MNIOpen\\Data\\Raw\\MatlabFile.mat"))

load_class.load_epo_stage("N3")

print(load_class.epochs)