# Analysis of neural timescales and spatial correlations in iEEG
The following is the code accompanying the paper:

*Cusinato et al. (2024). Sleep modulates neural timescales and spatiotemporal integration in the human cortex. Preprint at bioRxiv, https://doi.org/10.1101/2024.09.26.614972*

The main code is in the *mnitimescales* package, which contains different submodules related to timescales and spatial correlation computation, slow waves detection, and parcellation of results. The repo also contains the code to run the main analyses, through the *run_** scripts. Finally, the paper figures are reproduced in the jupyter notebooks in the *Figures/* folder.

## Installation
The *mnitimescales* package and accompanying code can be downloaded from this page, e.g. by running `git clone https://github.com/cusinatr/MNI-Analysis.git`

The package can be locally installed with *pip* by navigating to the folder where the repo is installed and running `pip install .` from the terminal.

Finally, the environment to run the scripts can be installed through *conda* from the *environment.yml* file with `conda env create -f environment.yml`


