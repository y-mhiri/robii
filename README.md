# ROBII: ROBust Interferometric Imaging Algorithm (in development)

ROBII is a Python package for interferometric imaging that uses a robust statistical model method to solve the imaging problem.
Additionnaly, ROBII includes RobiiNet, an unrolled deep neural network for radio-interferometric imaging based on the Robii algorithm.

For more details on the methodology, please refere to y-mhiri.github.io/publications

## Requirements

The following dependencies are required to run ROBII:

### Ubuntu 

The following requirements has been succesfully installed and tested on Ubuntu focal (20.04), support for other distribution and version is not garranteed.

- Python 3.8.10

Python can be installed with the following command 

        $ sudo apt-get install python3.8

Alternatively, a virtual environment manager like pyenv or conda can be used.

- KERN suite

KERN ([see here](https://kernsuite.info/)) is a set of radio astronomical software packages that contain most of the standard tools that a radio astronomer needs to work with radio telescope data.

        sudo apt-get install software-properties-common
        $ sudo add-apt-repository -s ppa:kernsuite/kern-8
        $ sudo apt-add-repository multiverse
        $ sudo apt-add-repository restricted
        $ sudo apt-get update

- Casacore

Casacore ([see here](https://github.com/casacore/casacore)) is a suite of C++ libraries for radio astronomy data processing and can be installed via KERN:

        $ sudo apt-get install python3-casacore



### MacOS

You can install the required packages on MacOS using Conda. First, install Conda by following the instructions here. Then, create a new environment with Python:

        $ conda create -n env_name python==3.8.10

Casacore can then be installed with the following command:

        $ conda install -c conda-forge casacore


### Windows

Use Vagrant ([see here](https://www.vagrantup.com/)) to build a ubuntu-focal image and follow the instructions for Ubuntu


## Installation 

1. Clone the ROBII repository:

        $ git clone https://github.com/y-mhiri/robbi.git

2. Navigate to the robii directory and install the package using pip:

        $ cd /path/to/robii
        $ pip install .
    

## Usage

### Create a dataset 

You can simulate a radio interferometric dataset using the following command 

       $ generate_dataset simulate --ndata 1000 --telescope vla --synthesis_time 1 --integration_time 60 --frequency 3.0e8 --npixel 128 --add_noise True --snr 20 --out /path/to/dataset/dataset_name

### Train a RobiiNet Model 

        $ train_model --dset_path /path/to/dataset --nepoche 100 --batch_size 64 --net_depth 10 --net_width 351 --learning_rate 0.001 --step 10 --model_name robiinet_trained --true_init 

### Make an image from 

ROBII can create an image from a zarr file containing a radio interferometric dataset dataset using the robii fromzarr command. Here's an example:

        $ robii fromzarr data.zarr -n 3 --out images --niter 10 --threshold 0.001 --dof 10

ROBII can create an image from an MS file using the robii __fromms__ command. The support for measurement set is still in progress and some issues can occur. The command is currently set to make an image from a unique spectral window. The robii imager can be tested on the measurement set from the Very Large Array telescope available [here](https://casaguides.nrao.edu/index.php?title=VLA_CASA_Imaging-CASA6.5.2). 

        $ robii fromms /path/to/ms robii fromms /path/to/ms --out path/to/out/filename --image_size 1280 --niter 10 --miter 10 --mstep_size 0.000005 --threshold 0.00001 --dof 10 --plot --fits






