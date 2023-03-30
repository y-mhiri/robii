# ROBII: ROBust Interferometric Imaging Algorithm

ROBII is a Python package for interferometric imaging that uses a robust statistical model method to solve the imaging problem. For more details on the methodology, please refere to y-mhiri.github.io/publications

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

Use Vagrant to build a ubuntu-focal image and follow the instructions for Ubuntu


## Installation 

1. Clone the ROBII repository:

        $ git clone https://github.com/y-mhiri/robbi.git

2. Navigate to the robii directory and install the package using pip:

        $ cd /path/to/robii
        $ pip install .
    

## Usage

### Create a dataset 

You can simulate a radio interferometric dataset using the following command 

       $ generate_dataset simulate --ndata 10 --telescope vla --frequency 3.0e8 --npixel 128 --add_noise True --snr 20 --out /path/to/dataset/dataset_name

### Make an image

ROBII can create an image from a zarr file using the robii fromzarr command. Here's an example:

        $ robii fromzarr data.zarr -n 3 --out images --niter 10 --threshold 0.001 --dof 10





