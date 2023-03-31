"""
Extract uv plane and antenna positions from a measurement set

"""

import numpy as np
import os
import sys

import zarr

from .ms import MS

import click



def save_telescope_raw(msname, out):
    """
    Save the uv plane and antenna positions in a zarr file

    Parameters
    ----------
    msname : str
        Path to the measurement set
    out : str
        Output path (without extension)
    """

    # Open the measurement set
    ms = MS(msname)

    # Get the uv plane and antenna positions
    uvw = ms.uvw
    antenna_positions = ms.antennaPos
    index = ms.uvw_index

    # Save the uv plane and antenna positions in a zarr file
    store = zarr.ZipStore(f'{out}.zip', mode='w')
    root = zarr.group(store=store)

    root.create_dataset('uvw', data=uvw)
    root.create_dataset('antenna_positions', data=antenna_positions)
    root.create_dataset('index', data=index)

    store.close()


def load_telescope(path):
    """
    Load the uv plane and antenna positions from a zarr file

    Parameters
    ----------
    path : str
        Path to the zarr file
    """

    # Open the zarr file
    z = zarr.open(path, mode='r')

    uvw = z['uvw'][:]
    antenna_positions = z['antenna_positions'][:]
    index = z['index'][:]

    return uvw, index, antenna_positions



