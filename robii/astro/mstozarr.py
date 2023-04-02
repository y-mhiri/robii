"""
Extract uv plane and antenna positions from a measurement set

"""


import numpy as np
import pandas as pd
import zarr

from astropy.coordinates import EarthLocation
from .ms import MS


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

    telescope_location_xyz = np.mean(antenna_positions, axis=0)
    telescope_location = EarthLocation.from_geocentric(*telescope_location_xyz, unit='m')
    telescope_location_lat_lon = (telescope_location.lon.rad, telescope_location.lat.rad)
    root.create_dataset('uvw', data=uvw)
    root.create_dataset('antenna_positions', data=antenna_positions)
    root.create_dataset('index', data=index)
    root.create_dataset('telescope_location', data=telescope_location_lat_lon)

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
    # telescope_location = z['telescope_location'][:]

    telescope_location_xyz = np.mean(antenna_positions, axis=0)
    telescope_location = EarthLocation.from_geocentric(*telescope_location_xyz, unit='m')
    telescope_location_lon_lat = (telescope_location.lon.rad, telescope_location.lat.rad)
    
    return uvw, index, antenna_positions, telescope_location_lon_lat


def load_telescope_from_itrf(path):
    """
    Load the uv plane and antenna positions from a text file in ITRF format

    Parameters
    ----------
    path : str
        Path to the text file
    """

    file = open(path, 'r')
    lines = file.readlines()
    file.close()


    lines = [line.split() for line in lines]
    lines = [[v for v in line] for line in lines]

    # Firts line indicates the column names
    # put in a dataframe
    df = pd.DataFrame(lines, columns=lines[0])
    df = df.drop(0)

    # Convert X, Y, Z columns to float
    df['X'] = df['X'].astype(float)
    df['Y'] = df['Y'].astype(float)
    df['Z'] = df['Z'].astype(float)

    antenna_positions = df[['X', 'Y', 'Z']].values

    telescope_location = EarthLocation.from_geocentric(*np.mean(antenna_positions, axis=0), unit='m')
    telescope_location_lon_lat = (telescope_location.lon.rad, telescope_location.lat.rad)

    return antenna_positions, telescope_location_lon_lat



