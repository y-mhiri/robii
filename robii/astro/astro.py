import numpy as np
from ..math.linalg import vec

def generate_directions(npix, cellsize):


    """
    generate direction cosines vectors (l,m,n)
    for an image of size (npix, npix) and a cellsize defined in radians
    Input :
    - (int) npix : image pixel size
    - (float) cellsize : size of a pixel in radians

    Returns : (ndarray) lmn 
    """

    k = np.arange(0, npix)
    l_grid = (-npix/2 + k)*cellsize
    m_grid = (-npix/2 + k)*cellsize


    LGRID, MGRID = np.meshgrid(l_grid, m_grid)
    NGRID = np.sqrt(1 - LGRID**2 - MGRID**2)


    l = vec(LGRID)
    m = vec(MGRID)
    n = vec(NGRID)

    lmn = np.array([l,m,n]).reshape(3,-1)

    # lmn = np.array([l,m,n]).T

    return lmn