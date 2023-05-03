import numpy as np
from ..arl.clean import deconvolve_cube


def clean_from_vis(vis, ops, psf, params={},  init=None):
    
    forward, backward = ops
    # nvis, nfreq = vis.shape

    if init is None:
        dirty = backward(vis.flatten())/len(vis.flatten())
    else:
        dirty = backward(vis.flatten() - forward(init).flatten())/len(vis.flatten())


    comp, residual = deconvolve_cube(dirty, np_psf=psf, params=params)
    comp = comp.data.reshape(dirty.shape) + init if init is not None else comp.data.reshape(dirty.shape)

    return comp
    


