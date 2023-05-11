import numpy as np
from ..arl.clean import deconvolve_cube


def clean_from_vis(vis, ops, psf, weights=None, params={},  init=None):
    
    if weights is None:
        weights = np.ones_like(vis.flatten())
    else:
        weights = weights.flatten()
        assert len(weights) == len(vis.flatten()) 

    forward, backward = ops
    # nvis, nfreq = vis.shape

    if init is None:
        dirty = backward(np.multiply(vis.flatten(), weights))#/len(vis.flatten())
        # dirty = backward(vis.flatten())#/len(vis.flatten())
    else:
        dirty = backward(np.multiply(vis.flatten(), weights)) #- backward(forward(init).flatten()) #/len(vis.flatten())
        # dirty = backward(vis.flatten())#/len(vis.flatten())


    comp, residual = deconvolve_cube(dirty, np_psf=psf, params=params)
    comp = comp.data.reshape(dirty.shape) # + init if init is not None else comp.data.reshape(dirty.shape)



    return comp
    


