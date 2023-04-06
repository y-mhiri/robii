import numpy as np
from ducc0.wgridder import ms2dirty, dirty2ms
from copy import deepcopy

from torchvision.transforms import ToTensor
import torch

import matplotlib.pyplot as plt

def robust_em_imager(vis, uvw, freq, cellsize, niter, 
                model_image=None, npix_x=32, npix_y=32, 
                dof=10, threshold=0.01, gaussian=False, mstep_size=1,
                miter=1, verbose=False, plot=False, out='out'):

    """
    The product by the model matrix or its adjoint is done with dirty2ms and ms2dirty


    """
    # vis = vis.reshape(-1)
    nvis  =  len(vis)
    nfreq =  len(freq)

    if model_image is None:
        model_image = ms2dirty(  
                                uvw = uvw,
                                freq = freq,
                                ms = vis.reshape(-1,len(freq)),
                                npix_x = npix_x,
                                npix_y = npix_y,
                                pixsize_x = cellsize,
                                pixsize_y = cellsize,
                                epsilon=1.0e-5 
                        )/nvis

    model_image_k = deepcopy(model_image) 
    npix_x, npix_y = model_image.shape


    expected_tau = np.ones((nvis, nfreq))
    # expected_tau = np.ones(nvis)

    if verbose:
        print('Starting robust imager')
        print('niter: ', niter)
        print('miter: ', miter)
        print('dof: ', dof)
        print('threshold: ', threshold)
        print('mstep_size: ', mstep_size)
        print('gaussian: ', gaussian)
        print('npix_x: ', npix_x)
        print('npix_y: ', npix_y)
        print('cellsize: ', cellsize)

    for it in range(niter):
        
        if verbose:
            print('Iteration: ', it)
            print('Computing model visibilities')
                  
        model_vis = dirty2ms(  
                            uvw = uvw,
                            freq = freq,
                            dirty = model_image_k,
                            pixsize_x = cellsize,
                            pixsize_y = cellsize,
                            epsilon=1.0e-5 
                    )

        if verbose:
            print("model_vis: ", model_vis.shape)
            print('Computing residual')

        # residual = (vis.reshape(-1) - model_vis.reshape(-1))
        # sigma2 = (1/nvis) * np.linalg.norm(np.multiply(np.sqrt(expected_tau), residual))**2
        # expected_tau = (dof + 1)/(dof + (1/sigma2) * np.linalg.norm(residual.reshape(-1,1), axis=1)**2) 

        # residual = (vis.reshape(-1) - model_vis.reshape(-1))
        residual = (vis - model_vis)
        # residual = np.multiply(np.sqrt(expected_tau)**-1 , residual) ?
        residual = np.multiply(np.sqrt(expected_tau) , residual)

        sigma2 = (1/nvis) * np.linalg.norm(residual, axis=0)**2
        if verbose:
            print('Computing expected weights')

        residual = np.linalg.norm(residual[...,np.newaxis], axis=-1)**2

        print("residual: ", residual.shape)
        expected_tau = (dof + 1)/(dof + (1/sigma2) * residual)
        print("expected_tau: ", expected_tau.shape)
        if gaussian:
            expected_tau = np.ones((nvis, nfreq)) 
            # expected_tau = np.ones(nvis) 

        if verbose:
            print('E step done.')
            print('Starting M step')
        
        for mit in range(miter):
            if verbose:
                print('M iteration: ', mit)
                print('Computing residual image')

            model_vis = dirty2ms(  
                                uvw = uvw,
                                freq = freq,
                                dirty = model_image_k,
                                pixsize_x = cellsize,
                                pixsize_y = cellsize,
                                epsilon=1.0e-7 
                        )

            if verbose:
                print('Computing residual')

            residual = (vis - model_vis)
            residual = np.multiply(expected_tau , residual)
            # residual = (vis.reshape(-1) - model_vis.reshape(-1))
            # residual = np.multiply(expected_tau , residual.reshape(-1))
            #print("residual: ", residual.shape)

            if verbose:
                print('Computing residual image')

            residual_image = ms2dirty(  
                                uvw = uvw,
                                freq = freq,
                                ms = residual.reshape(-1,len(freq)),
                                npix_x = npix_x,
                                npix_y = npix_y,
                                pixsize_x = cellsize,
                                pixsize_y = cellsize,
                                epsilon=1.0e-7 
                        )/nvis

            if verbose:
                print('Updating model image')

            model_image_k = model_image_k + mstep_size*residual_image
            model_image_k = np.sign(model_image_k) * np.max([np.abs(model_image_k)- threshold*np.max(np.abs(model_image_k)), np.zeros(model_image_k.shape)], axis=0)
            # model_image_k = np.abs(model_image_k)

            if plot:
                plt.figure()
                plt.title('Model image')
                plt.imshow(model_image_k)
                plt.colorbar()
                plt.show()
                # plt.imsave(f'{out}_tmp.png', model_image_k, origin='lower', cmap='Spectral_r')



            
    return model_image_k


