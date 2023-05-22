import numpy as np
from ducc0.wgridder import ms2dirty, dirty2ms
from copy import deepcopy

from torchvision.transforms import ToTensor
import torch

import matplotlib.pyplot as plt
from astropy.io import fits

import os


def em_imager_old(vis, uvw, freq, cellsize, niter, 
                model_image=None, npix_x=32, npix_y=32, 
                dof=10, threshold=0.01, gaussian=False, mstep_size=None, mstep='ista',
                miter=1, verbose=False, plot=False, save_fits=False, out='out'):

    """
 
    
    """
    # vis = vis.reshape(-1)
    nvis  =  len(vis)
    nfreq =  len(freq)

    lipschitz_constant = nvis * npix_x * npix_y  # The forward operator matrix H have Lipschitz constant equal to nvis * npix_x * npix_y (Tr(H^H @ H) = nvis * npix_x * npix_y)
    mstep_size = 1/lipschitz_constant if mstep_size is None else mstep_size
    
    if mstep_size > 1/lipschitz_constant:
        print('mstep_size is too large. It should be less than 1/lipschitz_constant = {}'.format(1/lipschitz_constant))


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


    expected_weights = np.ones((nvis, nfreq))
    # expected_weights = np.ones(nvis)

    if verbose:
        print('Starting imager')
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
        # sigma2 = (1/nvis) * np.linalg.norm(np.multiply(np.sqrt(expected_weights), residual))**2
        # expected_weights = (dof + 1)/(dof + (1/sigma2) * np.linalg.norm(residual.reshape(-1,1), axis=1)**2) 

        residual = (vis - model_vis)
        residual = np.multiply(np.sqrt(expected_weights) , residual)
        sigma2 = (1/nvis) * np.linalg.norm(residual, axis=0)**2


        if verbose:
            print('Computing expected weights')

        residual = np.linalg.norm(residual[...,np.newaxis], axis=-1)**2

        expected_weights = (dof + 1)/(dof + (1/sigma2) * residual)
        if gaussian:
            expected_weights = np.ones((nvis, nfreq)) 
            # expected_weights = np.ones(nvis) 

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
            residual = np.multiply(expected_weights , residual)
            # residual = (vis.reshape(-1) - model_vis.reshape(-1))
            # residual = np.multiply(expected_weights , residual.reshape(-1))
            #print("residual: ", residual.shape)

            if verbose:
                print('Computing residual image')

            residual_image = ms2dirty(  
                                uvw = uvw,
                                freq = freq,
                                ms = residual.reshape(-1,len(freq)),
                                wgt = expected_weights, # apply weights tp get the hessian for this particular iteration
                                npix_x = npix_x,
                                npix_y = npix_y,
                                pixsize_x = cellsize,
                                pixsize_y = cellsize,
                                epsilon=1.0e-7 
                        )

            if verbose:
                print('Updating model image')

            print('threshold: ', threshold)
            print(f'norm of residual image: {np.linalg.norm(mstep_size*residual_image)}')
            print(f'norm of model image: {np.linalg.norm(model_image_k)}')
        
            model_image_k = model_image_k + mstep_size*residual_image
            model_image_k = np.sign(model_image_k) * np.max([np.abs(model_image_k)- threshold, np.zeros(model_image_k.shape)], axis=0)
            # model_image_k = np.sign(model_image_k) * np.max([np.abs(model_image_k)- threshold*np.max(np.abs(model_image_k)), np.zeros(model_image_k.shape)], axis=0)
            # model_image_k = np.abs(model_image_k)

            if plot:
                # plt.figure()
                # plt.title('Model image')
                # plt.imshow(model_image_k)
                # plt.colorbar()
                # plt.show()
                if save_fits:
                    hdu = fits.PrimaryHDU(self.image)
                    if os.path.exists(f'{out}_tmp.fits'):
                        hdu.writeto(f'{out}_tmp.fits', overwrite=True)
                    else:
                        hdu.writeto(f'{out}_tmp.fits', overwrite=False)

                plt.imsave(f'{out}_tmp.png', model_image_k, origin='lower', cmap='viridis')



            
    return model_image_k


def student_estep(residual, sigma2, dof):
    return (dof + 1)/(dof + (1/sigma2) * np.linalg.norm(residual.reshape(1,-1), axis=0)**2)

def fftem_imager(vis, gridder, niter, dof, sigmae2, params, mstep_solver, estep=student_estep, init=None, verbose=True):
    
        fft2, ifft2 = np.fft.fft2, np.fft.ifft2 # check if fftshift and/or normalization is needed
        fftshift, ifftshift = np.fft.fftshift, np.fft.ifftshift

        F = lambda x: fftshift(fft2(x, norm='forward'))
        Fh = lambda x: ifft2(ifftshift(x), norm='forward') 

        grid, degrid = gridder

        if init is None:
            model_image = Fh(grid(vis))
        else:
            model_image = init
        
        model_image_k = deepcopy(model_image)

        for it in range(niter):
            
            if verbose:
                print('Iteration: ', it)

            ## Compute residual 
            residual = vis.reshape(-1) - degrid(F(model_image_k)).reshape(-1)
            # plt.figure()
            # plt.plot(vis.reshape(-1), label='vis')
            # plt.figure()
            # plt.plot(degrid(F(model_image_k)).reshape(-1), label='model vis')
            # plt.show()

            ## Compute expected weights
            sigma2 = (1/len(vis.reshape(-1))) * np.linalg.norm(residual)**2
            expected_weights = estep(residual, sigma2, dof)

            if verbose:
                print('Computing expected grid...')
            expected_grid = F(model_image_k) + sigmae2 * grid(np.multiply(expected_weights.reshape(-1), residual.reshape(-1)))
            # plt.imshow(expected_grid.real)
            # plt.colorbar()
            # plt.show()
            ## M step
            if verbose:
                print('Mstep starting...')

            model_image_k = mstep_solver(expected_grid, (F, Fh), init=model_image_k, **params)

        return model_image_k


def em_imager(vis, ops, niter, dof, params, mstep_solver, estep=student_estep, init=None, verbose=False):

    nvis = len(vis.flatten())
    forward, backward  = ops

    if init is None:
        model_image = backward(vis)/nvis
    else:
        model_image = init

    model_image_k = deepcopy(model_image)
    # sigma2 = np.linalg.norm(vis.flatten() - forward(model_image_k).flatten())**2/nvis
    # sigma2 = ((dof -2)/dof) * sigma2

    for it in range(niter):
        if verbose:
            print(f'Iteration {it}')
        ## Compute residual 
        residual = vis.flatten() - forward(model_image_k).flatten()

        ## Compute expected weights
        sigma2 = (1/nvis) * np.linalg.norm(residual)**2
        # sigma2 = ((dof -2)/dof) * sigma2

        expected_weights = estep(residual, sigma2, dof)

        if verbose:
            print('Estep done...')
        ## M step

        ## Update model image
        model_vis = np.multiply(np.sqrt(expected_weights).flatten(), vis.flatten())

        if verbose:
            print("MStep starting...")
        model_image_k_temp = mstep_solver(model_vis, ops, weights=np.sqrt(expected_weights), init=model_image_k, **params)
        
        
        delta = np.linalg.norm(model_image_k - model_image_k_temp)**2 #/ len(model_image.flatten())
        if delta < 1e-6:
            print(f'Converged at iteration {it}')
            break
        model_image_k = model_image_k_temp



        # ## Update sigma2
        # residual = vis.flatten() - forward(model_image_k).flatten()
        # expected_weights = estep(residual, sigma2, dof)
        # sigma2 = np.linalg.norm(np.multiply(residual.flatten(), np.sqrt(expected_weights.flatten())))**2/nvis

    return model_image_k



def ista(y, ops, niter, threshold, weights=None, init=None, step_size=None, decay=1, lipshitz=None, fista=False, eps=1e-6):
    """
        Solves the LASSO regression problem,
        $$
            y = Ax + \epsilon
        $$
        using the (F)ISTA algorithm.

        Parameters
        ----------
        y : array
            The data vector.
        ops : tuple
            A tuple containing the forward and backward  operators.
        niter : int
            The number of iterations to run the algorithm for.
        threshold : float
            The thresholding parameter.
        step_size : float
            The step size parameter.
        lipshitz : float
            The Lipshitz constant of the gradient of the loss function.
        fista : bool
            Whether to use the FISTA algorithm or not.
    
        Returns
        -------
        x : array
            The solution to the LASSO regression problem.
    
    """
    if weights is None:
        weights = np.ones(y.shape)
    else:
        assert weights.shape == y.shape, 'Weights must have the same shape as the data vector.'
    
    forward, backward  = ops

    weighted_forward = lambda x: np.multiply(weights.flatten(), forward(x).flatten()).reshape(y.shape)
    weighted_backward = lambda y: backward(np.multiply(weights.reshape(y.shape), y))

    # def Q(L, xk, xkm1):
    #     return np.linalg.norm(y - forward(xkm1))**2 + L/2 * np.linalg.norm(xk - xkm1)**2 + threshold * np.linalg.norm(xk, ord=1) + ((xk - xkm1).reshape(1,-1) @ backward(y - forward(xkm1)).reshape(-1,1))[0]

    # def F(xk):
    #     return np.linalg.norm(y - forward(xk))**2 + threshold * np.linalg.norm(xk, ord=1)

    if step_size is None and lipshitz is not None:
        step_size = 1/lipshitz
    elif step_size is None and lipshitz is None:
        raise ValueError('Either step_size or lipshitz must be provided.')
    else:
        pass

    # x = np.zeros(y.shape)
    if init is None:
        x = weighted_backward(y)
    else:
        x = init

    tk = 1

    x_temp = x
    xkm1 = x
    xk = x
    for it in range(niter):
        # Perform ISTA update

        model = weighted_forward(x_temp)
        r = y.reshape(model.shape) - model
        xk = x_temp + decay*step_size * weighted_backward(r)
 
        xk = np.sign(xk) * np.max([np.abs(xk) - threshold, np.zeros(xk.shape)], axis=0)

        xk = np.max([np.zeros_like(xk), xk], axis=0)
        

        # count = 0
        # while F(xk) > Q(step_size, xk, x_temp):
        #     step_size = step_size*1,1
        #     r = y.reshape(-1) - forward(x_temp)
        #     xk = x_temp + decay*step_size * backward(r)
        #     xk = np.sign(xk) * np.max([np.abs(xk) - threshold, np.zeros(xk.shape)], axis=0)
        #     count += 1
        #     print(step_size, count)
        #     if count > 100:
        #         print('Step size too small')
        #         break

        if fista:
            tkp1 = (1 + np.sqrt(1 + 4*tk**2))/2
            x_temp = xk + ( (tk - 1)/tkp1 )* (xk - xkm1)

            xkm1 = xk
            tk = tkp1
        else:
            xkm1 = x_temp
            x_temp = xk

            if np.linalg.norm(xk - xkm1)**2 < eps:
                print(f'Converged at iteration {it}')
                break



    return xk



