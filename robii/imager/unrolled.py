
import numpy as np
from ducc0.wgridder import ms2dirty, dirty2ms
from copy import deepcopy

import torch 
from torchvision.transforms import ToTensor
from torch import nn


import matplotlib.pyplot as plt


class RobustLayer():

    def __init__(self, nvis, model_path):
        self.nvis = nvis
        checkpoint = torch.load(model_path)
        self.W0 = checkpoint["model_state_dict"]["estep.weight"].detach().numpy()
        self.W1 = checkpoint["model_state_dict"]["update_layer.weight"].detach().numpy()
        self.W2 = checkpoint["model_state_dict"]["memory_layer.weight"].detach().numpy()
        self.b0 = checkpoint["model_state_dict"]["estep.bias"].detach().numpy()
        self.b1 = checkpoint["model_state_dict"]["update_layer.bias"].detach().numpy()
        self.b2 = checkpoint["model_state_dict"]["memory_layer.bias"].detach().numpy()

    def forward(self, residual, wprev):
        wtemp = 1/(1+np.exp(-np.matmul(residual, self.W0) - self.b0))

        wnew = np.maximum(0, np.matmul(wtemp, self.W1) + np.matmul(wprev, self.W2) + self.b1 + self.b2)
        return wnew
    

# class RobustLayer(nn.Module):
#     def __init__(self, nvis):
#         super().__init__()

#         self.nvis = nvis
#         self.estep = nn.Linear(self.nvis, self.nvis)
#         self.update_layer = nn.Linear(self.nvis, self.nvis)
#         self.memory_layer = nn.Linear(self.nvis, self.nvis)


#     def forward(self, residual, wprev):
#         wtemp = nn.Sigmoid()(self.estep(residual))
#         wnew = nn.ReLU()(self.update_layer(wtemp) + self.memory_layer(wprev))
#         return wnew
        



def unrolled_imager(vis, model_path, freq, uvw, npix_x, npix_y, cellsize, niter,
                     miter=1, mstep_size=0.1, model_image=None, threshold=.001, 
                     verbose=False, plot=False):

    """
    Imager Algorithm that uses as Expectation Step a recurent neural network 
    to compute the weights applied to the residual
    
    """

    nvis = len(uvw)
    nfreq = len(freq)

    checkpoint = torch.load(model_path)
    net_width = checkpoint["model_state_dict"]["W.weight"].shape[0]
    print('net_width: ', net_width)
    net = RobustLayer(net_width, model_path)

    if verbose:
        print('Loading model')
        print('Model path: ', model_path)
        print('Model width: ', net_width)
        print('Model threshold: ', threshold)


    if verbose:
        print('Model loaded')

    
    if model_image is None:

        if verbose:
            print('Computing initial model image')

        print(f'freq: {freq.shape}')
        print(f'uvw: {uvw.shape}')
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

    if verbose:
        print('Starting unrolled imager')
        print('niter: ', niter)
        print('miter: ', miter)
        print('npix_x: ', npix_x)
        print('npix_y: ', npix_y)
        print('cellsize: ', cellsize)
        print('threshold: ', threshold)

    model_image_k = deepcopy(model_image) 
    expected_weights = np.ones((nvis, nfreq)) 

    model_vis = dirty2ms(  
                        uvw = uvw,
                        freq = freq,
                        dirty = model_image_k,
                        pixsize_x = cellsize,
                        pixsize_y = cellsize,
                        epsilon=1.0e-5 
                )   

    expected_weights = 1/(expected_weights + np.abs(vis - model_vis)**2)
    # expected_weights = np.ones(nvis)

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
                        epsilon=1.0e-7 
                )

        if verbose:
            print('Computing residual')

        residual = vis - model_vis
        sigma2 = (1/nvis) * np.linalg.norm(residual, axis=0)**2
            
        npass = nvis // net_width
        expected_weights_tmp = np.zeros_like(expected_weights)
        if verbose:
            print('Computing weights')
            print('Number of passes: ', npass)

        for p in range(npass):
            # if verbose:
                # print('Pass: ', p)


            if ((net_width) * (p+1)) <= nvis:
                curr_residual = residual[net_width*p : (net_width) * (p+1), :]
                res = np.linalg.norm(curr_residual[...,np.newaxis], axis=-1)**2
                curr_weights = expected_weights[net_width*p : (net_width) * (p+1)]
            
            else:
                raise NotImplementedError('Support number of visibilities not multiple of net_width not implemented yet')
                # curr_residual = residual[net_width*p : (net_width) * (p+1), :]
                # res = np.concatenate(curr_residual, np.zeros((1,nvis-p)))
                # res = res.reshape(-1,1).astype(np.float32)**2
                # curr_weights = np.concatenate(curr_weights[net_width*p : (net_width ) * (p+1)], np.zeros(nvis-p))
                # curr_weights = curr_weights.reshape(-1,1).astype(np.float32)

            for f in range(nfreq):
                net_output = net.forward(res[:,f], curr_weights[:,f])
                expected_weights_tmp[net_width*p : (net_width) * (p+1), f] = net_output

        expected_weights = deepcopy(expected_weights_tmp)

        # print information about the weights
        if verbose:
            print('Weights statistics')
            print('Min: ', np.min(expected_weights))
            print('Max: ', np.max(expected_weights))
            print('Mean: ', np.mean(expected_weights))
            print('Std: ', np.std(expected_weights))



        if verbose:
            print(f'iteration {it} - E step done')

        for mit in range(miter):

            if verbose:
                print(f"iteration {it} - M step {mit}/ {miter}")

            if verbose:
                print('Computing model visibilities')

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
            residual = np.multiply(expected_weights, residual)

            if verbose:
                print('Computing residual image')

            residual_image = ms2dirty(  
                                    uvw = uvw,
                                    freq = freq,
                                    ms = residual,
                                    npix_x = npix_x,
                                    npix_y = npix_y,
                                    pixsize_x = cellsize,
                                    pixsize_y = cellsize,
                                    epsilon=1.0e-7 
                            )/nvis/(npix_x*npix_y)
            
            if verbose:
                print('Updating model image')

            model_image_k = model_image_k + mstep_size*residual_image
            model_image_k = np.sign(model_image_k) * np.max([np.abs(model_image_k)- threshold*np.max(np.abs(model_image_k)), np.zeros(model_image_k.shape)], axis=0)
            model_image_k = model_image_k

            if verbose:
                print(f"iteration {it} - M step {mit}/ {miter} done")

        if plot:
            plt.imshow(model_image_k)
            plt.show()
            
    return np.abs(model_image_k)
