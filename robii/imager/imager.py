from ..astro.ms import MS

import matplotlib.pyplot as plt
import numpy as np

from ducc0.wgridder import ms2dirty, dirty2ms
from ..astro.gridder import grid, degrid
from .em_imager import em_imager, ista, fftem_imager
from .clean_from_vis import clean_from_vis
from .unrolled import unrolled_imager
from ..deep.models import forward_operator

from ..arl.clean import deconvolve_cube

from astropy.io import fits
import os

class Imager():

    def __init__(self, vis, freq, uvw, cellsize, npix_x, npix_y, wgt=None, verbose=True):

        self.cellsize = cellsize
        self.npix_x = npix_x
        self.npix_y = npix_y

        self.uvw = uvw
        self.nvis = len(self.uvw)
        
        self.freq = freq
        self.vis = vis

        self.model_image = None
        self.model_vis = None
        self.residual = None
        self.residual_image = None

        self.wgt = wgt

        # print info about the data
        if verbose:
            print('Data info:')
            print('Number of visibilities: ', self.nvis)
            print('Number of channels: ', len(self.freq))
            print('Number of pixels: ', self.npix_x, self.npix_y)
            print('Cellsize: ', self.cellsize)
            print('Image size (degree): ', self.npix_x*self.cellsize, self.npix_y*self.cellsize)

    
    @classmethod
    def from_ms(cls, ms, cellsize, npix_x, npix_y, spw_id=1, corr_type='RR-LL'):

        ms = MS(ms)

        cellsize = cellsize / 3600 * np.pi / 180
        print(ms.chan_freq.shape)
        print(ms.vis_data.shape)

        vis = ms.vis_data
        wgt = ms.weight #[ms.data_desc_id == spw_id]

        # extend the weight to the number of channels
        wgt = np.repeat(wgt.reshape(-1,2, 1), ms.nb_chan, axis=-1)

        if corr_type == 'RR-LL':
            stokeI_vis = (vis[:, :, 0]*wgt[:,0] + vis[:, :, 1]*wgt[:,1])/2
            # stokeI_vis = (vis[:, :, 0] + vis[:, :, 1])/2


        stokeI_vis = stokeI_vis[ms.data_desc_id == spw_id]
        uvw = ms.uvw[ms.data_desc_id == spw_id]
        freq = ms.chan_freq[spw_id, :]

        return cls(stokeI_vis, freq, uvw, cellsize, npix_x, npix_y)
    

    def make_dirty(self, cellsize=None, npix_x=None, npix_y=None, plot=False):

        if cellsize is None:
            cellsize = self.cellsize
        if npix_x is None:
            npix_x = self.npix_x
        if npix_y is None:
            npix_y = self.npix_y

        # print shape of the inputs
        print('Computing dirty image')
        print(f'freq: {self.freq.shape}')
        print(f'uvw: {self.uvw.shape}')
        print(f'vis: {self.vis.shape}')
        print(f'npix_x: {npix_x}')
        print(f'npix_y: {npix_y}')
        print(f'cellsize: {cellsize}')
        
        
        wgt = self.wgt if self.wgt is not None else np.ones(self.nvis)


        self.dirty = ms2dirty(  
                            uvw = self.uvw,
                            freq = self.freq,
                            ms = self.vis.astype(np.complex64),
                            # do_wstacking = True,
                            npix_x = npix_x,
                            npix_y = npix_y,
                            pixsize_x = cellsize,
                            pixsize_y = cellsize,
                            epsilon=1.0e-5 
        )
        
        if plot:
            plt.figure(figsize=(8,8))
            plt.imshow(self.dirty, origin='lower', cmap='Spectral_r')
            plt.colorbar()
            plt.title('Dirty image')
            plt.show()


        return self.dirty
    

    def make_image(self, init=None, niter=10, dof=10, cellsize=None, npix_x=None, npix_y=None, method='dirty', useducc=True, params=None):

        if cellsize is None:
            cellsize = self.cellsize
        if npix_x is None:
            npix_x = self.npix_x
        if npix_y is None:
            npix_y = self.npix_y


        if useducc:
            adjoint = lambda vis : ms2dirty(  
                            uvw = self.uvw,
                            freq = self.freq,
                            ms = vis.reshape(-1,len(self.freq)),
                            npix_x = npix_x,
                            npix_y = npix_y,
                            pixsize_x = cellsize,
                            pixsize_y = cellsize,
                            epsilon=1.0e-5 
                    )
            
            forward = lambda x : dirty2ms(  
                        uvw = self.uvw,
                        freq = self.freq,
                        dirty = x,
                        pixsize_x = cellsize,
                        pixsize_y = cellsize,
                        epsilon=1.0e-5 
                        ).reshape(-1)
            
        else:

                H = forward_operator(self.uvw, self.freq, npix_x, cellsize=cellsize)

                adjoint = lambda vis : H.T.conj().dot(vis.flatten()).reshape(int(np.sqrt(H.shape[1])), int(np.sqrt(H.shape[1])))
                forward = lambda x : H.dot(x.flatten()).reshape(-1)
            
        ops = (forward, adjoint)


        if method == 'dirty':
            self.image = self.make_dirty(cellsize, npix_x, npix_y)
        # elif method == 'robiinet':
        #     self.image = unrolled_imager(vis=self.vis, 
        #                                  freq=self.freq, 
        #                                  uvw=self.uvw, 
        #                                  cellsize=cellsize, 
        #                                  npix_x=npix_x, 
        #                                  npix_y=npix_y, 
        #                                  **kwargs)
            # self.image = np.max(0, self.image)

        elif method == 'em-ista':


            self.image = em_imager(vis=self.vis,
                                   ops=ops,
                                   niter=niter,
                                   dof= dof,
                                   mstep_solver=ista,
                                   params=params,
                                   init=init)
            
        elif method == 'em-clean':
            self.image = em_imager(vis=self.vis,
                                   ops=ops,
                                   niter=niter,
                                   dof= dof,
                                   mstep_solver=clean_from_vis,
                                   params=params,
                                   init=init)

        elif method == 'fft-em-ista':

            _grid = lambda x : grid(x, self.uvw, self.freq, npix_x, npix_y, cellsize=cellsize)
            _degrid = lambda x : degrid(x, self.uvw, self.freq, cellsize=cellsize)

            gridder = (_grid, _degrid)

            try:
                sigmae2 = params['sigmae2'] 
                del params['sigmae2']
            except:
                raise ValueError('sigmae2 not provided')

            self.image = fftem_imager(vis=self.vis,
                          gridder=gridder,
                          sigmae2=sigmae2,
                          niter=niter,
                          dof= dof,
                          mstep_solver=ista,
                          params=params,
                          init=init).real
            

        # elif method == 'clean':

        # elif method == 'ista':
            # self.image = em_imager(vis=self.vis, 
                                        #   freq=self.freq, 
                                        #   uvw=self.uvw, 
                                        #   cellsize=cellsize, 
                                        #   npix_x=npix_x, 
                                        #   npix_y=npix_y, 
                                        #   **kwargs)
            # self.image = np.max(0, self.image)
        else:
            raise ValueError('Method not implemented')
        
        # self.image = self.image.T
        # #invert y axis
        # self.image = self.image[::-1, :]

        return self.image
    
    def save_image(self, filename, save_fits=False, overwrite=True):

        if save_fits:
            hdu = fits.PrimaryHDU(self.image)
            if os.path.exists(filename):
                if overwrite:
                    hdu.writeto(filename, overwrite=True)
                else:
                    raise ValueError('File already exists')
            else:
                hdu.writeto(filename, overwrite=False)

        else:
            plt.imsave(filename, self.image, origin='lower', cmap='Spectral_r')

    def plot_uv(self, ax=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8,8))

        ax.scatter(self.uvw[:,0], self.uvw[:,1], **kwargs)

        return ax


    




def convolve_with_gaussian(image, sigma,shape=None):
    """
    Convolve an image with a Gaussian kernel.
    """
    from scipy.ndimage import gaussian_filter
    if len(image.shape) == 2:
        return gaussian_filter(image, sigma)
    elif len(image.shape) == 1:
        return gaussian_filter(image.reshape(shape), sigma).reshape(image.shape)
