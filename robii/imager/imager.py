from ..astro.ms import MS

import matplotlib.pyplot as plt
import numpy as np

from ducc0.wgridder import ms2dirty, dirty2ms
from .robust_em import robust_em_imager
from .unrolled import unrolled_imager

from astropy.io import fits
import os

class Imager():

    def __init__(self, vis, freq, uvw, cellsize, npix_x, npix_y, verbose=True):

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

        if corr_type == 'RR-LL':
            stokeI_vis = (vis[:, :, 0] + vis[:, :, 1])/2
            

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
        
        

        self.dirty = ms2dirty(  
                            uvw = self.uvw,
                            freq = self.freq,
                            ms = self.vis,
                            npix_x = npix_x,
                            npix_y = npix_y,
                            pixsize_x = cellsize,
                            pixsize_y = cellsize,
                            epsilon=1.0e-5 
                    )/self.nvis 
        
        if plot:
            plt.figure(figsize=(8,8))
            plt.imshow(self.dirty, origin='lower', cmap='Spectral_r')
            plt.colorbar()
            plt.title('Dirty image')
            plt.show()


        return self.dirty
    

    def make_image(self, cellsize=None, npix_x=None, npix_y=None, method='dirty', **kwargs):

        if cellsize is None:
            cellsize = self.cellsize
        if npix_x is None:
            npix_x = self.npix_x
        if npix_y is None:
            npix_y = self.npix_y

        if method == 'dirty':
            self.image = self.make_dirty(cellsize, npix_x, npix_y)
        elif method == 'robiinet':
            self.image = unrolled_imager(vis=self.vis, 
                                         freq=self.freq, 
                                         uvw=self.uvw, 
                                         cellsize=cellsize, 
                                         npix_x=npix_x, 
                                         npix_y=npix_y, 
                                         **kwargs)
            # self.image = np.max(0, self.image)

        elif method == 'robii':
            self.image = robust_em_imager(vis=self.vis, 
                                          freq=self.freq, 
                                          uvw=self.uvw, 
                                          cellsize=cellsize, 
                                          npix_x=npix_x, 
                                          npix_y=npix_y, 
                                          **kwargs)
            # self.image = np.max(0, self.image)
        else:
            raise ValueError('Method not implemented')
        

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
