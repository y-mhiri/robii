"""
click commands for making images

"""
import click
from omegaconf import OmegaConf


import os
import pandas as pd

from ..imager.imager import Imager
from ..simulation.simulation import ViSim


import numpy as np
import matplotlib.pyplot as plt


# group of commands to make images
@click.group()
def robii():
    """
    Make images from measurement sets
    """
    pass


# command to make an image from a measurement set
@robii.command()
@click.argument('mspath', type=click.Path(exists=True))
@click.option('--out', '-o', default='.', help='output directory')
@click.option('--image_size', '-s', default=256, help='image size')
@click.option('--cellsize', '-c', default=8.0, help='cellsize in arcseconds')
@click.option('--nchan', '-n', default=1, help='number of channels')
@click.option('--niter', '-i', default=10, help='number of iterations')
@click.option('--miter', '-t', default=1, help='number of iterations for m step')
@click.option('--mstep_size', '-m', default=1.0, help='step size for m step')
@click.option('--threshold', '-t', default=0.1, help='threshold')
@click.option('--dof', '-d', default=1.0, help='degrees of freedom')
@click.option('--fits/--no_fits', '-f', default=False, help='save as fits')
@click.option('--plot/--no-plot', '-p', default=False, help='plot')
def fromms(mspath, out, fits, image_size, cellsize, nchan, niter, threshold, dof, miter, mstep_size, plot):
    """
    Make an image from a measurement set
    """

    # read ms and extractuvw and freq

    # convert cellsize to radians 

    npix_x, npix_y = image_size, image_size
    imager = Imager.from_ms(mspath, cellsize=cellsize, npix_x=npix_x, npix_y=npix_y)
    
    kwargs = {
        'niter': niter,
        'miter': miter,
        'mstep_size': mstep_size,
        'threshold': threshold,
        'dof': dof,
        'gaussian': False,
        'verbose': True,
        'plot': plot,
    }


    imager.make_image(method='robii', **kwargs)

    ext = 'fits' if fits else 'png'
    imager.save_image(f'{out}.{ext}', save_fits=fits)

    return True


# command to make an image from a zarr dataset and specifying the number of images
@robii.command()
@click.argument('zarr', type=click.Path(exists=True))
@click.option('--nimages', '-n', default=10, help='number of images to show')
@click.option('--idx', '-i', default=-1, help='index of image to show')
@click.option('--out', '-o', default='.', help='output directory')
@click.option('--fits/--no_fits', '-f', default=False, help='save as fits')
@click.option('--image_size', '-s', default=256, help='image size')
@click.option('--cellsize', '-c', default=None, help='cellsize in arcseconds')
@click.option('--niter', '-i', default=1000, help='number of iterations')
@click.option('--miter', '-t', default=1, help='number of iterations for m step')
@click.option('--mstep_size', '-m', default=1.0, help='step size for m step')
@click.option('--threshold', '-a', default=0.1, help='threshold')
@click.option('--dof', '-d', default=1, help='degrees of freedom')
def fromzarr(zarr, nimages, idx, out, fits, image_size, cellsize, niter, miter, mstep_size, threshold, dof):

    # read ViSim object from zarr
    visim = ViSim.from_zarr(zarr)

    # if cellsize is not specified, use the cellsize from the zarr
    if cellsize is None:
        cellsize = visim.cellsize
    else:
        cellsize = cellsize * np.pi / 180 / 3600
    
    npix_x, npix_y = image_size, image_size

    if idx == -1:
        idx = np.random.randint(0, visim.ndata, nimages)
    else:
        idx = [idx + i for i in range(nimages)]

    for i in idx:
        imager = Imager(visim.vis[i], cellsize=cellsize, npix_x=npix_x, npix_y=npix_y, freq=visim.freq, uvw=visim.uvw)
        dirty_image = imager.make_image(method='dirty')

        estimated_image = imager.make_image(method='robii', 
                                            niter=niter, 
                                            miter=miter, 
                                            mstep_size=mstep_size, 
                                            threshold=threshold, 
                                            dof=dof, 
                                            gaussian=False, 
                                            verbose=True, 
                                            plot=False)
        
        # make a directory to save the images
        if not os.path.exists(out):
            os.mkdir(out)

        if fits:
            imager.save_image(out, save_fits=True)

        ## show true and estimated image and save 
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(visim.model_images[i])
        plt.title('True Image')
        plt.subplot(1, 3, 2)
        plt.imshow(estimated_image)
        plt.title('Estimated Image')
        plt.subplot(1, 3, 3)
        plt.imshow(dirty_image)
        plt.title('Dirty Image')
        plt.savefig(os.path.join(out, f'image_{i}.png'))






## The next command group, named robiinet, is the same but uses unrolled_imager to image the data, the method key argument is changed to 'robiinet'
## The rest of the arguments are the same as the previous command 

@click.group()
def robiinet():
    """
    Make images from measurement sets
    """
    pass



@robiinet.command()
@click.argument('mspath', type=click.Path(exists=True))
@click.option('--out', '-o', default='.', help='output directory')
@click.option('--image_size', '-s', default=256, help='image size')
@click.option('--cellsize', '-c', default=8.0, help='cellsize in arcseconds')
@click.option('--niter', '-i', default=10, help='number of iterations')
@click.option('--miter', '-t', default=1, help='number of iterations for m step')
@click.option('--mstep_size', '-m', default=1.0, help='step size for m step')
@click.option('--threshold', '-t', default=0.001, help='threshold')
@click.option('--model_path', '-m', default=None, help='path to model pth')
@click.option('--fits/--no_fits', '-f', default=False, help='save as fits')
def fromms(mspath, out, fits, image_size, cellsize, niter, threshold, model_path, miter, mstep_size):
    """
    Make an image from a measurement set
    """

    # read ms and extractuvw and freq

    # convert cellsize to radians 
    cellsize = cellsize * np.pi / 180 / 3600

    npix_x, npix_y = image_size, image_size
    imager = Imager.from_ms(mspath, cellsize=cellsize, npix_x=npix_x, npix_y=npix_y)
    
    kwargs = {
        'niter': niter,
        'miter': miter,
        'mstep_size': mstep_size,
        'threshold': threshold,
        'model_path': model_path,
        'verbose': True,
        'plot': False,
    }


    imager.make_image(method='robiinet', **kwargs)
    imager.save_image(f'{out}/output.fits', save_fits=fits)

    return True


# command to make an image from a zarr dataset and specifying the number of images
@robiinet.command()
@click.argument('zarr', type=click.Path(exists=True))
@click.option('--nimages', '-n', default=1, help='number of images to show')
@click.option('--idx', '-i', default=-1, help='index of image to show')
@click.option('--out', '-o', default='.', help='output directory')
@click.option('--fits/--no_fits', '-f', default=False, help='save as fits')
@click.option('--image_size', '-s', default=None, help='image size')
@click.option('--cellsize', '-c', default=None, help='cellsize in arcseconds')
@click.option('--niter', '-i', default=10, help='number of iterations')
@click.option('--miter', '-t', default=1, help='number of iterations for m step')
@click.option('--mstep_size', '-m', default=1.0, help='step size for m step')
@click.option('--threshold', '-t', default=0.1, help='threshold')
@click.option('--model_path', '-p', default=None, help=' path to torch model')
def fromzarr(zarr, nimages, idx, out, fits, image_size, cellsize, niter, threshold, model_path, miter, mstep_size):
    
        # read ViSim object from zarr
        visim = ViSim.from_zarr(zarr)
    
        # if cellsize is not specified, use the cellsize from the zarr
        if cellsize is None:
            cellsize = visim.cellsize
        else:
            cellsize = cellsize * np.pi / 180 / 3600
        
        if image_size is None:
            image_size = visim.npixel

        npix_x, npix_y = image_size, image_size
    
        if idx == -1:
            idx = np.random.randint(0, visim.ndata, nimages)
        else:
            idx = [idx + i for i in range(nimages)]
    


        for i in idx:
            imager = Imager(visim.vis[i], 
                            cellsize=cellsize, 
                            npix_x=npix_x, 
                            npix_y=npix_y,
                            uvw=visim.uvw,
                            freq=visim.freq)
            
            estimated_image = imager.make_image(method='robiinet', 
                                                niter=niter, 
                                                miter=miter, 
                                                mstep_size=mstep_size, 
                                                threshold=threshold, 
                                                model_path=model_path, 
                                                verbose=True, 
                                                plot=False)
    
    
            if fits:
                imager.save_image(f'{out}/image_{i}.fits', save_fits=fits)
            ## show true and estimated image and save 
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(visim.model_images[i])
            plt.title('True Image')
            plt.subplot(1, 2, 2)
            plt.imshow(estimated_image)
            plt.title('Estimated Image')
            plt.savefig(os.path.join(out, f'image_{i}.png'))
    
        return True


if __name__ == '__main__':
    robiinet()
    robii()

