import torch
import torch.nn as nn

from ..imager.imager import Imager

from ..math.metrics import snr, psnr, ssim, normalized_cross_correlation

from ..simulation.simulation import ViSim

import pandas as pd

def test(dataset_path, model_path, mstep_size, miter, niter, threshold, out, name=None):

    # read dataset
    visim = ViSim.from_zarr(dataset_path)

    # create imager

    df = pd.DataFrame(columns=['SNR', 'PSNR', 'SSIM', 'NCC'])

    for i in range(visim.ndata):
        imager = Imager(visim.vis[i], 
                    cellsize=visim.cellsize, 
                    npix_x=visim.npixel, 
                    npix_y=visim.npixel,
                    uvw=visim.uvw,
                    freq=visim.freq)   
        
        # make image
        imager.make_image(method='robiinet',
                            mstep_size=mstep_size,
                            miter=miter,
                            niter=niter,
                            threshold=threshold,
                            model_path=model_path,
                            verbose=True,
                            plot=False)
        
        # calculate metrics
        snr_ = snr(visim.model_images[i], imager.image)
        psnr_ = psnr(visim.model_images[i], imager.image)
        ssim_ = ssim(visim.model_images[i], imager.image)
        ncc_ = normalized_cross_correlation(visim.model_images[i], imager.image)

        print(f'SNR: {snr_}')
        print(f'PSNR: {psnr_}')
        print(f'SSIM: {ssim_}')
        print(f'NCC: {ncc_}')

        # save metrics
        df.loc[i] = [snr_, psnr_, ssim_, ncc_]

    if name is None:
        df.to_csv(f'{out}/metrics.csv')
    else:
        df.to_csv(f'{out}/metrics_{name}.csv')

    return df


        