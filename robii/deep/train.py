import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from .models import RobiiNet, RobiiNetV2, forward_operator
from .datasets import ViDataset
from ..math.metrics import psnr, ssim, nmse, snr, normalized_cross_correlation

from ..imager.imager import Imager

import os
import pandas as pd
import numpy as np
np.seterr(over='ignore')


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

def logprint(msg, path):
    with open(path, 'a') as file:
        print(msg, file=file)
    return True



def save_model(filename, nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out):
    return torch.save({
                    'epoch': nepoch,
                    'net_width': model.net_width,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataset': dset_path,
                    'model name': model_name,
                    'loss': loss,
                    'uvw' : uvw,
                    'freq' : freq
                    }
                    , f'{out}/{filename}.pth')



def test_on_sample_dataset(ndata, dataset, model_path):

    # generate ndata unique random indices to test on   
    if ndata == -1:
        ndata = dataset.nimages
    idx = np.arange(ndata)
    idx = np.random.choice(idx, ndata, replace=False)

    df = pd.DataFrame(columns=['snr', 'nmse', 'ssim', 'ncc'])
    for i in range(ndata):
        vis, true_image = dataset[idx[i]]
        
        vis = np.cdouble(vis)
        if np.sum(true_image) == 0:
            raise ValueError("Sky image is empty 1")
        uvw = dataset.uvw
        freq = dataset.freq

        npixel = dataset.npixel
        true_image = true_image.reshape(npixel, npixel)

        cellsize = dataset.cellsize

        imager = Imager(vis.reshape(-1,1), uvw=uvw, freq=freq, cellsize=cellsize, npix_x=npixel, npix_y=npixel, verbose=False)
        estimated_image = imager.make_image(method='robiinet', 
                                            niter=10, 
                                            miter=50, 
                                            mstep_size=1, 
                                            threshold=0.001, 
                                            model_path=model_path, 
                                            verbose=False, 
                                            plot=False)
        # if np.sum(estimated_image[i]) == 0:
            # raise ValueError("empty image 1")
        snr_ = snr(estimated_image, true_image)
        nmse_ = nmse(estimated_image, true_image)
        ssim_ = ssim(estimated_image, true_image)
        ncc_ = normalized_cross_correlation(estimated_image, true_image)


        df.loc[i] = [snr_, nmse_, ssim_, ncc_]
    

    return df



def train(dset_path, nepoch, batch_size, net_depth,
           net_width, learning_rate, step, out, model_name,
             logpath, threshold=1e-3, mstep_size=1, SNR=10, true_init=False, monitor=True):
        
    dataset = ViDataset(dset_path)
    npixel = dataset.npixel
    uvw = dataset.uvw
    nvis = uvw.shape[0]
    freq = dataset.freq
    cellsize = dataset.cellsize
    ndata = dataset.nimages
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    print(f"Dataset loaded: {dset_path}")
    logprint(f"Dataset loaded: {dset_path}", path=logpath)


    print(f"Number training samples: {ndata}")
    logprint(f"Number training samples: {ndata}", path=logpath)
    print(f"Number of pixels: {npixel}")
    logprint(f"Number of pixels: {npixel}", path=logpath)
    print(f"Number of visibilities: {nvis}")
    logprint(f"Number of visibilities: {nvis}", path=logpath)
    print(f'Cellsize: {cellsize}')
    logprint(f'Cellsize: {cellsize}', path=logpath)
    print(f'UVW shape: {uvw.shape}')
    print(f"Number of frequencies: {freq.shape[0]}")
    logprint(f"Number of frequencies: {freq.shape[0]}", path=logpath)
    print(f"Number of epochs: {nepoch}")
    logprint(f"Number of epochs: {nepoch}", path=logpath)
    print(f"Batch size: {batch_size}")
    logprint(f"Batch size: {batch_size}", path=logpath)
    print(f"Network depth: {net_depth}")
    logprint(f"Network depth: {net_depth}", path=logpath)
    print(f"Network width: {net_width}")
    logprint(f"Network width: {net_width}", path=logpath)
    print(f"Learning rate: {learning_rate}")
    logprint(f"Learning rate: {learning_rate}", path=logpath)
    print(f"Step: {step}")
    logprint(f"Step: {step}", path=logpath)
    print(f"Output path: {out}")
    logprint(f"Output path: {out}", path=logpath)
    print(f"Model name: {model_name}")
    logprint(f"Model name: {model_name}", path=logpath)
    print(f"True init: {true_init}")
    logprint(f"True init: {true_init}", path=logpath)


    
    model = RobiiNetV2(net_width)
    optimizer = torch.optim.Adam(model.trainable_robust, lr=learning_rate)
    loss_fn = nn.MSELoss()

    H = forward_operator(uvw=uvw, freq=freq, npixel=npixel)
    H = ToTensor()(H)
    for epoch in range(nepoch):
                    
        print(f"Epoch {epoch+1}\n-------------------------------")
        logprint(f"Epoch {epoch+1}\n-------------------------------", path=logpath)
        loss = model.train_supervised(train_dataloader, loss_fn, optimizer, device,
                                       true_init=true_init,
                                       threshold=threshold,
                                       mstep_size=mstep_size,
                                       niter=net_depth,
                                       H=H,
                                       SNR=SNR)
        
        logprint(f"loss = {loss}", path=logpath)
        save_model(f"{model_name}_tmp", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

        # if monitor or not (epoch+1) % step:
        #     # compute estimated image on a sample of the dataset using the instanciated model and the train_dataloader


        #     snr_ = np.zeros(ndata)
        #     nmse_ = np.zeros(ndata)
        #     ssim_ = np.zeros(ndata)
        #     ncc_ = np.zeros(ndata)
        #     estimated_image = np.zeros((ndata,npixel, npixel))
        #     for ii,data in enumerate(dataset):

        #         vis, true_image = data
        #         true_image = true_image.reshape(npixel, npixel)

        #         pred = model(ToTensor()(vis.reshape(1,-1,1)), H=H, 
        #                     threshold=0.001, niter=net_depth, x0=None)
                
        #         estimated_image[ii] = pred.detach().numpy().reshape(npixel, npixel)

        #         # compute metrics on the sample of the datase
        #         snr_[ii] = snr(estimated_image[ii], true_image)
        #         nmse_[ii] = nmse(estimated_image[ii], true_image)
        #         ssim_[ii] = ssim(estimated_image[ii], true_image)
        #         ncc_[ii] = normalized_cross_correlation(estimated_image[ii], true_image)

            
               
                
        #     logprint(f"snr: {snr_.mean()} +/- {snr_.std()}", path=logpath)
        #     logprint(f"nmse: {nmse_.mean()} +/- {nmse_.std()}", path=logpath)
        #     logprint(f"ssim: {ssim_.mean()} +/- {ssim_.std()}", path=logpath)
        #     logprint(f"ncc: {ncc_.mean()} +/- {ncc_.std()}", path=logpath)
            

    
            
            # df = test_on_sample_dataset(-1, dataset, f"{out}/{model_name}_tmp.pth")

            # logprint(f"snr: {df['snr'].mean()} +/- {df['snr'].std()}", path=logpath)
            # logprint(f"nmse: {df['nmse'].mean()} +/- {df['nmse'].std()}", path=logpath)
            # logprint(f"ssim: {df['ssim'].mean()} +/- {df['ssim'].std()}", path=logpath)
            # logprint(f"ncc: {df['ncc'].mean()} +/- {df['ncc'].std()}", path=logpath)     





        if not (epoch+1) % step:
            save_model(f"{model_name}_ep-{epoch}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

    save_model(f"{model_name}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)




"""
OLD VERSION OF FUNCTION TO TRAIN ROBIINET


def train(dset_path, nepoch, batch_size, net_depth, learning_rate, step, out, model_name, logpath, true_init=False, monitor=True):
        
    dataset = ViDataset(dset_path)
    npixel = dataset.npixel
    uvw = dataset.uvw
    nvis = uvw.shape[0]
    freq = dataset.freq
    cellsize = dataset.cellsize
    ndata = dataset.nimages
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    
    print(f"Dataset loaded: {dset_path}")
    logprint(f"Dataset loaded: {dset_path}", path=logpath)


    print(f"Number training samples: {ndata}")
    logprint(f"Number training samples: {ndata}", path=logpath)
    print(f"Number of pixels: {npixel}")
    logprint(f"Number of pixels: {npixel}", path=logpath)
    print(f"Number of visibilities: {nvis}")
    logprint(f"Number of visibilities: {nvis}", path=logpath)
    print(f'Cellsize: {cellsize}')
    logprint(f'Cellsize: {cellsize}', path=logpath)
    print(f'UVW shape: {uvw.shape}')
    print(f"Number of frequencies: {freq.shape[0]}")
    logprint(f"Number of frequencies: {freq.shape[0]}", path=logpath)
    print(f"Number of epochs: {nepoch}")
    logprint(f"Number of epochs: {nepoch}", path=logpath)
    print(f"Batch size: {batch_size}")
    logprint(f"Batch size: {batch_size}", path=logpath)
    print(f"Network depth: {net_depth}")
    logprint(f"Network depth: {net_depth}", path=logpath)
    print(f"Learning rate: {learning_rate}")
    logprint(f"Learning rate: {learning_rate}", path=logpath)
    print(f"Step: {step}")
    logprint(f"Step: {step}", path=logpath)
    print(f"Output path: {out}")
    logprint(f"Output path: {out}", path=logpath)
    print(f"Model name: {model_name}")
    logprint(f"Model name: {model_name}", path=logpath)
    print(f"True init: {true_init}")
    logprint(f"True init: {true_init}", path=logpath)


    
    model = RobiiNet(net_depth, npixel, uvw, freq)
    optimizer = torch.optim.Adam(model.trainable_robust, lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(nepoch):
                    
        print(f"Epoch {epoch+1}\n-------------------------------")
        logprint(f"Epoch {epoch+1}\n-------------------------------", path=logpath)
        loss = model.train_supervised(train_dataloader, loss_fn, optimizer, device, true_init=true_init)
        logprint(f"loss = {loss}", path=logpath)
        save_model(f"{model_name}_tmp", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

        if monitor or not (epoch+1) % step:
            # compute estimated image on a sample of the dataset using the instanciated model and the train_dataloader
            df = test_on_sample_dataset(-1, dataset, f"{out}/{model_name}_tmp.pth")


            snr_ = np.zeros(ndata)
            nmse_ = np.zeros(ndata)
            ssim_ = np.zeros(ndata)
            ncc_ = np.zeros(ndata)
            estimated_image = np.zeros((ndata,npixel, npixel))
            for ii,data in enumerate(dataset):

                vis, true_image = data
                true_image = true_image.reshape(npixel, npixel)

                pred = model(ToTensor()(vis.reshape(1,-1,1)))
                estimated_image[ii] = pred.detach().numpy().reshape(npixel, npixel)

            # compute metrics on the sample of the dataset


                if np.sum(true_image) == 0:
                    raise ValueError("Sky image is empty 1")
                
                

                snr_[ii] = snr(estimated_image[ii], true_image)
                nmse_[ii] = nmse(estimated_image[ii], true_image)
                ssim_[ii] = ssim(estimated_image[ii], true_image)
                ncc_[ii] = normalized_cross_correlation(estimated_image[ii], true_image)

            
                # if np.sum(estimated_image[ii]) == 0:
                    # raise ValueError("empty image")
                
            logprint(f"snr: {snr_.mean()} +/- {snr_.std()}", path=logpath)
            logprint(f"nmse: {nmse_.mean()} +/- {nmse_.std()}", path=logpath)
            logprint(f"ssim: {ssim_.mean()} +/- {ssim_.std()}", path=logpath)
            logprint(f"ncc: {ncc_.mean()} +/- {ncc_.std()}", path=logpath)
            

    
            

            logprint(f"snr: {df['snr'].mean()} +/- {df['snr'].std()}", path=logpath)
            logprint(f"nmse: {df['nmse'].mean()} +/- {df['nmse'].std()}", path=logpath)
            logprint(f"ssim: {df['ssim'].mean()} +/- {df['ssim'].std()}", path=logpath)
            logprint(f"ncc: {df['ncc'].mean()} +/- {df['ncc'].std()}", path=logpath)     





        if not (epoch+1) % step:
            save_model(f"{model_name}_ep-{epoch}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

    save_model(f"{model_name}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

"""