import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .models import RobiiNet
from .datasets import ViDataset
from ..math.metrics import psnr, ssim, nmse, snr, normalized_cross_correlation

import click
from omegaconf import OmegaConf

import os
import pandas as pd

import shutil

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
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataset': dset_path,
                    'model name': model_name,
                    'loss': loss,
                    'uvw' : uvw,
                    'freq' : freq
                    }
                    , f'{out}/{filename}.pth')


def load_dataset(dset_path, batch_size):
    dataset = ViDataset(dset_path)
    npixel = dataset.npixel
    uvw = dataset.uvw
    nvis = uvw.shape[0]
    freq = dataset.freq
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    return train_dataloader, npixel, uvw, nvis, freq


def train(dset_path, nepoch, batch_size, net_depth, learning_rate, step, out, model_name, logpath, true_init=False):
        
    train_dataloader, npixel, uvw, nvis, freq = load_dataset(dset_path, batch_size)
    print(f"Dataset loaded: {dset_path}")
    logprint(f"Dataset loaded: {dset_path}", path=logpath)
    print(f"Number of pixels: {npixel}")
    logprint(f"Number of pixels: {npixel}", path=logpath)
    print(f"Number of visibilities: {nvis}")
    logprint(f"Number of visibilities: {nvis}", path=logpath)
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

    
    model = RobiiNet(net_depth, npixel, uvw, freq)
    optimizer = torch.optim.Adam(model.trainable_robust, lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(nepoch):
                    
        print(f"Epoch {epoch+1}\n-------------------------------")
        logprint(f"Epoch {epoch+1}\n-------------------------------", path=logpath)
        loss = model.train_supervised(train_dataloader, loss_fn, optimizer, device, true_init=true_init)
        logprint(f"loss = {loss}", path=logpath)
        save_model(f"{model_name}_tmp", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

        if not (epoch+1) % step:
            save_model(f"{model_name}_ep-{epoch}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

    save_model(f"{model_name}", nepoch, model, optimizer, dset_path, model_name, loss, uvw, freq, out)

