import numpy as np
from ..astro.astro import generate_directions
from scipy.constants import speed_of_light

import torch

from torch import nn

from copy import deepcopy


"""
Pytorch Models and operators


"""


def forward_operator(uvw, freq, npixel, cellsize=None):
    """
    Used to initialize the model matrix

    """
    wl = speed_of_light / freq
    if cellsize is None:
        cellsize = np.min(wl)/(2*np.max(uvw))

    # cellsize = np.rad2deg(cellsize)
    # fov = cellsize*npixel
    lmn = generate_directions(npixel, cellsize).reshape(3,-1)
    uvw = uvw.reshape(-1,3)
    # add conjugate
    uvw = np.concatenate((uvw, -uvw), axis=0)


    return np.exp(-1j*(2*np.pi/np.min(wl))* uvw @ lmn)


class RobiiNet(nn.Module):
    def __init__(self, depth, npixel, uvw, freq, alpha=None):
        super().__init__()
        self.depth = depth
        self.npixel = npixel

        H = forward_operator(uvw, freq, npixel)
        nvis, npix2 = H.shape
        self.nvis = nvis 


        # layers that should not be trained
        self.W = nn.Linear(npix2,  nvis).to(torch.cdouble)
        self.Wt = lambda a : torch.matmul(a, self.W.weight.conj()/npix2)

        if alpha is not None:
            self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=False))
        else:
            self.alpha = nn.Parameter(torch.tensor(0.001, requires_grad=False))
            
        self.softthresh = nn.ReLU()
          
        Wreal = torch.tensor(H.real)
        Wimag = torch.tensor(H.imag)
        self.W.weight = torch.nn.Parameter(torch.complex(Wreal, Wimag))
        self.W.bias =  torch.nn.Parameter(torch.zeros_like(self.W.bias))

        self.model_parameters = []
        self.model_parameters.append(self.W.weight)
        self.model_parameters.append(self.alpha)


        # layers that should be trained
        self.estep = nn.Linear(self.nvis, self.nvis)
        self.update_layer = nn.Linear(self.nvis, self.nvis)
        self.memory_layer = nn.Linear(self.nvis, self.nvis)
        self.trainable_robust = []
        for param in self.estep.parameters():
            self.trainable_robust.append(param)
        for param in self.update_layer.parameters():
            self.trainable_robust.append(param)
        for param in self.memory_layer.parameters():
            self.trainable_robust.append(param)


    def set_threshold(self, alpha):
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=False))
        return True
    


    def compute_loss(self, dataloader, loss_fn, device):

        size = len(dataloader.dataset)

        for batch, (y, x) in enumerate(dataloader):
            y, x = y.to(device), x.to(device)

            # Compute prediction error
            pred = self(y)
            # loss = loglike_fn(pred, y)
            loss = loss_fn(pred, x)

            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()      


    def train_supervised(self, dataloader, loss_fn, optimizer, device, true_init=False):

        size = len(dataloader.dataset)
        self.train()
        for batch, (y, x) in enumerate(dataloader):
            y, x = y.to(device), x.to(device)

            # Compute prediction error

            # std = torch.std(x)/10
            # xtrain = x + torch.normal(0, std, size=x.shape, device=device)
            # pred = self(y, xtrain.to(torch.cdouble))

            # x0 = torch.zeros_like(x).to(torch.cdouble)

            if true_init:
                pred = self(y, x.to(torch.cdouble))
            else:
                pred = self(y)
                
            loss = loss_fn(pred, x)

            # loss = torch.norm(y - self.W(pred)) 
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()


    def expectation_step(self, xt, wprev):
        wtemp = nn.Sigmoid()(self.estep(xt))
        wnew = nn.ReLU()(self.update_layer(wtemp) + self.memory_layer(wprev))
        return wnew
        # return nn.ReLU()(self.estep(1/xt))
    

    def one_iteration(self, xk, y, tau):

        ## Apply encoder to estimated image
        zk = self.W(xk)

        ## Compute robust weight
        tau = self.expectation_step((torch.abs(y- zk).real**2).to(torch.float), tau)

        ## M Step
        zk = torch.mul(zk , tau)
        x = xk - self.Wt(zk)/self.nvis
        x = x + self.Wt(torch.mul(y, tau))/self.nvis
        x = torch.sgn(x) * self.softthresh(torch.abs(x).real - self.alpha*torch.max(torch.abs(x)))


        return x, tau


    def forward(self, y, x0=None):

        if x0 is None:
            x0 = self.Wt(y)

        xk = x0

        # tau = 1 / (torch.ones(self.nvis) + (torch.abs(y - self.W(xk))**2).to(torch.float))
        tau = torch.zeros(self.nvis)
        for k in range(self.depth):
           
           xk, tau = self.one_iteration(xk, y, tau)

        return torch.abs(xk).real



class RobiiNetV2(nn.Module):

    def __init__(self, net_width):
        super().__init__()


        self.net_width = net_width

        self.estep = nn.Linear(self.net_width, self.net_width)
        self.update_layer = nn.Linear(self.net_width, self.net_width)
        self.memory_layer = nn.Linear(self.net_width, self.net_width)
        self.trainable_robust = []
        for param in self.estep.parameters():
            self.trainable_robust.append(param)
        for param in self.update_layer.parameters():
            self.trainable_robust.append(param)
        for param in self.memory_layer.parameters():
            self.trainable_robust.append(param)

        # init layers to identity
        self.estep.weight.data = torch.eye(self.net_width)
        self.estep.bias.data = torch.zeros(self.net_width)
        self.update_layer.weight.data = torch.eye(self.net_width)
        self.memory_layer.weight.data = 0.000001*torch.eye(self.net_width)

        # do not learn estep weights
        # self.estep.weight.requires_grad_(False)
        self.estep.bias.requires_grad_(False)

        # self.update_layer.weight.requires_grad_(False)
        self.update_layer.bias.requires_grad_(False)

        # self.memory_layer.weight.requires_grad_(False)
        self.memory_layer.bias.requires_grad_(False)

    def expectation_step(self, xt, wprev):

        # normalize by variance
        xt = xt / torch.std(xt)

        # create random normal input
        # ht = nn.Sigmoid()(-self.estep(torch.log(xt)))
        ht = nn.Sigmoid()(self.estep(xt))
        wnew = nn.ReLU()(self.update_layer(ht) + self.memory_layer(wprev))
        return wnew
    


    def forward(self, y, H, threshold=1e-3, niter=10, x0=None, mstep_size=1, tau=None, gaussian=False):
        
        # include conjugate visiblities
        y = torch.cat((y, torch.conj(y)), dim=2)

        nvis = y.shape[2]
        batch_size = y.shape[0]

        assert nvis % self.net_width == 0, "nvis must be divisible by net_width"

        nblocks = nvis // self.net_width
        H_tensor = H.reshape(nblocks, self.net_width, -1)
        y = y.reshape(-1, nblocks, self.net_width)

        
        if x0 is None:
            x0 = torch.zeros((batch_size, 1, H.shape[-1]), dtype=torch.cdouble)

        xk = x0
        if tau is None:
            tau = torch.ones((batch_size, nblocks, self.net_width))

        for k in range(niter):

            # compute hessian for each H of H_tensor
            # hessian = torch.zeros((nblocks, self.net_width, self.net_width), dtype=torch.cdouble)
            # for bloc_index in range(H_tensor.shape[0]):
            #     omega = torch.diag(tau[:, bloc_index, :].reshape(-1))
            #     hessian[:, bloc_index, :, :] = torch.matmul(H_tensor[bloc_index, :, :].T,  H_tensor[bloc_index, :, :])



            xk, tau = self.one_iteration(xk, y, H=H_tensor, tau=tau,
                                            threshold=threshold, 
                                            mstep_size=mstep_size, gaussian=gaussian)

        return torch.real(xk), tau
    


    def one_iteration(self, xk, y, H, tau, threshold, mstep_size=1, gaussian=False):

        ## Apply encoder to estimated image

        x = xk
        dim = x.shape[-1]

        new_tau = torch.zeros_like(tau)
        for bloc_index in range(H.shape[0]):
            H_bloc = H[bloc_index, :, :].clone()
            y_bloc = y[:, bloc_index, :].reshape(-1,1, self.net_width).clone()
            zk = torch.matmul(xk, H_bloc.T)
            ## Compute robust weight
            tau_bloc = tau[:, bloc_index].reshape(-1,1, self.net_width).clone()
            if gaussian:
                tau_bloc = torch.ones_like(tau_bloc)
            else:
                tau_bloc = self.expectation_step((torch.abs(y_bloc - zk).real**2).to(torch.float), tau_bloc)
            new_tau[:, bloc_index] = tau_bloc.reshape(-1, self.net_width)

            sigma2 = torch.mean(torch.abs(y_bloc - zk)**2, dim=2).reshape(-1,1,1)
            # print(sigma2.shape)
            ## M Step
            # for mit in range(miter):
            residual = mstep_size * torch.mul(y_bloc - zk , tau_bloc)/sigma2
            residual_image = torch.matmul(residual, H_bloc.conj())/self.net_width/dim
            

            x += residual_image

        
        x = torch.sgn(x) * nn.ReLU()(torch.abs(x).real - threshold*torch.max(torch.abs(x)))

        return x, new_tau


    def train_supervised(self, dataloader, loss_fn, optimizer, device, H, threshold=1e-3, mstep_size=1, niter=10, SNR=10, true_init=False):


        size = len(dataloader.dataset)
        self.train()
        for batch, (y, x, xdirty) in enumerate(dataloader):
            y, x, xdirty = y.to(device), x.to(device), xdirty.to(device)

            # Compute prediction error

            # pred = self(y, xtrain.to(torch.cdouble))

            # x0 = torch.zeros_like(x).to(torch.cdouble)

            if true_init:
                # std = np.sqrt(10**-(SNR/10) * torch.var(x))
                # xtrain = x + torch.normal(0, std, size=x.shape, device=device)
                alpha = 0.9
                xtrain = alpha*x + (1-alpha)*xdirty

                pred, tau = self(y, 
                                x0=xtrain.to(torch.cdouble),
                                H=H,
                                threshold=threshold, 
                                niter=niter, 
                                mstep_size=mstep_size)




                loss = loss_fn(pred, x)
                optimizer.zero_grad()   
                loss.backward(retain_graph=True)

                optimizer.step()


                # for ii, alpha in enumerate(np.linspace(0.1, 1, 10)):


                #     if ii == 0:
                #         pred, tau = self(y, 
                #                     x0=xtrain.to(torch.cdouble),
                #                     H=H,
                #                     threshold=threshold, 
                #                     niter=niter, 
                #                     mstep_size=mstep_size)
                #     else:                    
                #         pred, tau = self(y, 
                #                 x0=xtrain.to(torch.cdouble),
                #                 H=H,
                #                 threshold=threshold, 
                #                 niter=niter, 
                #                 mstep_size=mstep_size)
                #                 # tau=tau.detach())
                        
                #     xtrain = alpha*x + (1-alpha)*xdirty

                #     loss = loss_fn(pred, xtrain) + loss_fn(tau, torch.zeros_like(tau))



                #     optimizer.zero_grad()   
                #     loss.backward(retain_graph=False)

                #     optimizer.step()

                    
            else:
                pred = self(y,H=H, threshold=threshold, niter=niter, mstep_size=mstep_size)
                
                loss = loss_fn(pred, x)

                # loss = torch.norm(y - self.W(pred)) 
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print('here')
            if batch % 1 == 0:
                loss_val, current = loss.item(), batch * len(x)
                print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

        return loss.item()