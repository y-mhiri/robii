import numpy as np
from scipy.constants import speed_of_light
from ducc0.wgridder import dirty2ms, ms2dirty

def convolutional_gridding(vis, kernel, grid):
    """
        - image is assumed square of size sqrt(len(grid))
    """
    npixel = np.sqrt(len(grid)).astype(int)
    G = np.zeros((npixel, npixel)).astype(complex)
    for (u_grid, u_idx, v_grid, v_idx) in grid:
        if np.linalg.norm(kernel[-v_idx, u_idx]) > 0:
            G[-v_idx, u_idx] = kernel[-v_idx, u_idx]@vis
            # G[-v_idx, u_idx] = (1/np.linalg.norm(kernel[-v_idx, u_idx])**2) * kernel[-v_idx, u_idx]@vis
    
    G[-npixel//2::, -npixel//2::] = np.flip(G[0:npixel//2, 0:npixel//2:].conj(), axis=(0,1))
    G[-npixel//2::, 0:npixel//2] = np.flip(G[0:npixel//2, -npixel//2::].conj(), axis=(0,1))

    return G


def convolutional_degridding(gridded_vis, kernel):

    npixel = gridded_vis.shape[0]
    nvis = kernel.shape[-1]
    vis = np.zeros(nvis).astype(complex)

    
    U, V = np.meshgrid(*[np.arange(npixel) for _ in range(2)])
    for idx,_ in enumerate(vis):
        vis[idx] = np.sum([kernel[v, u, idx] * gridded_vis[v, u] 
                            for (u,v) in zip(vec(U),vec(V))])
        # vis[idx] = (1/np.linalg.norm(kernel[:,:,idx])**2) * np.sum([kernel[v, u, idx] * gridded_vis[v, u] 
        #                     for (u,v) in zip(vec(U),vec(V))])

    return vis



def grid(vis, uvw, freq, npix_x, npix_y, cellsize=None):
    
    uvw = np.concatenate((uvw, -uvw), axis=0)
    vis = np.concatenate((vis, vis.conj()), axis=0)
    if cellsize is None:
        cellsize = speed_of_light/freq.max()/2/np.abs(uvw).max()

    dirty = ms2dirty(
        uvw = uvw,
        freq = freq,
        ms = vis.reshape(-1,len(freq)),
        npix_x = npix_x,
        npix_y = npix_y,
        pixsize_x = cellsize,
        pixsize_y = cellsize,
        epsilon=1.0e-5
    )

    fft2, fftshift = np.fft.fft2, np.fft.fftshift
    gridded_visibilities = fftshift(fft2(dirty,norm='forward'))

    return gridded_visibilities

def degrid(gridded_visibilities, uvw, freq, cellsize=None):

    if cellsize is None:
        cellsize = speed_of_light/freq.max()/2/np.abs(uvw).max()


    ifft2, ifftshift = np.fft.ifft2, np.fft.ifftshift

    dirty = ifft2(ifftshift(gridded_visibilities), norm='backward')

    degridded_visibilities = dirty2ms(
        uvw = uvw,
        freq = freq,
        dirty = dirty.real,
        pixsize_x = cellsize,
        pixsize_y = cellsize,
        epsilon=1.0e-5
    )


    return degridded_visibilities 