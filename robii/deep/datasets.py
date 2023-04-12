"""
Datasets class are defined.


"""


import zarr 
import numpy as np
    
class ViDataset():

    def __init__(self, vis, uvw, freq, model_images, dirty_images, nimages, npixel, cellsize):

        self.vis = vis
        self.uvw = uvw
        self.freq = freq
        self.model_images = model_images
        self.dirty_images = dirty_images
        self.nimages = nimages
        self.npixel = npixel
        self.cellsize = cellsize
        
    @classmethod
    def from_sim(cls, visim):
        vis = visim.vis
        uvw = visim.uvw
        freq = visim.freq
        model_images = visim.model_images
        dirty_images = visim.dirty_images
        nimages = visim.ndata
        npixel = visim.npixel
        cellsize = visim.cellsize

        return cls(vis, uvw, freq, model_images, dirty_images, nimages, npixel, cellsize)


    @classmethod
    def from_zarr(cls, path):
        store = zarr.ZipStore(path)
        z = zarr.group(store=store)

        metadata = dict(z["metadata"])
        nimages = metadata["nimage"].astype(int)
        npixel = metadata["npixel"].astype(int)
        nvis = len(z["data/uvw"])
        uvw = np.array(z['data/uvw']).squeeze()     
        freq = np.array(z['data/freq']).reshape(-1)
        cellsize = metadata["cellsize"].astype(float)

        vis = z["data/vis"][:]
        model_images = z["data/model_images"][:]
        dirty_images = z["data/dirty_images"][:]

        return cls(vis, uvw, freq, model_images, dirty_images, nimages, npixel, cellsize)






    def __getitem__(self,idx):

        return self.vis[idx].reshape(1,-1), \
                self.model_images[idx].reshape(1,-1).astype(np.float64), \
                self.dirty_images[idx].reshape(1,-1).astype(np.float64)

    def __len__(self):
        return self.nimages
