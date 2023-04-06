"""
Datasets class are defined.


"""


import zarr 
import numpy as np
    
class ViDataset():

    def __init__(self, path):

        store = zarr.ZipStore(path)
        self.z = zarr.group(store=store)

        metadata = dict(self.z["metadata"])
        self.nimages = metadata["nimage"].astype(int)
        self.npixel = metadata["npixel"].astype(int)
        self.nvis = len(self.z["data/uvw"])
        self.uvw = np.array(self.z['data/uvw']).squeeze()     
        self.freq = np.array(self.z['data/freq']).reshape(-1)
        self.cellsize = metadata["cellsize"].astype(float)

    def __getitem__(self,idx):

        return self.z["data/vis"][idx].reshape(1,-1), \
                self.z["data/model_images"][idx].reshape(1,-1).astype(np.float64), \
                self.z["data/dirty_images"][idx].reshape(1,-1).astype(np.float64)

    def __len__(self):
        return self.nimages
