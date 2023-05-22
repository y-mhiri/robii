from copy import deepcopy
import os

from ducc0.wgridder import dirty2ms, ms2dirty
import numpy as np
from ..math.stats import complex_normal
from .rfi import RFI 
from .synthesis import compute_uvw_synthesis
from scipy.constants import speed_of_light
from scipy.stats import invgamma, gamma, invgauss
from .sky_model import generate_sky_model, Source
from ..astro.mstozarr import load_telescope, load_telescope_from_itrf

from PIL import Image
from omegaconf import OmegaConf
import zarr 
import datetime

import click 

from .. import ROOT_DIR


class ViSim():

    def __init__(self, 
                 ndata=1,
                 npixel=64, 
                 sources_params={
                                    'sources_density' : 1,
                                    'sources_power' : [2.5, 3.5],
                                    'sources_scale' : [5, 10],
                                    'sources_component' : 'gaussian'},
                 telescope='meerkat', 
                 synthesis_time = 0,
                 integration_time = 0,
                 dec='zenith',
                 cellsize=None,
                 freq=np.array([3e8]),
                 add_noise=False, 
                 snr=100, 
                 add_compound=False, 
                 texture_distributions=None, 
                 dof_ranges=None,
                 add_lr=False,
                 ratio_lr=0.1,
                 p_lr=100,
                 add_rfi=False, 
                 rfi_array=[RFI(1)], 
                 add_calibration_error=False,
                 std_calibration_error=0.1,
                 model_images=None,
                 dirty_images=None,
                 vis=None, 
                 clean_vis=None,
                 noise=None, 
                 gains=None,
                 vis_rfi=None, 
                 do_sim=True,
                 rng=np.random.default_rng()):

     


        self.rng = rng
        self.ndata = ndata

        self.telescope = telescope
        self.synthesis_time = synthesis_time
        self.integration_time = integration_time
        self.dec = dec
    
        # self.uvw, self.uvw_index, self.antenna_positions, self.telescope_location_lon_lat = load_telescope(f'{ROOT_DIR}/../data/telescopes/{telescope}.zip')
        self.antenna_positions, self.telescope_location_lon_lat = load_telescope_from_itrf(f'{ROOT_DIR}/../data/telescopes/{telescope}.itrf')

        if synthesis_time == 0:
            snapshot = True
        else:
            snapshot = False
        

        self.nscans = int(synthesis_time*3600 / integration_time)

        if dec == 'zenith':
            self.dec = self.telescope_location_lon_lat[1] * 180 / np.pi
        else:
            self.dec = dec
        
        # Define the frequency and wavelength
        self.freq = np.array([freq]).reshape(-1)
        self.nfreq = len(self.freq)
        self.wl = speed_of_light / self.freq



        
        # Load the telescope configuration and select the number of visibilities
        self.uvw, self.uvw_index = compute_uvw_synthesis(antenna_positions=self.antenna_positions,
                                         telescope_location=self.telescope_location_lon_lat,
                                         dec=self.dec,
                                         synthesis_time=synthesis_time,
                                         integration_time=integration_time,
                                         snapshot=snapshot)
        self.nvis = len(self.uvw)

        # radio interferometer is assumed coplanar
        self.uvw[:,-1] = 0

        # Compute the maximum cellsize

        self.cellsize_max = np.min(self.wl)/(np.max(np.abs(self.uvw))*2)
        if cellsize is None:
            self.cellsize = self.cellsize_max
        else:
            self.cellsize = cellsize

        # compute dirty beam 
        self.dirty_beam = self.compute_dirty_beam(npixel=npixel, cellsize=self.cellsize, freq=self.freq)

        self.noise_kwargs = {
            'add_noise': add_noise,
            'snr': snr,
            'add_compound': add_compound,
            'texture_distributions': texture_distributions,
            'dof_ranges': dof_ranges
        }

        self.calibration_kwargs = {
            'add_calibration_error': add_calibration_error,
            'std_calibration_error': std_calibration_error
        }


        # Define the noise parameters
        self.add_noise = add_noise
        self.snr = snr

        # Define the texture distributions for the compound noise
        self.add_compound = add_compound
        self.dof_ranges = dof_ranges

        invgamma_texture = lambda dof: invgamma.rvs(dof/2, 0, dof/2, size=1, random_state=rng)
        gamma_texture = lambda dof: gamma.rvs(dof, 0, 1/dof, size=1, random_state=rng)
        inv_gauss_texture = lambda dof: invgauss.rvs(mu=1, loc=0, scale=1/dof, size=1, random_state=rng)

        self.distributions = {
            'invgamma': invgamma_texture,
            'gamma': gamma_texture,
            'invgauss': inv_gauss_texture
        }

        if texture_distributions is None:
            self.texture_distributions = list(self.distributions.keys())
        else:
            self.texture_distributions = [item for key,item in self.distributions.items()
                                           if key in texture_distributions]  

        # Define the low rank noise parameters
        self.add_lr = add_lr
        self.ratio_lr = ratio_lr
        self.p_lr = p_lr

        # Define the RFI parameters
        self.add_rfi = add_rfi
        self.rfi_array = rfi_array

        # Define the calibration error parameters
        self.add_calibration_error = add_calibration_error
        self.std_calibration_error = std_calibration_error

        self.npixel = npixel
        self.sources_density = sources_params['sources_density']
        self.sources_component = sources_params['sources_component']
        self.sources_power = sources_params['sources_power']
        self.sources_scale = sources_params['sources_scale']


        print('Initializing simulation')
        print('-----------------------')
        print(f'Number of data: {ndata}')
        print(f'Number of pixels: {npixel}')
        print(f'Telescope: {telescope}')
        print(f'number of visibilities: {self.nvis}')
        print(f'Frequency: {freq}')
        print(f'Add noise: {add_noise}')
        print(f'SNR: {snr}')
        print(f'Add compound: {add_compound}')
        print(f'Texture distributions: {texture_distributions}')
        print(f'Degrees of freedom ranges: {dof_ranges}')
        print(f'Add rfi: {add_rfi}')
        print(f'Add calibration error: {add_calibration_error}')
        print(f'Standard deviation of calibration error: {std_calibration_error}')
        print(f'RFI array: {rfi_array}')
        print(f'Random number generator: {rng}')
        print('-----------------------')


        if do_sim is True:
            self.model_images, self.dirty_images, self.clean_vis, self.vis, self.vis_rfi, self.noise, self.gains = self.simulate(verbose=True)
        else:
            self.model_images, self.dirty_images, self.clean_vis, self.vis, self.vis_rfi, self.noise, self.gains = model_images, dirty_images, clean_vis, vis, vis_rfi, noise, gains

    @classmethod
    def from_yaml(cls, config):

        conf = OmegaConf.load(config)
        conf = OmegaConf.to_container(conf, resolve=True)
    

        conf_sim = conf['simulation_params']

        if "rfi" in conf.keys():
            rfi_config = conf['rfi']
        
            rfi_array = []
            for rfi in rfi_config:
                rfi_array.append(RFI(**rfi))
                    
            conf_sim['rfi_array'] = rfi_array

        return cls(**conf_sim)
    
    @classmethod
    def from_zarr(cls, path_to_zarr):

        store = zarr.ZipStore(path_to_zarr)
        root = zarr.group(store=store)


        # add_rfi=False, 
        # rfi_array=RFI(1), 
  

        noise_kwargs = dict(root["noise_kwargs"][:])
        calibration_kwargs = dict(root["calibration_kwargs"][:])

        metadata = dict(root["metadata"])
        # cellsize = metadata["cellsize"]

        # transform the metadata into a dictionary
        # metadata = {key: metadata[key] for key in metadata.keys()}

        ndata = int(metadata["nimage"])
        npixel = int(metadata["npixel"])
        telescope = metadata["telescope"]
        # nvis = metadata["nvis"]
        add_rfi = metadata["add_rfi"]
        rfi_power = root["rfi_power"][:]
        rfi_array = [RFI(power) for power in rfi_power]

        synthesis_time = float(metadata["synthesis_time"]) 
        integration_time = float(metadata["integration_time"]) 
        dec = float(metadata["dec"]) if metadata["dec"] != 'zenith' else 'zenith'


        model_images = root["data/model_images"][:]
        dirty_images = root["data/dirty_images"][:]
        vis = root["data/vis"][:]
        clean_vis = root["data/clean_vis"][:]
        noise = root["data/noise"][:]
        vis_rfi = root["data/rfi"][:]
        calibration_gains = root["data/gains"][:]
        do_sim = False



        # uvw = root["data/uvw"][:]
        nvis = vis.shape[1]

        freq = root["data/freq"][:]
        # antenna_positions = root["data/antenna_positions"][:]
        # uvw_index = root["data/uvw_index"][:]

        texture_distributions = root["texture_distributions"][:]
        dof_ranges = root["dof_ranges"][:]
        add_noise =  True if noise_kwargs["add_noise"]== "True" else False
        add_compound = True if noise_kwargs["add_compound"]== "True" else False
        add_calibration_error = True if calibration_kwargs["add_calibration_error"]== "True" else False

        store.close()
        return cls(ndata=ndata,
                     npixel=npixel,
                        telescope=telescope,
                        synthesis_time=synthesis_time,
                        integration_time=integration_time,
                        dec=dec,
                        add_noise=add_noise,
                        snr=float(noise_kwargs["snr"]),
                        add_compound=add_compound,
                        texture_distributions=texture_distributions,
                        dof_ranges=dof_ranges,
                        add_calibration_error=add_calibration_error,
                        std_calibration_error=float(calibration_kwargs["std_calibration_error"]),
                        add_rfi=add_rfi,
                        rfi_array=rfi_array,
                        do_sim=do_sim,
                        model_images=model_images,
                        dirty_images=dirty_images,
                        vis=vis,
                        clean_vis=clean_vis,
                        noise=noise,
                        vis_rfi=vis_rfi,
                        gains=calibration_gains,
                        freq=freq,
        )


    def save_as_zarr(self, path):
        now = datetime.datetime.now()

        metadata = {"last_modified" : now.strftime("%Y-%m-%d %H:%M"),
                    "cellsize": self.cellsize,
                    "nimage" : self.ndata,
                    "npixel" : self.npixel,
                    "nfreq" : self.nfreq,
                    "telescope" : self.telescope,
                    "add_rfi" : self.add_rfi,
                    "nrfi" : len(self.rfi_array),
                    "synthesis_time" : self.synthesis_time,
                    "integration_time" : self.integration_time,
                    "dec" : self.dec,
                    }

        
        store = zarr.ZipStore(f"{path}.zip", mode='w')
        root = zarr.group(store=store)

        root.create_dataset('metadata', data=list(metadata.items()))

        # remove lists from kwargs

        root.create_dataset('texture_distributions', data=self.noise_kwargs["texture_distributions"])
        root.create_dataset('dof_ranges', data=self.noise_kwargs["dof_ranges"])

        del self.noise_kwargs["texture_distributions"] 
        del self.noise_kwargs["dof_ranges"]


        root.create_dataset('noise_kwargs', data=list(self.noise_kwargs.items()))
        root.create_dataset('calibration_kwargs', data=list(self.calibration_kwargs.items()))

        rfi_power = [rfi.power for rfi in self.rfi_array]
        root.create_dataset('rfi_power', data=rfi_power)

        # save input of class in the zarr file


        z = root.create_group('data')
        z.create_dataset('vis', data=self.vis)
        z.create_dataset('uvw', data=self.uvw)
        z.create_dataset('noise', data=self.noise)
        z.create_dataset('rfi', data=self.vis_rfi)
        z.create_dataset('gains', data=self.calibration_gains)
        z.create_dataset('clean_vis', data=self.clean_vis)
        z.create_dataset('model_images', data=self.model_images)
        z.create_dataset('dirty_images', data=self.dirty_images)
        z.create_dataset('freq', data=self.freq)
        z.create_dataset('antenna_positions', data=self.antenna_positions)
        z.create_dataset('uvw_index', data=self.uvw_index)


        store.close()

        return True




        



    def __str__(self):
       # add a line return between each parameter
        return f"""{self.__class__.__name__}( 
                    ndata={self.ndata},
                    telescope={self.telescope},
                    npixel={self.npixel},
                    snr={self.snr},
                    texture_distributions={self.texture_distributions},
                    dof_ranges={self.dof_ranges},
                    add_noise={self.add_noise},
                    add_compound={self.add_compound.dtype},
                    add_rfi={self.add_rfi},
                    add_calibration_error={self.add_calibration_error},
                    std_calibration_error={self.std_calibration_error},
                    rfi_array={self.rfi_array},
                    nvis={self.nvis},
                    freq={self.freq},
                    rng={self.rng}
                )"""
    

    def __get_item__(self, key):
        # return model image and visibilities
        return self.model_images[key], self.vis[key], self.vis_rfi[key], self.noise[key], self.gains[key]    
    
    # def __set_item__(self, idx, dict):

    #     self.model_images[idx] = dict['model_images']
    #     self.vis[idx] = dict['vis']
    #     self.vis_rfi[idx] = dict['vis_rfi']
    #     self.noise[idx] = dict['noise']
    #     self.gains[idx] = dict['gains']
    
    def __del_item__(self, key):
        del self.model_images[key]
        del self.vis[key]
        del self.vis_rfi[key]
        del self.noise[key]
        del self.gains[key]

    # def __iter__(self):
    #     return 

    def __len__(self):
        return self.ndata
    

    def set_texture_distributions(self, texture_distributions):
        self.texture_distributions = [item for key,item in self.distributions.items()
                                           if key in texture_distributions]
        
    def set_freq(self, freq):
        self.freq = np.array([freq])
        self.nfreq = len(self.freq)
        self.wl = speed_of_light / self.freq
        self.cellsize = np.min(self.wl)/(np.max(self.uvw)*2)

    def set_uvw(self, uvw, uvw_index=None, coplanar=True):

        self.uvw = uvw
        self.uvw_index = uvw_index
        self.nvis = len(self.uvw)


        if coplanar:
            self.uvw[:,-1] = 0

        self.cellsize = np.min(self.wl)/(np.max(self.uvw)*2)



    def set_dof_ranges(self, dof_ranges):
        self.dof_ranges = dof_ranges
    
    def set_image_size(self, npixel):
        self.npixel = npixel 
        self.cellsize = np.min(self.wl)/(np.max(self.uvw)*2)
           

    def from_image_folder(self, folder_path):

        ## read config file
        config = OmegaConf.load(f"{folder_path}/config.yaml")

        self.npixel = config.npixel
        self.ndata = len(os.listdir(f"{folder_path}/images"))

        self.model_images = zarr.zeros((self.ndata, self.npixel, self.npixel))

        for ii, filename in enumerate(os.listdir(f"{folder_path}/images")):
            if filename.endswith(".png"): 

                # load image with PIL and check size
                img = Image.open(f"{folder_path}/images/{filename}")
                if img.size[0] != self.npixel or img.size[1] != self.npixel:
                    raise ValueError(f"Image size {img.size} does not match npixel {self.npixel}")
                
                # convert to numpy array and normalize
                img = np.array(img)
                img = img / np.max(img)

                # update model_images
                self.model_images[ii] = img
        
        self.simulate(update_sky_images=False)
    
    def compute_dirty_beam(self, npixel=None, cellsize=None, freq=None):

        npixel = self.npixel if npixel is None else npixel
        cellsize = self.cellsize if cellsize is None else cellsize
        freq = self.freq if freq is None else freq

        return ms2dirty(
                uvw = self.uvw,
                freq = freq,
                ms = np.ones((self.nvis, 1)).astype(np.complex64),
                npix_x = npixel,
                npix_y = npixel,
                pixsize_x = cellsize,
                pixsize_y = cellsize,
                epsilon=1.0e-5)#/self.nvis
    


        
    def compute_dirty_image(self, idx=0, vis=None, npix_x=None, npix_y=None, cellsize=None):

        npix_x = self.npixel if npix_x is None else npix_x
        npix_y = self.npixel if npix_y is None else npix_y
        cellsize = self.cellsize if cellsize is None else cellsize

        if vis is None:
            dirty_image = ms2dirty(
                    uvw = self.uvw,
                    freq = self.freq,
                    ms = self.vis[idx],
                    npix_x = npix_x,
                    npix_y = npix_y,
                    pixsize_x = cellsize,
                    pixsize_y = cellsize,
                    epsilon=1.0e-7)#/self.nvis
        else:
            dirty_image = ms2dirty(
                    uvw = self.uvw,
                    freq = self.freq,
                    ms = vis,
                    npix_x = npix_x,
                    npix_y = npix_y,
                    pixsize_x = cellsize,
                    pixsize_y = cellsize,
                    epsilon=1.0e-7)#/self.nvis
        
        return dirty_image

    def simulate_sky_image(self, sources=None, add_noise=False, rng=np.random.default_rng()):

        if sources is None:
            nsources = np.random.poisson(self.sources_density)
            power = np.random.uniform(self.sources_power[0], self.sources_power[1], nsources)
            scale = np.random.uniform(self.sources_scale[0], self.sources_scale[1], (nsources, 2))
            center = np.random.randint(-self.npixel//2, self.npixel//2, (nsources, 2))
            rot = np.random.uniform(0,360, nsources)
            component = self.sources_component

            sources = [Source(center=c, power=p, scale=s, rot=r, component=component) for c,p,s,r in zip(center, power, scale, rot)]
        else:
            if not isinstance(sources, list):
                sources = [sources]
            
        sky_image = generate_sky_model(self.npixel, sources=sources, add_noise=add_noise, rng=rng).squeeze()
        if np.sum(sky_image) == 0:
            raise ValueError("Sky image is empty")
        
        
        return sky_image
        # return generate_sky_model(self.npixel, rng=rng).squeeze()

    def simulate_noise_free_visibilities(self, model_image, cellsize=None):

        cellsize = self.cellsize if cellsize is None else cellsize
        
        return dirty2ms(
                    uvw = self.uvw,
                    freq = self.freq,
                    dirty = model_image,
                    pixsize_x = cellsize,
                    pixsize_y = cellsize,
                    epsilon=1.0e-7)
    
    def simulate_calibration_gains(self, std=1):

        gains = np.zeros((self.nvis, self.nfreq), dtype=complex)
        for k in range(self.nfreq):
            gains[:,k] = complex_normal(np.ones(self.nvis), std**2 * np.eye(self.nvis)).squeeze()
        
        return gains


    def simulate_speckle_noise(self, vis, snr, rng=np.random.default_rng()):

        P0 = np.linalg.norm(vis)**2 / self.nvis
        sigma2 = 10**(-snr/10)*P0

        speckle = complex_normal(np.zeros(self.nvis), sigma2*np.ones(self.nvis), rng=rng)
        return speckle

    def simulate_texture_noise(self, dof_ranges, texture_distributions=None, rng=np.random.default_rng()):


        if texture_distributions is not None:
            self.texture_distributions = [item for key,item in self.distributions.items()
                                           if key in texture_distributions]  

        texture = np.zeros((self.nvis,1))
        for ii in range(self.nvis):
            d_idx = rng.integers(len(self.texture_distributions))
            dof = dof_ranges[d_idx]

            arg = rng.uniform(*dof)  # invgamma (2.5, 7) #  gamma (.1, 5) #invgauss (.5,1)

            texture[ii] = self.texture_distributions[d_idx](arg)
        return texture

    def simulate_rfi_visibilities(self, rfi_array, rng=np.random.default_rng()):

        vis_rfi = np.zeros((self.nvis, self.nfreq), dtype=complex)

        
    
        n_affected_scans = int(0.25*self.nscans)
        selected_scans = rng.choice(np.arange(self.nscans), replace=False, size=n_affected_scans)
        for _, rfi in enumerate(rfi_array):

            # rfi_gains = rfi.compute_gains(self.uvw_index, self.antenna_positions)

     

            nvis_per_scan = self.nvis//self.nscans
            rfi_idx = np.array([scan * nvis_per_scan + np.arange(nvis_per_scan) for scan in selected_scans])
            rfi_idx = np.hstack(rfi_idx)

            weights = np.zeros_like(vis_rfi, dtype=float)
            weights[rfi_idx] = 1
            
            vis_rfi += dirty2ms(
                        uvw = self.uvw,
                        freq = self.freq,
                        wgt = weights,
                        dirty = rfi.sky_model(self.cellsize, self.npixel),
                        pixsize_x = self.cellsize,
                        pixsize_y = self.cellsize,
                        epsilon=1.0e-7
                            )

        return vis_rfi, selected_scans
    def simulate_low_rank_noise(self, vis, ratio_lr=0.1, p_lr=-10, rng=np.random.default_rng()):

        assert ratio_lr <= 1 and ratio_lr >= 0, "ratio_lr must be between 0 and 1"
        assert len(vis) == self.nvis, "vis must have the same length as the number of visibilities"

        rank = int(ratio_lr*self.nvis)
        # create a sparse matrix with 10% of random coefficients of size nvis x rank
        W = np.zeros((self.nvis, rank), dtype=complex)
        idxs = rng.permutation(np.arange(self.nvis))
        idxs_lr = idxs[0:rank]
        W[idxs_lr, np.arange(rank)] = np.random.uniform(0,1,rank) + 1j*np.random.uniform(0,1,rank)   



        P0 = np.linalg.norm(vis)**2  / self.nvis
        sigma2_o = 10**(p_lr/10)*P0
        W = W * np.sqrt(sigma2_o) / np.linalg.norm(W)
        vis_contaminated =  W @ complex_normal(np.zeros(rank), np.ones(rank), rng=rng).reshape(-1,1)

        # ncontaminated = int(ratio_lr*self.nvis)
        # idxs = rng.permutation(np.arange(self.nvis))
        # idxs_contaminated = idxs[0:ncontaminated]


        # P0 = np.var(vis[idxs_contaminated])
        # P0 = np.linalg.norm(vis)**2/self.nvis

        # sigma2_o = 10**(p_lr/10)
        # p0 = np.abs(vis[idxs_contaminated].flatten())**2
        # vis_contaminated = np.zeros_like(vis)
        # vis_contaminated[idxs_contaminated] = complex_normal(np.zeros(ncontaminated), sigma2_o*p0, rng=rng)          

        return vis_contaminated      

    def simulate(self, ndata=None, update_sky_images=True, sources=None, verbose=False, rng=np.random.default_rng()):

        if verbose:
            print("Simulating data...")
            print(self.__repr__())

        if ndata is not None:
            self.ndata = ndata
        

        if update_sky_images:
            self.model_images = zarr.zeros((self.ndata, self.npixel, self.npixel))
            self.dirty_images = zarr.zeros((self.ndata, self.npixel, self.npixel))

        self.clean_vis = zarr.zeros((self.ndata, self.nvis, self.nfreq), dtype=complex)
        self.vis = zarr.zeros((self.ndata, self.nvis, self.nfreq), dtype=complex)
        self.vis_rfi = zarr.zeros((self.ndata, self.nvis, self.nfreq), dtype=complex)

        self.noise = zarr.zeros((self.ndata, self.nvis, self.nfreq), dtype=complex)
        self.calibration_gains = zarr.zeros((self.ndata, self.nvis, self.nfreq), dtype=complex)


        for n in range(self.ndata):

            if verbose:
                print(f"Simulating data {n+1}/{self.ndata}")

            if verbose:
                print("Simulating sky image...")

            if update_sky_images:
                self.model_images[n] = self.simulate_sky_image(sources=sources).squeeze() #squeeze frequency dimension

            if verbose:
                print("Simulating noise-free visibilities...")


            self.clean_vis[n] = self.simulate_noise_free_visibilities(self.model_images[n])

            self.calibration_gains[n] = np.ones_like(self.clean_vis[n])



            if self.add_calibration_error:
                if verbose:
                    print("Simulating calibration gains...")
                print(type(self.add_calibration_error))
                self.calibration_gains[n] = self.simulate_calibration_gains(std=self.std_calibration_error)
            
            self.vis[n] = deepcopy(self.calibration_gains[n] * self.clean_vis[n])

            if self.add_noise:
                if verbose:
                    print("Simulating speckle noise...")

                speckle = self.simulate_speckle_noise(self.clean_vis[n], self.snr, rng)

                if self.add_compound:
                    if verbose:
                        print("Simulating texture noise...")

                    texture = self.simulate_texture_noise(self.dof_ranges, rng=rng)
                    self.noise[n] = texture * speckle
                else:
                    self.noise[n] = speckle

                self.vis[n] += self.noise[n]

            if self.add_lr:

                vis_contaminated = self.simulate_low_rank_noise(self.vis[n], self.ratio_lr, self.p_lr, rng=rng)            

                self.vis[n] += vis_contaminated

            if self.add_rfi:
                if verbose:
                    print("Simulating RFI...")

                self.vis_rfi[n],_ = self.simulate_rfi_visibilities(self.rfi_array, rng=rng)
                self.vis[n] += self.vis_rfi[n]

            self.dirty_images[n] = self.compute_dirty_image(vis=self.vis[n])

        return self.model_images, self.dirty_images, self.clean_vis, self.vis, self.vis_rfi,self.noise, self.calibration_gains







@click.command()
@click.argument('path_to_yaml', type=click.Path(exists=True))
def simulate(path_to_yaml):

    conf = OmegaConf.load(path_to_yaml)
    conf = OmegaConf.to_container(conf, resolve=True)
    

    conf_sim = conf['simulation_params']

    if "rfi" in conf.keys():
        rfi_config = conf['rfi']
    
        rfi_array = []
        for rfi in rfi_config:
            rfi_array.append(RFI(**rfi))
            
        conf_sim['rfi_array'] = rfi_array


    sim = ViSim(**conf_sim)

    # simulate data
    model_images, vis, vis_rfi, noise, gains = sim.simulate()

    # save data to zarr
    out = conf["out"]
    sim.save_as_zarr(f'{out}.zip', config=conf)

    return True



