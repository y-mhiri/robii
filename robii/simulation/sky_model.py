import numpy as np



class Source():
    def __init__(self, center, power, scale):

        self.center = center # in degree (l, m) are cosine direction from the phase center
        self.power = power # in Jy
        self.scale = scale # in degree

    

class SkyModel():

    def __init__(self, size, cellsize, src_density=13000, scale_min=5, scale_max=20,
                 add_noise=False, rng=np.random.default_rng()):

        if type(size) is int:
            self.npix_x = size
            self.npix_y = size
        elif type(size) is tuple or type(size) is list:
            self.npix_x = size[0]
            self.npix_y = size[1]
        else:
            raise TypeError('size must be an integer or a tuple or list of two integers')


        if type(cellsize) is int or type(cellsize) is float:
            self.cellsize_x = cellsize
            self.cellsize_y = cellsize
        elif type(cellsize) is tuple or type(cellsize) is list:
            self.cellsize_x = cellsize[0]
            self.cellsize_y = cellsize[1]
        else:
            raise TypeError('cellsize must be a float or a tuple or list of two floats')
        

        self.fov = (self.npix_x*self.cellsize_x, self.npix_y*self.cellsize_y)
        self.src_density = src_density # number of sources per square degree
        self.scale_min = scale_min # in degree 
        self.scale_max = scale_max # in degree
        self.add_noise = add_noise
        self.rng = rng

        # self.skymodel = self.generate_sky_model()

    def generate_sky_model(self, sources=None):

        if sources is None:
            nsources = self.rng.poisson(self.src_density* (self.fov[0]*self.fov[1]))
            print( 'Number of sources: ', nsources)
            sources = []
            for _ in range(nsources):
                center_x = self.rng.uniform(-self.fov[0]/2, self.fov[0]/2)
                center_y = self.rng.uniform(-self.fov[1]/2, self.fov[1]/2)
                center = (center_x, center_y)

                power = self.rng.uniform(1, 10)
                scale = self.rng.uniform(self.scale_min, self.scale_max, size=2)
                sources.append(Source(center, power, scale))

        sky_model = np.zeros((self.npix_x, self.npix_y))

        for s in sources:
            source_center = (s.center[0]/self.cellsize_x , s.center[1]/self.cellsize_y)
            source_power = s.power
            source_scale = (s.scale[0]/self.cellsize_x, s.scale[1]/self.cellsize_y)
            sky_model += self.ellipsoid(source_center, source_power, source_scale, (self.npix_x, self.npix_y))

        if self.add_noise:
            std = np.sqrt(np.sum(sky_model**2)/nsources)/100
            sky_model += self.rng.normal(0, std, size=(self.npix_x, self.npix_y))
        
        return sky_model
    

    def ellipsoid(self, center, P, scale, size):

        npix_x, npix_y = size

        X,Y = np.meshgrid(np.linspace(-npix_x//2, npix_x//2, npix_x),np.linspace(-npix_y//2, npix_y//2, npix_y))
        ellipsoid_image = P * np.exp(-((X-center[0]*np.ones_like(X))**2/scale[0] + (Y-center[1]*np.ones_like(Y))**2/scale[1]))
        return ellipsoid_image

def generate_sky_model(npixel, src_density=.5, scale_min=5, scale_max=20,
                       sources=None, add_noise=False, rng=np.random.default_rng()):

    skymodel = np.zeros((npixel, npixel))

    if sources is None:
        nsources = rng.poisson(src_density*(npixel**2/scale_max**2))
        sources = []
        for _ in range(nsources):
            center = rng.uniform(-npixel//2, npixel//2, size=2)
            power = rng.uniform(1, 10)
            scale = rng.uniform(scale_min, scale_max, size=2)
            sources.append(Source(center, power, scale))     
    else:
        nsources = len(sources)


    for s in sources:
        source_center = s.center
        source_power = s.power
        source_scale = s.scale
        skymodel += ellipsoid(source_center, source_power, source_scale, npixel)

    if add_noise:
        std = np.sqrt(np.sum(skymodel**2)/nsources)/100
        skymodel += rng.normal(0, std, size=(npixel, npixel))

    return skymodel