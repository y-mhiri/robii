import numpy as np



class Source():
    def __init__(self, center, power, scale=None, rho=0, rot=0, component='point_source'):

        self.center = center # in degree (l, m) are cosine direction from the phase center
        self.power = power # in Jy
        self.scale = scale # in degree
        self.rho = rho # correlation coefficient
        self.rot = rot # rotation in degrees

        if component not in ['gaussian', 'point_source']:
            raise ValueError('component must be gaussian or point_source')
        
        self.component = component

    

class SkyModel():

    def __init__(self, size, cellsize):

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


        # self.skymodel = self.generate_sky_model()

    def generate_sky_model(self, sources=None, src_density=None, scale_min=None, scale_max=None, component='point_source', add_noise=False, std=1, rng= np.random.default_rng()):

        if sources is None:
            nsources = rng.poisson(src_density* (self.fov[0]*self.fov[1]))
            print( 'Number of sources: ', nsources)
            sources = []
            for _ in range(nsources):

                center_x = rng.uniform(0, self.fov[0]/2) * rng.choice([-1, 1]) # done this way to avoid the the extreme edges of the fov
                center_y = rng.uniform(0, self.fov[0]/2) * rng.choice([-1, 1])
                center = (center_x, center_y)


                power = rng.uniform(1, 10)
                if component == 'gaussian':
                    scale = rng.uniform(scale_min, scale_max, size=2)
                    sources.append(Source(center, power, scale, component=component))
                elif component == 'point_source':
                    sources.append(Source(center, power, component=component))
                else:
                    raise ValueError('component must be gaussian or point_source')

                

        sky_image = np.zeros((self.npix_x, self.npix_y))
        self.nsources = len(sources)

        for s in sources:
            source_center = (s.center[0]/self.cellsize_x , s.center[1]/self.cellsize_y)
            source_power = s.power
            component = s.component

            if component == 'gaussian':
                source_scale = (s.scale[0]/self.cellsize_x, s.scale[1]/self.cellsize_y)
                print(source_center, source_power, source_scale)
                sky_image += self.ellipsoid(source_center, source_power, source_scale, (self.npix_x, self.npix_y))
            elif component == 'point_source':
                sky_image += self.point_source(source_center, source_power, (self.npix_x, self.npix_y))

        if add_noise:
            sky_image += self.rng.normal(0, std, size=(self.npix_x, self.npix_y))
        

        return sky_image
    

    def ellipsoid(self, center_px, P, scale_px, size):

        npix_x, npix_y = size

        X,Y = np.meshgrid(np.linspace(-npix_x//2, npix_x//2, npix_x),np.linspace(-npix_y//2, npix_y//2, npix_y))
        ellipsoid_image = P * np.exp(-((X-center_px[0]*np.ones_like(X))**2/scale_px[0]**2 + (Y-center_px[1]*np.ones_like(Y))**2/scale_px[1]**2))
        return ellipsoid_image
    

    def point_source(self, center, P, size):

        npix_x, npix_y = size

        point_source_image = np.zeros((npix_x, npix_y))
        point_source_image[np.round(center[0] + npix_x//2).astype(int), np.round(center[1] + npix_y//2).astype(int)] = P
        return point_source_image
    
def gaussian2D(   
                  size=(256,256), # Size of the image
                  amplitude=1,  # Highest intensity in image.
                  center_px=(0,0),  # Coordinates of the center of the gaussian
                  scale_px=(10,10),  # Standard deviation in x.
                  rho=0,  # Correlation coefficient.
                  rot=0):  # rotation in degrees.
        
        x, y = np.meshgrid(np.linspace(-size[0]//2, size[0]//2, size[0]),np.linspace(-size[1]//2, size[1]//2, size[1]))
        rot = np.deg2rad(rot)

        x_ = np.cos(rot)*x - y*np.sin(rot)
        y_ = np.sin(rot)*x + np.cos(rot)*y

        xo, yo = center_px
        xo = float(xo)
        yo = float(yo)

        xo_ = np.cos(rot)*xo - yo*np.sin(rot) 
        yo_ = np.sin(rot)*xo + np.cos(rot)*yo

        x,y,xo,yo = x_,y_,xo_,yo_

        C = 4 * np.log(2)

        # Create covariance matrix

        sigma_x, sigma_y = scale_px
        mat_cov = [[C * sigma_x**2, rho * sigma_x * sigma_y],
                   [rho * sigma_x * sigma_y, C * sigma_y**2]]
        mat_cov = np.asarray(mat_cov)
        # Find its inverse
        mat_cov_inv = np.linalg.inv(mat_cov)

        # PB We stack the coordinates along the last axis
        mat_coords = np.stack((x - xo, y - yo), axis=-1)

        G = amplitude * np.exp(-np.matmul(np.matmul(mat_coords[:, :, np.newaxis, :],
                                                        mat_cov_inv),
                                              mat_coords[..., np.newaxis])) 
        return G.squeeze()


def ellipsoid(center_px, P, scale_px, size):

    npix_x, npix_y = size

    X,Y = np.meshgrid(np.linspace(-npix_x//2, npix_x//2, npix_x),np.linspace(-npix_y//2, npix_y//2, npix_y))
    ellipsoid_image = P * np.exp(-((X-center_px[0]*np.ones_like(X))**2/scale_px[0]**2 + (Y-center_px[1]*np.ones_like(Y))**2/scale_px[1]**2))
    return ellipsoid_image

def point_source(center, P, size):

    npix_x, npix_y = size

    point_source_image = np.zeros((npix_x, npix_y))
    point_source_image[np.round(center[0] + npix_x//2).astype(int), np.round(center[1] + npix_y//2).astype(int)] = P
    return point_source_image


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
            rot = rng.uniform(0, 360)
            rho = rng.uniform(-1, 1)
            sources.append(Source(center, power, scale, rot, rho))     
    else:
        nsources = len(sources)


    for s in sources:
        source_center = s.center
        source_power = s.power
        source_scale = s.scale
        source_rot = s.rot
        source_rho = s.rho
        # skymodel += ellipsoid(source_center, source_power, source_scale, (npixel, npixel))

        if s.component == 'gaussian':
            skymodel += gaussian2D(size=(npixel, npixel), 
                               amplitude=source_power, 
                               center_px=source_center, 
                               scale_px=source_scale, 
                               rho=source_rho,
                               rot=source_rot)
            
        elif s.component == 'point_source':
            skymodel += point_source(source_center, source_power, (npixel, npixel))

        else:
            raise ValueError('Source component not recognized; must be "gaussian" or "point_source"')
        
    if add_noise:
        std = np.sqrt(np.sum(skymodel**2)/nsources)/100
        skymodel += rng.normal(0, std, size=(npixel, npixel))

    return skymodel