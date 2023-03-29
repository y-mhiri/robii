import numpy as np



class Source():
    def __init__(self, center, power, scale):
        self.center = center # in pixels but think about it in radians (ra, dec)
        self.power = power # in Jy
        self.scale = scale # in pixels but think about it in radians


    
    

def ellipsoid(center, P, scale, npixel):
    X,Y = np.meshgrid(np.linspace(-npixel//2, npixel//2, npixel),np.linspace(-npixel//2, npixel//2, npixel))
    ellipsoid_image = P * np.exp(-((X-center[0]*np.ones_like(X))**2/scale[0] + (Y-center[1]*np.ones_like(Y))**2/scale[1]))
    return ellipsoid_image

def generate_sky_model(npixel, src_density=.5, scale_min=5, scale_max=20,
                       sources=None, rng=np.random.default_rng()):

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

    return skymodel