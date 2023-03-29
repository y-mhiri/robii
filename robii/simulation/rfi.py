import numpy as np
from scipy.stats import lognorm

class RFI():

    def __init__(self, power=1, sky_coordinates=None, rfi_position=None):

        self.power = power # in Jy
        # self.frequency_range = frequency_range # in Hz
        self.sky_coordinates = sky_coordinates # in radians
        self.rfi_position = rfi_position # position relative to the centroid of the sensor array

    def __str__(self):
        return f"""
                RFI properties:
                Power: {self.power} Jy
                Sky coordinates: {self.sky_coordinates}
                RFI position: {self.rfi_position}
                """


    def sky_model(self, cellsize, npixel, rng=np.random.default_rng()):
        """
        Create a sky model for the RFI modeled as a point source

        """

        sky_model = np.zeros((npixel, npixel))

        # Compute the pixel coordinates of the RFI

        if self.sky_coordinates is not None: # if coordinates are given in radians
            x = np.int(npixel//2 + self.sky_coordinates[0]/cellsize)
            y = np.int(npixel//2 + self.sky_coordinates[1]/cellsize)
        else: # random coordinates
            x = rng.integers(0, npixel)
            y = rng.integers(0, npixel)
        
        sky_model[x, y] = self.power

        return sky_model


    def compute_gains(self, index, antenna_positions, nstd=1 , s=0.001):
        """
        Compute the gains of the RFI
        """

        # Compute the center of the array
        center = np.mean(antenna_positions, axis=0)
        std = np.std(antenna_positions, axis=0)

        if self.rfi_position is not None:
            rfi_position = center + self.rfi_position 
        else:
            rfi_position = center + np.random.multivariate_normal(np.zeros(3),(nstd*std)**2*np.eye(3))

        gains = np.zeros((index.shape[0]))

        for ii, b in enumerate(index):
            antenna1 = antenna_positions[b[0]]
            antenna2 = antenna_positions[b[1]]

            # Compute the distance between the antennas and the RFI source
            distance_antenna1 = np.linalg.norm(antenna1 - rfi_position)
            distance_antenna2 = np.linalg.norm(antenna2 - rfi_position)

            # Compute the gains
            gain1 = 1/distance_antenna1
            gain2 = 1/distance_antenna2

            gains[ii] = gain1*gain2
        
        gains = gains/np.max(gains)
        gains += lognorm.rvs(s=s, size=index.shape[0])

        return gains
    
