from ..simulation.simulation import ViSim
from ..simulation.rfi import RFI
from ..astro.mstozarr import save_telescope_raw
from ..imager.imager import Imager
from ..plot.plot import show_images

import click 
import numpy as np

@click.group()
def generate_dataset():
    """
    Generate a dataset
    """
    pass

@generate_dataset.command()
@click.option('--ndata', default=10, help='Number of data to generate')
@click.option('--telescope', default='vla', help='Telescope name')
@click.option('--npixel', default=32, help='Number of pixels')
@click.option('--synthesis', default=0, help='synthesis time')
@click.option('--dtime', default=0.0, help='Integration time')
@click.option('--dec', default='zenith', help='Declination of source')
@click.option('--snr', default=10.0, help='Signal to noise ratio')
@click.option('--texture_distributions', default=['invgamma', 'gamma', 'invgauss'], help='Texture distributions', multiple=True)
@click.option('--dof_ranges', default=[(5.0,10.0)], help='Degrees of freedom ranges', multiple=True)
@click.option('--add_noise/--no_noise', default=False, help='Add noise')
@click.option('--add_compound/--no_compound', default=False, help='Add compound')
@click.option('--add_rfi/--no_rfi', default=False, help='Add rfi')
@click.option('--add_calibration_error/--no_calibration_error', default=False, help='Add calibration error')
@click.option('--std_calibration_error', default=0.05, help='Standard deviation of calibration error')
@click.option('--rfi_power', default=[1.0], help='Rfi array', multiple=True)
@click.option('--freq', default=3.0e8, help='Frequency')
@click.option('--out', default='dataset.zarr', help='Output path')
def simulate(ndata, telescope, synthesis, dtime, dec, npixel, snr, texture_distributions, dof_ranges, add_noise, add_compound, 
             add_rfi, add_calibration_error, std_calibration_error, rfi_power, freq, out):
    """
    Generate a dataset
    """

    rfi_array = [RFI(power) for power in rfi_power]

    sim = ViSim(ndata=ndata,
                telescope=telescope,
                synthesis_time=synthesis,
                integration_time=dtime,
                dec=dec,
                npixel=npixel,
                snr=snr,
                texture_distributions=texture_distributions,
                dof_ranges=dof_ranges,
                add_noise=add_noise,
                add_compound=add_compound,
                add_rfi=add_rfi,
                add_calibration_error=add_calibration_error,
                std_calibration_error=std_calibration_error,
                rfi_array=rfi_array,
                freq=freq)
    
    sim.save_as_zarr(out)


@generate_dataset.command()
@click.argument('yaml', type=click.Path(exists=True))
@click.argument('out', type=click.Path(exists=False))
def fromyaml(yaml, out):
    """
    Generate a dataset from a yaml file
    """
    sim = ViSim.from_yaml(yaml)
    sim.save_as_zarr(out)



@click.command()
@click.argument('MS', type=click.Path(exists=True))
@click.argument('OUT', type=click.Path(exists=False))
def save_telescope(ms, out):
    """
    Save the uv plane and antenna positions in a zarr file
    """

    save_telescope_raw(ms, out)


@click.command()
@click.argument('DSETPATH', type=click.Path(exists=True))
@click.argument('OUTPATH', type=click.Path(exists=True))
@click.option('--nimages', '-n', default=1, help='number of images to plot')
# @click.option('--idx')
def plot_images(dsetpath, outpath, nimages):

    sim = ViSim.from_zarr(dsetpath)

    image_idxs = np.random.choice(np.arange(sim.ndata), size=nimages, replace=False)

    cellsize = sim.cellsize
    npix_x, npix_y = sim.npixel, sim.npixel
    freq = sim.freq
    uvw = sim.uvw

    for ii, idx in enumerate(image_idxs):

        imager = Imager(sim.vis[idx], cellsize=cellsize, npix_x=npix_x, npix_y=npix_y, freq=freq, uvw=uvw)
        # dirty_image = imager.make_image(method='dirty')

        images = [(sim.model_images[idx], 'Sky image')]
                #   (dirty_image, 'Dirty image')]

        show_images(*images, cmap='Spectral_r', figsize=(10,10), save=True, out=f'{outpath}/image_{ii}')        

if __name__ == '__main__':
    generate_dataset()
    
    save_telescope()