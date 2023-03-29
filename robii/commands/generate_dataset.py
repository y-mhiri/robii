from ..simulation.simulation import ViSim
from ..astro.mstozarr import save_telescope_raw
import click 


@click.group()
def generate_dataset():
    """
    Generate a dataset
    """
    pass

@generate_dataset.command()
@click.option('--ndata', default=1000, help='Number of data to generate')
@click.option('--telescope', default='vla', help='Telescope name')
@click.option('--npixel', default=512, help='Number of pixels')
@click.option('--snr', default=10, help='Signal to noise ratio')
@click.option('--texture_distributions', default='gaussian', help='Texture distributions')
@click.option('--dof_ranges', default='(0,1)', help='Degrees of freedom ranges')
@click.option('--add_noise', default=False, help='Add noise')
@click.option('--add_compound', default=False, help='Add compound')
@click.option('--add_rfi', default=False, help='Add rfi')
@click.option('--add_calibration_error', default=False, help='Add calibration error')
@click.option('--std_calibration_error', default=0.05, help='Standard deviation of calibration error')
@click.option('--rfi_array', default='rfi_array.npy', help='Rfi array')
@click.option('--nvis', default=1000, help='Number of visibilities')
@click.option('--freq', default=1000, help='Frequency')
@click.option('--out', default='dataset.zarr', help='Output path')
def simulate(ndata, telescope, npixel, snr, texture_distributions, dof_ranges, add_noise, add_compound, add_rfi, add_calibration_error, std_calibration_error, rfi_array, nvis, freq, out):
    """
    Generate a dataset
    """
    sim = ViSim(ndata=ndata,
                telescope=telescope,
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
                nvis=nvis,
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


if __name__ == '__main__':
    generate_dataset()
    
    save_telescope()