from ..deep.train import train
from ..deep.test import test
from ..imager.imager import Imager


import click

@click.command()
@click.option('--dset_path', default='dataset.zarr', help='Path to dataset')
@click.option('--nepoch', default=100, help='Number of epochs')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--net_depth', default=5, help='Network depth')
@click.option('--learning_rate', default=0.001, help='Learning rate')
@click.option('--step', default=10, help='Step')
@click.option('--out', default='.', help='Output directory')
@click.option('--model_name', default='robii', help='Model name')
@click.option('--logpath', default='log.txt', help='Log path')
@click.option('--true_init', default=False, help='True initialisation')
def train_model(dset_path, nepoch, batch_size, net_depth, learning_rate, step, out, model_name, logpath, true_init):
    train(dset_path, nepoch, batch_size, net_depth, learning_rate, step, out, model_name, logpath)



# parameters are dataset_path, mstep_size, miter, niter, threshold
@click.command()
@click.option('--dset_path', default='dataset.zarr', help='Path to dataset')
@click.option('--mstep_size', default=1.0, help='Step size for m step')
@click.option('--miter', default=1, help='Number of iterations for m step')
@click.option('--niter', default=10, help='Number of iterations')
@click.option('--threshold', default=0.1, help='Threshold')
@click.option('--out', default='.', help='Output directory')
@click.option('--model_path', default='robii', help='Model name')
@click.option('--logpath', default='log.txt', help='Log path')
def test_model(dset_path, mstep_size, miter, niter, threshold, out, model_path, logpath):
    test(dset_path, model_path, mstep_size, miter, niter, threshold, out)

