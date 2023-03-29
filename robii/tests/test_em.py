"""
Scripts that test the robust em imager
- plot images
- save dataframes with results

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
import argparse
import pickle

from ..imager.imager import Imager


# function to 

def test_robust_em_imager(ms, cellsize, npix_x, npix_y, niter, nu, alpha, gamma, gaussian, miter, verbose, plot, save, save_path):
