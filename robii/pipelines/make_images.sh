#!/bin/sh

robii fromms SNR_G55_10s.calib.ms \
--out SNR_G55 \
--image_size 1280 \
--niter 10 \
--miter 10 \
--mstep_size 0.000005 \
--threshold .00001 \
--dof 10
--plot