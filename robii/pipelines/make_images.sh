#!/bin/sh

robii fromms SNR_G55_10s.calib.ms \
--out SNR_G55 \
--image_size 512 \
--niter 10 \
--miter 10 \
--mstep_size 0.0005 \
--threshold .00025 \
--dof 1
--plot