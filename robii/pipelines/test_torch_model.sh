#!/bin/bash


output_path=/workdir/mhiriy/robii/outputs
folder_name=vla_test-thr0.0001-mstep0.001-lr0.0001

out=$output_path/$folder_name
cd $out

mkdir test_output
mkdir test_output/images

datasets_path=$out/datasets

# Generate test dataset

generate_dataset simulate --ndata 10 \
--telescope vla \
--synthesis 8 \
--dtime 60 \
--dec 'zenith' \
--npixel 128 \
--out $datasets_path/test \
--freq 1.4e9 \
--add_noise \
--snr 20 \
--add_compound \
--texture_distributions invgamma \
--dof_ranges 3 10 \
--texture_distributions gamma \
--dof_ranges .1 5 \
--texture_distributions invgauss \
--dof_ranges .5 1



# make example image

robiinet fromzarr $datasets_path/test.zip \
--nimages 10 \
--out $out/test_output/images \
--niter 10 \
--miter 10 \
--mstep_size 0.001 \
--threshold 0.0001 \
--model_path $out/train_output/robiinet_tmp.pth


# test the model
# test_model --dset_path $datasets_path/test.zip \
# --mstep_size .1 \
# --niter 10 \
# --miter 100 \
# --threshold 0.001 \
# --out $out/test_output \
# --model_path $out/train_output/robiinet.pth \
# --logpath $out/log 



# maker image from real data

# mspath=/Users/ymhiri/Documents/Dev/unrolled-robust-imaging/robii/data/ms

# robiinet fromms $mspath/SNR_G55_10s.calib.ms \
# --out $output_path/test_output/real_data/ \
# --model_path $output_path/train_output/robiinet.pth \
# --niter 10 \
# --miter 1 \
# --mstep_size 0.0005 \
# --threshold 0.00025 \
# --image_size 32 \
# --cellsize 8  \
# --fits True 

