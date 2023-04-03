#!/bin/bash

# output directory
output_path=/workdir/mhiriy/robii/outputs
folder_name=vla_8h_1h

out=$output_path/$folder_name
cd $out

datasets_path=$out/datasets
mkdir synthesis_output


## Generate test sets and compute metrics for snapshot observation

### generate test sets
generate_dataset simulate --ndata 128 \
--telescope vla \
--synthesis 0 \
--dtime 0 \
--dec 'zenith' \
--npixel 128 \
--out $datasets_path/test_snapshot \
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

### test the model
test_model --dset_path $datasets_path/test_snapshot.zip \
--mstep_size .1 \
--niter 10 \
--miter 100 \
--threshold 0.001 \
--out $out/test_output \
--model_path $out/train_output/robiinet.pth \
--logpath $out/log 
--name snapshot


## Generate test sets and compute metrics for multiple synthesis time

for synthesis_time in 0.25 1 4
do 

generate_dataset simulate --ndata 128 \
--telescope vla \
--synthesis $synthesis_time \
--dtime 60 \
--dec 'zenith' \
--npixel 128 \
--out $datasets_path/test_$synthesis_time \
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


### test the model
test_model --dset_path $datasets_path/test_$synthesis_time.zip \
--mstep_size .1 \
--niter 10 \
--miter 100 \
--threshold 0.001 \
--out $out/test_output \
--model_path $out/train_output/robiinet.pth \
--logpath $out/log 
--name $synthesis_time

done
