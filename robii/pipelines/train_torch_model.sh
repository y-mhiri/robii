#!/bin/bash


output_path=/workdir/mhiriy/robii/outputs
folder_name=vla_8h_1h

out=$output_path/$folder_name
mkdir $out
cd $out

mkdir datasets
datasets_path=$out/datasets

mkdir train_output
mkdir log

# Generate train datasets 
generate_dataset simulate --ndata 128 \
--telescope vla \
--synthesis 8 \
--dtime 3600 \
--dec 'zenith' \
--npixel 128 \
--out $datasets_path/train \
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


# train the model
train_model --dset_path $datasets_path/train.zip \
--nepoch 100 \
--batch_size 16 \
--net_depth 10  \
--learning_rate 0.0001 \
--step 10 \
--out $out/train_output \
--model_name robiinet \
--logpath $out/log/log.out \
--true_init


