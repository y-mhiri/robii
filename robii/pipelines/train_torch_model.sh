#!/bin/bash


output_path=/workdir/mhiriy/robii/outputs
folder_name=vla-snr$1-n$2-t$3-m$4-lr$5-snr$6

out=$output_path/$folder_name
mkdir $out
cp train_torch_model.sh $out

cd $out

mkdir datasets
datasets_path=$out/datasets

mkdir train_output
mkdir log

# Generate train datasets 
generate_dataset simulate --ndata 1024 \
--telescope vla \
--synthesis 0 \
--dtime 0 \
--dec 'zenith' \
--npixel 128 \
--out $datasets_path/train \
--freq 1.4e9 \
--add_noise \
--snr $1 \
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
--net_depth $2  \
--net_width 351 \
--threshold $3 \
--mstep_size $4 \
--learning_rate $5 \
--step 10 \
--snr $6
--out $out/train_output \
--model_name robiinet \
--logpath $out/log/log.out \
--true_init


