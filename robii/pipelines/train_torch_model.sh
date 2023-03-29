#!/bin/bash


output_path=/Users/ymhiri/Documents/Dev/unrolled-robust-imaging/robii/outputs/test
dataset_config_file=/Users/ymhiri/Documents/Dev/unrolled-robust-imaging/robii/data/datasets/test.yaml
mspath=/Users/ymhiri/Documents/Dev/unrolled-robust-imaging/robii/data/ms

cd $output_path

mkdir datasets
datasets_path=$output_path/datasets

cp $dataset_config_file $datasets_path/config.yaml

mkdir test_output
mkdir test_output/images
mkdir test_output/real_data
mkdir train_output
mkdir log



# Generate the datasets from a yaml config file
generate_dataset fromyaml $datasets_path/config.yaml $datasets_path/train
generate_dataset fromyaml $datasets_path/config.yaml $datasets_path/test

# train the model
train_model --dset_path $datasets_path/train.zip \
--nepoch 100 \
--batch_size 64 \
--net_depth 10  \
--learning_rate 0.0001 \
--step 10 \
--out $output_path/train_output \
--model_name robiinet \
--logpath $output_path/log/log.out \
--true_init True



# make example image

robiinet fromzarr $datasets_path/test.zip \
-n 10 \
-o $output_path/test_output/images \
--niter 10 \
--miter 100 \
--mstep_size .1 \
--threshold 0.001 \
--model_path $output_path/train_output/robiinet.pth


# test the model
test_model --dset_path $datasets_path/test.zip \
--mstep_size .1 \
--niter 10 \
--miter 100 \
--threshold 0.001 \
--out $output_path/test_output \
--model_path $output_path/train_output/robiinet.pth \
--logpath $output_path/log 



# maker image from real data
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

