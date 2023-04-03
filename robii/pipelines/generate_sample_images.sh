#! /bin/sh

out='/workdir/mhiriy/robii/outputs/images'

generate_dataset simulate --ndata 10 \
--telescope vla \
--synthesis 8 \
--dtime 3600 \
--dec 'zenith' \
--npixel 128 \
--out $out/dataset \
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


plot_images $out/dataset.zip $out -n 10