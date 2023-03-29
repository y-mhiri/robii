#!/bin/sh

# remove .zarr extension from the file name for each file in the directory

for f in *.zarr; do
    mv $f "${f%.MS.zarr.zarr}.zarr"
done