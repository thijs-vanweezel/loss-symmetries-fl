#!/usr/bin/env bash
set -e

if [ -d oxford_pets/images ] && [ "$(ls -A oxford_pets/images 2>/dev/null)" ]; then
    echo "Oxford-IIIT Pets dataset already present. Skipping download."
    exit 0
fi

mkdir -p oxford_pets

curl -L https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz -o oxford_pets/images.tar.gz
curl -L https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz -o oxford_pets/annotations.tar.gz

tar -xf oxford_pets/images.tar.gz -C oxford_pets
tar -xf oxford_pets/annotations.tar.gz -C oxford_pets