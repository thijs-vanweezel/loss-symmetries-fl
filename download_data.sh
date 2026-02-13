#!/usr/bin/env bash
set -euo pipefail

# -- OXFORD-IIIT PETS DOWNLOAD --
if [ -d oxford_pets ] && [ "$(ls -A oxford_pets/images 2>/dev/null)" ]; then
    echo "Oxford-IIIT Pets dataset already present. Skipping download."
fi
else
    mkdir -p oxford_pets
    
    curl -L https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz -o oxford_pets/images.tar.gz
    curl -L https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz -o oxford_pets/annotations.tar.gz
    
    tar -xf oxford_pets/images.tar.gz -C oxford_pets
    tar -xf oxford_pets/annotations.tar.gz -C oxford_pets

# -- IMAGENET DOWNLOAD --
if [ -d imagenet ] && [ "$(ls -A imagenet/ILSVRC 2>/dev/null)" ]
    echo "ImageNet already present. Skipping download."
else
    mkdir -p imagenet
    cd ./imagenet
    COOKIE_JAR=
    
    curl -c cookies.txt -sSL https://image-net.org/login -o login_page.html
    
    curl -b cookies.txt -c cookies.txt -sSL \
      -d email=t.g.g.v.weezel@student.tue.nl \
      -d "password=${IMAGENET_PASSWORD}" \
      https://image-net.org/login -o /dev/null
    
    curl -L -C - --retry 10 --retry-delay 5 \
            -b cookies.txt \
            -O data.tar.gz \
            https://image-net.org/data/ILSVRC/2017/ILSVRC2017_DET.tar.gz
    
    tar -xzf data.tar.gz
    
    cd ..
