#!/bin/sh

# Install the docker image
docker build install/. -t vae:latest

# Delete the container if a previous version already exists, TODO delete iff it exists
echo 'Trying to delete container'
docker container rm vae_container

# Create the container and Install python depedencies
docker run --name vae_container --gpus all -v $PWD:/home/vae/project/ vae:latest pip install -r install/requirements.txt

# Update the image with the installed depedencies
docker commit $(docker ps -aqf "name=^vae_container$") vae_test