#!/bin/sh

# Install the docker image
docker build install/. -t vae:latest

# Delete the container if a previous version already exists, TODO delete iff it exists
echo 'Trying to delete container'
docker container rm vae_container

# Create the container and Install python depedencies
docker run -dit --name vae_container --gpus all -v $PWD:/home/vae/project/ vae:latest
docker exec vae_container pip install -r install/requirements.txt
docker stop vae_container

# Update the image with the installed depedencies
docker commit $(docker ps -aqf "name=^vae_container$") vae:latest