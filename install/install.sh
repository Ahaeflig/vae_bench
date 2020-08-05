#!/bin/sh

# Install the docker image
docker build . -t vae:latest

# Eventually add other build steps