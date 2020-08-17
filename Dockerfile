FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
LABEL version=1.0

RUN useradd -ms /bin/bash vae

# Dependencies
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libhdf5-dev libatlas-base-dev sudo zip unzip build-essential python3-setuptools ca-certificates

RUN apt-get -y upgrade

RUN python3 -m pip install --upgrade pip

USER vae

WORKDIR /home/vae/project/

COPY install/requirements.txt /tmp/

RUN pip install -r /tmp/requirements.txt

# We rather mount the local directory in the container such that output file are directly accesible
# COPY ./ /home/vae/project/