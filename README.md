# Variational Auto Encoder Simple Benchmarks


## Installation (Linux + Docker + Nvidia GPU)

1. Install the [NVIDIA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation).

2. Install [Docker ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

3. [OPTIONALLY] [Manage docker as a non-root user ](https://docs.docker.com/install/linux/linux-postinstall/). If you don't do this step, please run the docker command with the sudo prefix.

4. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

5. Clone or copy this repository, assuming it was copied in $WORKDIR

6. Run the installer
  ```shell
  cd $WORKDIR/
  sudo chmod +x install/install.sh
  ./install/install.sh
  ```

7. Run the benchmark script
  ```shell
  docker container exec vae_container python3 benchmark.py
  ```