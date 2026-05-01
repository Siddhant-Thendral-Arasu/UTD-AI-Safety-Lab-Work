FROM nvidia/vulkan:1.1.121-cuda-10.1--ubuntu18.04

# Install keys
RUN apt-key adv --fetch-keys "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub"

# Install dependencies + Python 3.7 + pip
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libsdl2-2.0 xserver-xorg libvulkan1 libomp5 \
    python3.7 python3.7-distutils python3-pip \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip for Python 3.7 and install Jupyter + ipykernel
RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install notebook ipykernel && \
    python3.7 -m ipykernel install --user --name python3.7 --display-name "Python 3.7"

# Add carla user & copy files
RUN useradd -m carla
COPY --chown=carla:carla . /home/carla

# Expose Jupyter port
EXPOSE 8888

USER carla
WORKDIR /home/carla

# Copy startup script that launches CARLA + Jupyter
COPY --chown=carla:carla start.sh /home/carla/start.sh
RUN chmod +x /home/carla/start.sh

# Default: run both
CMD ["/home/carla/start.sh"]