#!/bin/bash

# Start CARLA simulator in the background
./CarlaUE4.sh -RenderOffScreen -vulkan -nosound -carla-rpc-port=2000 &

# Start Jupyter Notebook server
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser
