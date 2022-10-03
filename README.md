# retina-vessel-segmentation-competetion
Repository dedicated to training retinal vessel segmentation models


# Container operations

*inside docker/ folder*

## Build docker container

0. container configuration with port mappings and container naming can be found in **container_source** file
1. ./build.sh 

## Run container

0. container configuration with port mappings and container naming can be found in **container_source** file
1. ./container_up.sh

## Attach to a running container's shell

1. ./bash_exec.sh

## View running container logs

1. ./container_logs.sh

## Run jupyter notebook inside a running container

1. ./jupyter_exec.sh

## Shutdown and remove running container

1. ./container_rm.sh

# Train model
**execute everything in running docker container**

1. configure experiment in **config.py**
2. ./train.sh 0 **first argument has to be a number pointing to an existing gpu_id in PCI_BUS_ID order**

Experiment configuration, logs, epoch metrics, tensorboard data and checkpoints will be stored in experiment folder in **experiments/**

# Evaluate model

1. ./eval.sh <path to experiment configuration file .json> <path to folder with images to process> <device_id>

Evaluation result will be placed in selected model's experiment folder. Submission .zip file will be created too.
