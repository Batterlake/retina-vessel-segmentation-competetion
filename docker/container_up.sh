#!/bin/bash

BEGIN_PROCESS="sleep inf"
BEGIN_PROCESS=" /bin/bash -c \"service ssh start && sleep inf\" "

source container_source

docker run \
        --name "$CONTAINER_NAME" \
        -d \
        --restart always \
        --gpus "$CONTAINER_DEVICE" \
        $CONTAINER_MOUNTS \
        $CONTAINER_PORT_MAPPINGS \
        -w "$CONTAINER_HOME" \
        -e LANG=C.UTF-8 \
        --shm-size=2gb \
        ${CONTAINER_IMAGE_NAME} /bin/bash -c "sleep inf"