#!/bin/bash

source container_source
docker exec -t $CONTAINER_NAME jupyter notebook --ip=0.0.0.0 --port=$JUPYTER_PORT --NotebookApp.token='gorai' --allow-root --no-browser