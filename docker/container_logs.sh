#!/bin/bash

source container_source

docker logs -t $CONTAINER_NAME -f -n 10