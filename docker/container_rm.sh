#!/bin/bash
source container_source
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
