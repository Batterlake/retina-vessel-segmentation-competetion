#!/bin/bash

DEVICE=$1
if [ -z $DEVICE ] 
then
    DEVICE=3
fi

CUDA_VISIBLE_DEVICES=$DEVICE CUDA_DEVICE_ORDER=PCI_BUS_ID python main.py