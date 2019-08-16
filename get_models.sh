#!/bin/bash
GRAPH_FILE="frozen_inference_graph.pb"
HAARLIKE_FILE="haarcascade_frontalface_alt.xml"

if [ ! -e $GRAPH_FILE ]; then
	wget https://raw.githubusercontent.com/karaage0703/handtracking/master/model-checkpoint/ssdlitemobilenetv2/frozen_inference_graph.pb
fi


if [ ! -e $HAARLIKE_FILE ]; then
	wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml
fi
