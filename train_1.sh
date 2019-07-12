#!/bin/sh
#if ! test -f example/MobileNetSSD_train.prototxt ;then
	#echo "error: example/MobileNetSSD_train.prototxt does not exist."
	#echo "please use the gen_model.sh to generate your own model."
        #exit 1
/opt/caffe/build/tools/caffe train \
-solver /DockerData/mobilenetv1/solver_train.prototxt \
-gpu 0,1,2 
