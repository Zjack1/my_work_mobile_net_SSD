#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import numpy as np
import cv2
import math
os.environ["GLOG_minloglevel"] = "2"
import caffe
import copy
det_net ="./deploy.prototxt"
det_model = "./wsnet/wsnet_train_iter_450000.caffemodel"
gpu_id = 2
test_image_path="../yawn_data/test/yawn" #test file
#test_image_path="../grayTest"
conf_thres = float(0.3)


def get_all_files(dir):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_


if gpu_id < 0:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
# det caffe
det_caffe = caffe.Net(det_net, det_model, caffe.TEST)

det_caffe_channel, det_caffe_height, det_caffe_width = det_caffe.blobs['data'].data.shape[1:]
det_caffe.blobs["data"].reshape(1, det_caffe_channel, det_caffe_height, det_caffe_width)

det_trans = caffe.io.Transformer({"data": det_caffe.blobs["data"].data.shape})
det_trans.set_transpose("data", (2, 0, 1))
det_trans.set_raw_scale('data', 255)
det_trans.set_mean("data", np.array([104, 117, 123]))
det_trans.set_channel_swap('data', (2, 1, 0))

len_test_files = len(get_all_files(test_image_path))
all_test_path = get_all_files(test_image_path)
#print(all_test_path)
a=0
b=[]
c=[]
d=[]
for i in range(0,len_test_files):
    
        #print("loading ", i, " picture")
    print(all_test_path[i])
    det_caffe.blobs["data"].data[...] = det_trans.preprocess("data", caffe.io.load_image(all_test_path[i]))
    out = det_caffe.forward()
    try:
        prob= out['prob']
        if prob[0][1]>0.5:
            a=a+1
        else:
            b.append(all_test_path[i])
        print(prob[0])
    except:
        prob= out['prob']
        c.append(all_test_path[i])
        d.append(prob)
        continue
print(a/float(len_test_files))
#print(c)
#print(d)
       # continue


