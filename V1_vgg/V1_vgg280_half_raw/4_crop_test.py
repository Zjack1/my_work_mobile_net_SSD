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
det_model = "./wsnet/wsnet_train_iter_180000.caffemodel"
gpu_id = 2
test_image_path="../crop_DMS_data/test.txt" #test file

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

a=0
b=0
c=[]
d=[]
with open(test_image_path, "r") as il:
    for line in il:
        #line.split(' ', 1 )
        #line=line[:-2]
        print("-------------------------")
        print(line.split(' ', 1 )[0])
        det_caffe.blobs["data"].data[...] = det_trans.preprocess("data", caffe.io.load_image('../crop_DMS_data/test/'+line.split(' ', 1 )[0]))
        out = det_caffe.forward()
        prob= out['prob'][0]
        prob_list=prob.tolist()
        print(prob_list)
        print('label: ',int(line.split(' ', 1 )[1]),'model: ',prob_list.index(max(prob)))
        b=b+1
        c=[int(line.split(' ', 1 )[1]),prob_list.index(max(prob))]
        #print(c)
        d.append(c)
        if int(line.split(' ', 1 )[1]) == prob_list.index(max(prob)):
            a=a+1
        #label = prob.index(max(prob))
        #print(label)
        #print()
A0=0
A1=0
A2=0
A3=0
B0=0
B1=0
B2=0
B3=0
C0=0
C1=0
C2=0
C3=0
for i in range(b):
    if d[i][0] == 0:
        A0=A0+1
    if d[i][0] == 1:
        A1=A1+1
    if d[i][0] == 2:
        A2=A2+1
    if d[i][0] == 3:
        A3=A3+1
    if d[i][0] == 0 and d[i][1] != 0: # miss detect  
        B0=B0+1
    if d[i][0] == 1 and d[i][1] != 1: # miss detect  (1,0)
        B1=B1+1
    if d[i][0] == 2 and d[i][1] != 2: # miss detect
        B2=B2+1
    if d[i][0] == 3 and d[i][1] != 3: # miss detect
        B3=B3+1
    if d[i][0] != 0 and d[i][1] == 0: # error detect
        C0=C0+1
    if d[i][0] != 1 and d[i][1] == 1: # error detect(2,1)
        C1=C1+1
    if d[i][0] != 2 and d[i][1] == 2: # error detect
        C2=C2+1
    if d[i][0] != 3 and d[i][1] == 3: # error detect
        C3=C3+1
print('drink recall = ', A0/float(A0+C0), '  drink precision =  ', A0/float(A0+B0))
print('phone recall = ', A1/float(A1+C1), '  phone precision =  ', A1/float(A1+B1))
print('eye recall = ', A2/float(A2+C2), '  eye precision =  ', A2/float(A2+B2))
print('smoke recall = ', A3/float(A3+C3), '  smoke precision =  ', A3/float(A3+B3))
print('all detect = ',a)
print('all test = ',b)
print('all acc =  ',a/float(b))
'''
len_test_files = len(get_all_files(test_image_path))
all_test_path = get_all_files(test_image_path)
#print(all_test_path)
a=0
b=[]
for i in range(0,len_test_files):
    #print("loading ", i, " picture")
    print(all_test_path[i])
    det_caffe.blobs["data"].data[...] = det_trans.preprocess("data", caffe.io.load_image(all_test_path[i]))
    out = det_caffe.forward()
    prob= out['prob'][0]
    if prob[0]>0.6:
        a=a+1
    else:
        b.append(all_test_path[i])
    print(prob)
   # except:
      #  a.append(all_test_path[i])
print(a/float(len_test_files))
#print(b)
       # continue
'''


