#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import numpy as np
import cv2
import math
os.environ["GLOG_minloglevel"] = "2"
import caffe


det_net ="./exp96_96/deploymid.prototxt"
det_model = "./models96/vgg_iter_25000.caffemodel"
gpu_id = 2
test_image_list="../data_new/test_image_all.txt"
conf_thres = float(0.3)
detection_out_path = "./detection_out/"

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

p=0
with open(test_image_list, "r") as il:
    for line in il:
        image_path = "../data_new/dataset/VOC_Car_Images_and_Labels/"+line.strip("\n")
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print("Error: Image is None. %s" % image_path)
            continue
        image_height, image_width = cv_image.shape[:2]

        # detect
        det_caffe.blobs["data"].data[...] = det_trans.preprocess("data", caffe.io.load_image(image_path))
        det_caffe_output = det_caffe.forward()
        det_results = det_caffe_output["detection_out"][0][0].copy()
        num_results = 0
        fo = open(detection_out_path + line.replace(".jpg\n",".txt"), "w")
        for i in range(det_results.shape[0]):
            conf = det_results[i][2]
            if conf < conf_thres:
                continue
            num_results += 1
            bbox_xmin = det_results[i][3] * image_width
            bbox_ymin = det_results[i][4] * image_height
            bbox_xmax = det_results[i][5] * image_width
            bbox_ymax = det_results[i][6] * image_height
            
            d=[str(int(math.ceil(conf))) +' ',str(int(bbox_xmin)) +' ',str(int(bbox_ymin)) +' ',str(int(bbox_xmax)) +' ',str(int(bbox_ymax)) +'\n']
            fo.writelines(d)
            #print("Conf: %f BBox: %f %f %f %f" % (conf, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
        p=p+1
        print(p)
        fo.close()
        #print("Image: %s %d" % (image_path, num_results))

