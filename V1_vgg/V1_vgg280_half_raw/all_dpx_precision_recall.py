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

#det_net ="./tinyssd_deploy.prototxt"
det_net ="./deploy.prototxt"
det_model = "./vgg300half/vgg_iter_20000.caffemodel"
gpu_id = 2
test_image_list="../all_demo/all_1.txt" #999张测试集
#test_image_list="../data_new/test_image_path.txt" #1239张测试集
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
        image_path = "../all_demo/1T/"+line.strip("\n").strip()
        cv_image = cv2.imread(image_path)
	#print(image_path)
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
        print("loading......" + str(p) + "  picture")
        fo.close()
        #print("Image: %s %d" % (image_path, num_results))

def IOU(x_labels, y_labels, w_labels, h_labels, x_detection, y_detection, w_detection, h_detection):
    x_iou = max(x_labels, x_detection)
    y_iou = max(y_labels, y_detection)
    w_iou = min(w_labels, w_detection)
    h_iou = min(h_labels, h_detection)
    S_labels = (w_labels - x_labels)*(h_labels - y_labels)
    S_detection = (w_detection - x_detection)*(h_detection - y_detection)
    w = w_iou - x_iou
    h = h_iou - y_iou
    if w < 0 or h < 0:
        return 0
    else:
        S_iou = w * h
        iou = S_iou / float((S_labels + S_detection - S_iou))
        return iou



A = 0 # 检测到了对的物体
B = 0 # 正确的物体没检测到
C = 0 # 检测到了错误的物体
#test_image_list = "C:\\Users\\shzhoujun\\Desktop\\testmodels\\test_all_image_path.txt"
with open(test_image_list, "r") as il:
    for line in il:
        test_labs_path = "../all_demo/1T_txt_labels/" + line.strip("\n").replace(".jpg",".txt")
        detection_out_p = "./detection_out/" + line.replace(".jpg\n",".txt")
#	print(detection_out_p)
        with open(detection_out_p, "r") as dp:
            dp = dp.readlines()
            dp1 = copy.deepcopy(dp)#用来移除值的
            with open(test_labs_path, "r") as lp:
                lp = lp.readlines()
                lp1 = copy.deepcopy(lp) #用来移除值的

                for d in dp:
                    # 若该图片一个都没检测到
                    if len(dp) == 0:
                        B = B + len(lp)
                        break
                    # 若有检测到物体
                    else:
                        d_new = d.split(' ')  # 按空格切分列表
                        for l in lp1:
                            l_new = l.split(' ')
                            iou = IOU(int(l_new[1]), int(l_new[2]), int(l_new[3]), int(l_new[4]), int(d_new[1]), int(d_new[2]),
                                      int(d_new[3]), int(d_new[4]))
                            if iou >= 0.5:
                                A = A + 1 #检测到了正确的物体
                                lp1.remove(l)
                                dp1.remove(d)
                                break

        C = C + len(dp1) #检测到了错误的物体（不在GT中）
        B = B + len(lp1) #没有检测到的物体


print("precision = ", A / float((A + C)))
print("recall = ", A / float((A + B)))
print( A)
print( B)
print( C)


