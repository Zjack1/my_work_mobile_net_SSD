#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
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
        iou = S_iou / (S_labels + S_detection - S_iou)
        return iou


A = 0 # 检测到了对的物体
B = 0 # 正确的物体没检测到
C = 0 # 检测到了错误的物体
test_image_list = "C:\\Users\\shzhoujun\\Desktop\\testmodels\\test_image_all.txt"
with open(test_image_list, "r") as il:
    for line in il:
        test_labs_path = "C:\\Users\\shzhoujun\\Desktop\\testmodels\\testlabs\\" + line.replace(".jpg\n",".txt")
        detection_out_path = "C:\\Users\\shzhoujun\\Desktop\\testmodels\\detection_out96mid\\" + line.replace(".jpg\n",".txt")
        with open(detection_out_path, "r") as dp:
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


print("precision = ", A / (A + B))
print("recall = ", A / (A + C))
print("检测到了对的物体: ", A)
print("正确的物体没检测到:", B)
print("检测到了错误的物体:", C)







