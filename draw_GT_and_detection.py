#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
test_image_path = "C:\\Users\\shzhoujun\\Desktop\\testmodels\\GTimg\\"
test_labs_path = "C:\\Users\\shzhoujun\\Desktop\\testmodels\\detection_out128\\"
test_image_list = "C:\\Users\\shzhoujun\\Desktop\\testmodels\\test_image_all.txt"
with open(test_image_list, "r") as il:
    for line in il:
        image_path = test_image_path+line.strip("\n")
        imglabs_path = test_labs_path + line.replace(".jpg\n",".txt")
        cv_image = cv2.imread(image_path)
        with open(imglabs_path, "r") as lp:
            for i in lp:
                i = i.split(' ')
                #print(i[0],i[1],i[2],i[3],i[4])
                cv2.rectangle(cv_image, (int(i[1]), int(i[2])), (int(i[3]), int(i[4])), (255, 0, 0), 2)
        cv2.imshow(line.strip("\n"), cv_image)
        #cv2.imwrite("C:\\Users\\shzhoujun\\Desktop\\testmodels\\GTimg\\" + line.strip("\n"),cv_image )
        cv2.waitKey(0)
