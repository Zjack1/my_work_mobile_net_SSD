import os
import cv2
test_dir = "C:\\Users\\shzhoujun\\Pictures\\QQplayerPic"
detect_file = "C:\\Users\\shzhoujun\\Desktop\\detect.txt"
files = os.listdir(test_dir)
files.sort()
#print(len(files))
fo = open(detect_file, "r")
j=0
box = []
for line in fo.readlines():
    line = line.strip("(").strip().strip(")")
    line_1 = line.split(",")
#    for i in range(len(files)):
    if line_1[0].find("-",1) == True:
        j=j+1
        if j==1:
            continue
        origimg = cv2.imread("C:\\Users\\shzhoujun\\Pictures\\QQplayerPic\\" + str(j-1) + ".png")
        origimg = origimg[(int)(1080 * 0.3): 1080]
        for n in range(len(box)//4):
            cv2.rectangle(origimg, (box[0+n*4], box[1+n*4]), (box[2+n*4],box[3+n*4]), (0, 255, 0), 2)
        cv2.imshow("image", origimg )
        m = cv2.waitKey(0)

        box = []
        continue
    else:


        for k in range(len(line_1)-2):
            box.append(int(line_1[2+k]))


fo.close()
