import os
import shutil
import random


train_path = './train'
test_path = './test'

subPath = ['drink','phone','smoke','yawn']

restoreFile = './'
for i in range(len(subPath)):
    train_path1 = os.path.join(train_path,subPath[i])
    if not os.path.exists(train_path1):
        raise Exception('error')
    restoreFile_train = os.path.join(restoreFile,'train.txt')
    with open(restoreFile_train,'a') as f:
        files = os.listdir(train_path1)
        for s in files:
            f.write(os.path.join(subPath[i],s)+' '+str(i)+'\n')

for i in range(len(subPath)):
    test_path1 = os.path.join(test_path,subPath[i])
    if not os.path.exists(test_path1):
        raise Exception('error')
    restoreFile_test = os.path.join(restoreFile,'test.txt')
    with open(restoreFile_test,'a') as f:
        files = os.listdir(test_path1)
        for s in files:
            f.write(os.path.join(subPath[i],s)+' '+str(i)+'\n')
