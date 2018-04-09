# cell_seperate.py

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
import numpy as np
import os

# echo 'export PATH="/usr/local/opt/python@2/bin:$PATH"' >> ~/.bash_profile

dir = "/Users/april/Desktop/2018Spring/cell_nucleus/data/stage1_train/"
dir_gray = "/Users/april/Desktop/2018Spring/cell_nucleus/data/train_gray/"
dir_color = "/Users/april/Desktop/2018Spring/cell_nucleus/data/train_color/"
dir_data = "/Users/april/Desktop/2018Spring/cell_nucleus/data/"

def judge_gray(image):
### 1 is gray, 0 is not gray ###
    judge_matrix=np.asarray([[0,1,-1],[-1,0,1],[1,-1,0]])
    flag=1
    for x in image:
        for y in x:
            if (np.dot(np.asarray(y),judge_matrix)==np.asarray([0,0,0])).all() == False:
                flag=0
                break
    return flag
    
files = os.listdir(dir)

# images=[]
gray=[]
color=[]
i=1
for name in files[1:]:
    print(str(i))
    data=cv2.imread(dir+name+"/images/"+name+".png")
    # print 'Filename: %-25s %s' % (name, fnmatch.fnmatch(name, pattern))
    # images.append(data)
    if judge_gray(data)==1:
        print("1 "+name)
        gray.append(name)
        # cv2.imwrite(dir_gray+name+".png", data)
        # dir_from=dir+name
        with open(dir_data+"/dir_of_gray.list","a+") as f:
            f.write(dir+name+"\n")
    else:
        print("0 "+name)
        color.append(name)
        # cv2.imwrite(dir_color+name+".png", data)
        with open(dir_data+"/dir_of_color.list","a+") as f:
            f.write(dir+name+"\n")
    i += 1












