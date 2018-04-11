# jpg2png.py
import cv2
import numpy as np
import os

dir_jpg = "./data/weeblydata/"
dir_png = "./data/weeblydata2/"

files = os.listdir(dir_jpg)
# files2 = [files[i] for i in range(len(files)) if ".jpg" not in files[i] ]

i=1
for name in files[1:]:
    print(str(i))
    data=cv2.imread(dir_jpg+name+"/images/"+name+".jpg")
    output_dir_image=dir_png+name
    if not os.path.exists(output_dir_image):
        os.makedirs(output_dir_image)
        os.makedirs(output_dir_image+"/images/")
        os.makedirs(output_dir_image+"/masks/")
    cv2.imwrite(dir_png+name+"/images/"+name+".png", data)
    files2 = os.listdir(dir_jpg+name+"/masks/")
    for mask in files2:
        mask=mask.replace(".jpg","")
        data=cv2.imread(dir_jpg+name+"/masks/"+mask+".jpg")
        cv2.imwrite(dir_png+name+"/masks/"+mask+".png", data)
    i += 1
