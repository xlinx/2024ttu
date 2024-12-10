import cv2
import numpy as np
import os
import re
import time
import cv2
import numpy as np
import datetime
import glob
import easyocr
from matplotlib import pyplot as plt

reader=easyocr.Reader(['en'])
extension = '.jpg'
# imagePath = 'images/WM-8888.jpg'
patternsPath = 'solid_patterns'
att_dir_input = 'input'
att_dir_output = 'output'
if not os.path.exists(att_dir_input):
    os.makedirs(att_dir_input)
if not os.path.exists(att_dir_output):
    os.makedirs(att_dir_output)

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

def directory_modified(dir_path, poll_timeout=1):
    init_mtime = os.stat(dir_path).st_mtime
    while True:
        now_mtime = os.stat(dir_path).st_mtime
        if init_mtime != now_mtime:
            init_mtime = now_mtime
            print(datetime.datetime.now(),"[1][monitor][different]input=",att_dir_input,", output=",att_dir_output)
            allImages = []
            for ext in ('*.gif', '*.png', '*.jpg'):
                allImages.extend(glob.glob(att_dir_input+os.sep+ ext))
            # allImages=glob.glob(att_dir_input+os.sep+'*.jpg')
            print(datetime.datetime.now(), "[2][monitor][different]allImages=", allImages)
            for imagePathX in allImages:
                step1_rawImageX=cv2.imread(imagePathX)
                working_haar(step1_rawImageX,imagePathX)
                os.remove(imagePathX)
        else:
            print(datetime.datetime.now(),"[][monitor][same]input=",att_dir_input)
        time.sleep(poll_timeout)

def working_haar(rawImage,imagePath):
    plate_input_gray = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    # bfilter = cv2.bilateralFilter(plate_input_gray, 11, 17, 17)
    faces = faceCascade.detectMultiScale(plate_input_gray,scaleFactor=1.2,
        minNeighbors = 5, minSize=(25,25))

    for (x,y,w,h) in faces:
        cropped_image = plate_input_gray[ y:y+h + 1,x:x+w + 1]
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title("cropped_image")
        plt.show()
        result_array=reader.readtext(cropped_image)
        # result_text = result_array[0][-2]
        print(datetime.datetime.now(), "[1][ocr]", result_array)

        cv2.rectangle(plate_input_gray,(x,y),(x+w,y+h),(255,0,0),2)
        plate = plate_input_gray[y: y+h, x:x+w]
        plate = cv2.blur(plate,ksize=(20,20))
        plate_input_gray[y: y+h, x:x+w] = plate
    plt.imshow(plate_input_gray)
    plt.title("detect_gray_plate")
    plt.show()
directory_modified(att_dir_input, 5)
