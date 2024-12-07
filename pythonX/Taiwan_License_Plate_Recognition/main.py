import os
import time
import cv2
import numpy as np
import datetime
import glob
from matplotlib import pyplot as plt
extension = '.jpg'
# imagePath = 'images/WM-8888.jpg'
patternsPath = 'solid_patterns'
att_dir_input = 'input'
att_dir_output = 'output'
if not os.path.exists(att_dir_input):
    os.makedirs(att_dir_input)
if not os.path.exists(att_dir_output):
    os.makedirs(att_dir_output)

def directory_modified(dir_path, poll_timeout=1):
    init_mtime = os.stat(dir_path).st_mtime
    while True:
        now_mtime = os.stat(dir_path).st_mtime
        if init_mtime != now_mtime:
            init_mtime = now_mtime
            print(datetime.datetime.now(),"[1][monitor][different]input=",att_dir_input,", output=",att_dir_output)
            allImages=glob.glob(att_dir_input+os.sep+'*.jpg')
            print(datetime.datetime.now(), "[2][monitor][different]allImages=", allImages)
            for imagePathX in allImages:

                rawImageX = cv2.cvtColor(np.array(cv2.imread(imagePathX)), cv2.COLOR_RGB2BGR)
                contoursX, hierarchyX = cv2.findContours(cv2.Canny(
                    cv2.GaussianBlur(cv2.bilateralFilter(cv2.cvtColor(rawImageX, cv2.COLOR_BGR2GRAY), 11, 17, 17), (5, 5),
                                     0), 170, 200), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 轉為灰階，去除背景雜訊，高斯模糊，取得邊緣，取得輪廓
                rectangleContoursX = []
                working_mosaic(contoursX,hierarchyX,rectangleContoursX,rawImageX,imagePathX)
                os.remove(imagePathX)
        else:
            print(datetime.datetime.now(),"[][monitor][same]input=",att_dir_input)
        time.sleep(poll_timeout)

def working_mosaic(contours,hierarchy,rectangleContours,rawImage,imagePath):
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:  # 只取前三十名輪廓
        if len(cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)) == 4:  # 取得輪廓周長*0.02(越小，得到的多邊形角點越多)後，得到多邊形角點，為四邊形者
            rectangleContours.append(contour)
    x, y, w, h = cv2.boundingRect(rectangleContours[0])  # 只取第一名，用一個最小的四邊形，把找到的輪廓包起來。
    ret, plateImage = cv2.threshold(cv2.cvtColor(cv2.GaussianBlur(rawImage[y:y + h, x:x + w], (3, 3), 0), cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_OTSU)  # 找到車牌後，由原來的圖截取出來，再將其高斯模糊以及取得灰階，再獲得Binary圖

    # 取出車牌文字 Getting License Plate Number
    contours, hierarchy = cv2.findContours(plateImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 取得車牌文字輪廓
    letters = []
    for contour in contours:  # 遍歷取得的輪廓
        rect = cv2.boundingRect(contour)
        if (rect[3] > (rect[2] * 1.5)) and (rect[3] < (rect[2] * 3.5) and (rect[2] > 10)):  # 過濾雜輪廓
            letters.append(cv2.boundingRect(contour))  # 存入過濾過的輪廓
    letter_images = []
    for letter in sorted(letters, key=lambda s: s[0], reverse=False):  # 重新安排號碼順序遍歷
        letter_images.append(plateImage[letter[1]:letter[1] + letter[3], letter[0]:letter[0] + letter[2]])  # 將過濾過的輪廓使用原圖裁切
    # show文字裁切成果(可選) Showing License Plate Number (optional)

    for i, j in enumerate(letter_images):
        plt.subplot(1, len(letter_images), i + 1)
        plt.imshow(letter_images[i], cmap='gray')
    plt.show()

    # 匹配車牌文字 Matching License Plate Number
    results = []
    for index, letter_image in enumerate(letter_images):
        best_score = []
        patterns = os.listdir(patternsPath)
        for filename in patterns:  # 讀取資料夾下所有的圖片
            ret, pattern_img = cv2.threshold(cv2.cvtColor(cv2.imdecode(np.fromfile(patternsPath +os.sep+ filename, dtype=np.uint8), 1), cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_OTSU)  # 將範本進行格式轉換，再獲得Binary圖
            pattern_img = cv2.resize(pattern_img, (letter_image.shape[1], letter_image.shape[0]))  # 將範本resize至與圖像一樣大小
            best_score.append(cv2.matchTemplate(letter_image, pattern_img, cv2.TM_CCOEFF)[0][0])  # 範本匹配，返回匹配得分
        i = best_score.index(max(best_score))  # 取得最高分的index
        results.append(patterns[i])
    resultX="".join(results).replace(extension, "")
    output_filename="mosaic_"+os.path.basename(imagePath)
    print(resultX,os.path.join(att_dir_output,output_filename))

    res = cv2.rectangle(rawImage, (x,y), (x+w,y+h), (0, 255, 0), 3)
    blured=cv2.blur(rawImage[y:y+h, x:x+w] ,(23,23))
    rawImage[y:y+h, x:x+w] = blured
    # plt.imshow(cv2.cvtColor(rawImage, cv2.COLOR_BGR2RGB))
    plt.imshow(rawImage)
    cv2.imwrite(os.path.join(att_dir_output,output_filename),rawImage)

directory_modified(att_dir_input, 5)