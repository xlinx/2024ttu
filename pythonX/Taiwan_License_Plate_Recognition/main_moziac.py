import os
import re
import time
import cv2
import numpy as np
import datetime
import glob
from os.path import join
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
            allImages = []
            for ext in ('*.gif', '*.png', '*.jpg'):
                allImages.extend(glob.glob(att_dir_input+os.sep+ ext))
            # allImages=glob.glob(att_dir_input+os.sep+'*.jpg')
            print(datetime.datetime.now(), "[2][monitor][different]allImages=", allImages)
            for imagePathX in allImages:
                # step1_rawImageX = cv2.cvtColor(cv2.imread(imagePathX), cv2.COLOR_BGR2GRAY)
                # step7_Contours, hierarchyX = cv2.findContours(cv2.Canny(
                #     cv2.GaussianBlur(cv2.bilateralFilter(cv2.cvtColor(step1_rawImageX, cv2.COLOR_BGR2GRAY), 11, 17, 17), (5, 5),
                #                      0), 170, 200), cv2.RETR_LIST,
                #     cv2.CHAIN_APPROX_SIMPLE)  # 轉為灰階，去除背景雜訊，高斯模糊，取得邊緣，取得輪廓

                step1_rawImageX=cv2.imread(imagePathX)
                # step2_img_arr = np.array(step1_rawImageX)
                # step3_gray =step1_rawImageX
                # detect_plate(step1_rawImageX)

                step3_gray=cv2.cvtColor(step1_rawImageX, cv2.COLOR_RGB2GRAY)
                plt.imshow(step3_gray)
                plt.show()

                step4_bilateralFilter=cv2.bilateralFilter(step3_gray, 15, 17, 17)
                plt.imshow(step4_bilateralFilter)
                plt.show()

                # step5_GaussianBlur=cv2.GaussianBlur(step4_bilateralFilter, (5, 5),0)
                # plt.imshow(step5_GaussianBlur)
                # plt.show()

                # kernel = np.ones((7, 7), np.uint8)
                # step5_blackhat = cv2.morphologyEx(step4_bilateralFilter, cv2.MORPH_BLACKHAT, kernel)
                # plt.imshow(step5_blackhat)
                # plt.title("step5_blackhat")
                # plt.show()

                step6_Canny =cv2.Canny( step4_bilateralFilter, 170, 200)
                plt.imshow(step6_Canny)
                plt.title("step6_Canny")
                plt.show()

                step7_Contours, _ = cv2.findContours(step6_Canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


                rectangleContoursX = []
                working_mosaic(step7_Contours,rectangleContoursX,step1_rawImageX,imagePathX)
                os.remove(imagePathX)
        else:
            print(datetime.datetime.now(),"[][monitor][same]input=",att_dir_input)
        time.sleep(poll_timeout)


def detect_plate(image, lower=0, upper=20):
    image_contours = image.copy()
    blur = cv2.medianBlur(image_contours, 3)
    edges = cv2.Canny(blur.copy(), lower, upper)
    edges_3_channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # cv2.drawContours(image_contours, contours, -1, (0,255,0), 3)
    plate = []
    for cont in contours:
        peri = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.01 * peri, True)
        if len(approx) == 4:
            plate.append(approx)
            break
    if plate:
        cv2.drawContours(image_contours, plate, -1, (0, 255, 255), 3)
    output = np.concatenate([edges_3_channel, image_contours], axis=1)
    output = cv2.resize(output, (int(output.shape[1] * 0.6), int(output.shape[0] * 0.6)))
    plt.imshow(output)
    plt.title("detect_plate")
    plt.show()
    return output

def xAxB(answer,dect_num):
    print(f'[xAxB][1]{answer}/{dect_num} (target/dect_num) ')
    a = b  = 0
    user = list(dect_num)
    for i in range(min(len(answer), len(user))):
        if user[i] == answer[i]:
            a += 1
        else:
            if user[i] in answer:
                b += 1
    output = ','.join(user).replace(',', '')  # 四個數字都判斷後，使用 join 將串列合併成字串
    print(f'[xAxB][2]{answer}/{dect_num} (target/dect_num)=> {a}A{b}B')
    return [a,b]

def working_mosaic(contours,rectangleContours,rawImage,imagePath):
    target_number=os.path.splitext(os.path.basename(imagePath))[0].split("_")
    # regex = r"(.*)[.]"
    # target_number=re.search(regex,input_filename)
    print(datetime.datetime.now(), "[3][monitor][different]target_number=", target_number)

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:  # 只取前三十名輪廓
        approx=cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # 取得輪廓周長*0.02(越小，得到的多邊形角點越多)後，得到多邊形角點，為四邊形者
            # print(approx)
            rectangleContours.append(contour)


    for rectangleContour in rectangleContours:
        x, y, w, h = cv2.boundingRect(rectangleContour)  # 只取第一名，用一個最小的四邊形，把找到的輪廓包起來。
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
        #  Showing License Plate Number (optional)

        for i, j in enumerate(letter_images):
            plt.subplot(1, len(letter_images), i + 1)
            plt.imshow(letter_images[i], cmap='gray')
        plt.title("step_analy_letters")
        plt.show()

        results = []
        for index, letter_image in enumerate(letter_images):
            best_score = []
            patterns = os.listdir(patternsPath)
            for filename in patterns:  # read taiwan ocr letters folder #decade
                ret, pattern_img = cv2.threshold(cv2.cvtColor(cv2.imdecode(np.fromfile(patternsPath +os.sep+ filename, dtype=np.uint8), 1), cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_OTSU)  # 將範本進行格式轉換，再獲得Binary圖
                pattern_img = cv2.resize(pattern_img, (letter_image.shape[1], letter_image.shape[0]))  # 將範本resize至與圖像一樣大小
                best_score.append(cv2.matchTemplate(letter_image, pattern_img, cv2.TM_CCOEFF)[0][0])  # 範本匹配，返回匹配得分
            i = best_score.index(max(best_score))  # 取得最高分的index
            results.append(patterns[i])
        resultX="".join(results).replace(extension, "")
        only_filename, *_ = os.path.basename(imagePath).partition('.')
        target_number_arr=only_filename.split("_")
        output_filename="mosaic_"+os.path.basename(imagePath)
        print(datetime.datetime.now(),"[!!!] Found Number=",resultX,os.path.join(att_dir_output,output_filename))
        is_target_number = False
        for target_number in target_number_arr:
            ans=xAxB(target_number,resultX)
            if ans[0]>=3:
                is_target_number = True
        if is_target_number:
            res = cv2.rectangle(rawImage, (x,y), (x+w,y+h), (255,0, 0), 3)
            blured=cv2.blur(rawImage[y:y+h, x:x+w] ,(23,23))
            rawImage[y:y + h, x:x + w] = blured
        else:
            res = cv2.rectangle(rawImage, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # plt.imshow(cv2.cvtColor(rawImage, cv2.COLOR_BGR2RGB))

        plt.imshow(rawImage)
        plt.title("output_rawImage")
        plt.show()
        cv2.imwrite(os.path.join(att_dir_output,output_filename),rawImage)

directory_modified(att_dir_input, 5)