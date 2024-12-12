import os
import re
import time
import cv2
import numpy as np
import datetime
import glob
import easyocr
import pytesseract
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
import platform
from os.path import join
from matplotlib import pyplot as plt
yolo_model = YOLO('yolov8n.pt')
# yolo_model = YOLO("yolo11n.pt")
# yolo_model = YOLO("car_plate.pt")
# results = model.train(data='custom_dataset.yaml', epochs=100, imgsz=640)
# results = model.train(data='custom_dataset.yaml', epochs=100, imgsz=640, project='YOLOv8_training', name='experiment1')
# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
# # Train the model
# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)

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
                cv2_img_readed = cv2.imread(imagePathX)
                opt_A_rawImage_yolo = yolo_decade(cv2_img_readed,imagePathX)
                # opt_B_rawImage_yolo = openCV_algorithm_processing(cv2_img_readed,imagePathX)

                FINAL_RESULT=opt_A_rawImage_yolo
                print("FINAL_RESULT",FINAL_RESULT)
                for r in FINAL_RESULT:
                    working_with_filename_and_blur_it(r['PLATE_NUM'],imagePathX,cv2_img_readed,r['X'],r['Y'],r['W'],r['H'])
                os.remove(imagePathX)
        else:
            print(datetime.datetime.now(),"[][monitor][same]input=",att_dir_input)
        time.sleep(poll_timeout)

def openCV_algorithm_processing(rawImageX,imagePathX):
    # step2_img_arr = np.array(step1_rawImageX)
    # step3_gray =step1_rawImageX
    # detect_plate(step1_rawImageX)
    step3_gray = cv2.cvtColor(rawImageX, cv2.COLOR_RGB2GRAY)
    plt.imshow(step3_gray)
    plt.show()
    step4_bilateralFilter = cv2.bilateralFilter(step3_gray, 15, 17, 17)
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
    step6_Canny = cv2.Canny(step4_bilateralFilter, 170, 200)
    plt.imshow(step6_Canny)
    plt.title("step6_Canny")
    plt.show()
    step7_Contours, _ = cv2.findContours(step6_Canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectangleContoursX = []
    working_mosaic(step7_Contours, rectangleContoursX, rawImageX, imagePathX)


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

def ocr_x(plate_input_gray,x,y,w,h):
    cropped_image = plate_input_gray[y:y + h + 1, x:x + w + 1]
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("ocr_x_cropped_image")
    plt.show()
    result_array = reader.readtext(cropped_image)
    # result_text = result_array[0][-2]
    print(datetime.datetime.now(), "[1][ocr]", result_array)
    return result_array

def ocr_decade(rawImage,x,y,w,h):
    ret, plateImage = cv2.threshold(
        cv2.cvtColor(cv2.GaussianBlur(rawImage[y:y + h, x:x + w], (3, 3), 0), cv2.COLOR_RGB2GRAY), 0, 255,
        cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(plateImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 取得車牌文字輪廓
    letters = []
    for contour in contours:  # 遍歷取得的輪廓
        rect = cv2.boundingRect(contour)
        if (rect[3] > (rect[2] * 1.5)) and (rect[3] < (rect[2] * 3.5) and (rect[2] > 10)):  # 過濾雜輪廓
            letters.append(cv2.boundingRect(contour))  # 存入過濾過的輪廓
    letter_images = []
    for letter in sorted(letters, key=lambda s: s[0], reverse=False):  # 重新安排號碼順序遍歷
        letter_images.append(
            plateImage[letter[1]:letter[1] + letter[3], letter[0]:letter[0] + letter[2]])  # 將過濾過的輪廓使用原圖裁切
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
            ret, pattern_img = cv2.threshold(
                cv2.cvtColor(cv2.imdecode(np.fromfile(patternsPath + os.sep + filename, dtype=np.uint8), 1),
                             cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_OTSU)  # 將範本進行格式轉換，再獲得Binary圖
            pattern_img = cv2.resize(pattern_img, (letter_image.shape[1], letter_image.shape[0]))  # 將範本resize至與圖像一樣大小
            best_score.append(cv2.matchTemplate(letter_image, pattern_img, cv2.TM_CCOEFF)[0][0])  # 範本匹配，返回匹配得分
        i = best_score.index(max(best_score))  # 取得最高分的index
        results.append(patterns[i])
    resultX = "".join(results).replace(extension, "")
    print(datetime.datetime.now(), "[1][decade_hand_made_ocr]", resultX)
    return resultX

#yolo class names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
# 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
# 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
# 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
# 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
# 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
# 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
# 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
# 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
# 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
# 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
# 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
# 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
# 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
# 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
# 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.BaseTensor.shape
def yolo_decade(raw,imagePath):
    # yolo_results=yolo_model(imagePath,  classes=[2,3])
    yolo_results = yolo_model(imagePath)
    print(datetime.datetime.now(), "[1][yolo_decade]",
          # yolo_results[0].names,
          yolo_results[0].boxes.cls,
          # yolo_results[0].boxes.data,
          "[1][yolo_decade]conf=",yolo_results[0].boxes.conf,
          yolo_results[0].boxes.xywh,
          # yolo_results[0].plot()
          )
    boxes = yolo_results[0].boxes.xyxy
    result=[]
    for index,box in enumerate(boxes):
        # if yolo_results[0].boxes.conf[index] <0.8:
        #     continue

        # box=box.cpu().numpy()
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.rectangle(raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tmp = cv2.cvtColor(raw[y1:y2, x1:x2].copy(), cv2.COLOR_RGB2GRAY)
        license = pytesseract.image_to_string(tmp, lang='eng', config='--psm 11').replace(" ", "").replace("\n", "").replace("-", "").replace("]", "1")
        result.append({'PLATE_NUM': license, 'X': x1, 'Y': y1, 'W': abs(x2-x1), 'H': abs(y2-y1)})
        print(datetime.datetime.now(), "[2][ocr=pytesseract][yolo_decade][license]",license)
        # img = text(raw, license, (x1, y1 - 20), (0, 255, 0), 25)
        plt.imshow(tmp)
        plt.title("yolo_plate_decade")
        plt.show()
    # plt.subplot(2,3,i+1)
    # plt.axis("off")
    # plt.imshow(raw)
    # for result in yolo_results:
    #     result.show()
    # annotated_frame=result[0].plot
    # cv2.imshow("yolo_decade",annotated_frame)
    plt.imshow(X=yolo_results[0].plot()[:,:,::-1])
    plt.title("yolo_decade")
    plt.show()

    return result
    # cv2.waitKey(0)

def working_with_filename_and_blur_it(resultX, imagePath,rawImage,x, y, w, h):
    only_filename, *_ = os.path.basename(imagePath).partition('.')
    target_number_arr = only_filename.split("_")
    output_filename = "mosaic_" + os.path.basename(imagePath)
    print(datetime.datetime.now(), "[!!!] Found Number=", resultX, os.path.join(att_dir_output, output_filename))
    is_target_number = False
    for target_number in target_number_arr:
        ans = xAxB(target_number, resultX)
        if ans[0] >= 3:
            is_target_number = True
    if is_target_number:
        res = cv2.rectangle(rawImage, (x, y), (x + w, y + h), (255, 0, 0), 3)
        blured = cv2.blur(rawImage[y:y + h, x:x + w], (23, 23))
        rawImage[y:y + h, x:x + w] = blured
    else:
        res = cv2.rectangle(rawImage, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(os.path.join(att_dir_output, output_filename), rawImage)

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
        # resultX1=ocr_x(rawImage,x, y, w, h)
        # resultX2=ocr_decade(rawImage,x, y, w, h)

def text(img, text, xy=(0, 0), color=(0, 0, 0), size=20):
    pil = Image.fromarray(img)
    s = platform.system()
    if s == "Linux":
        font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size)
    elif s == "Darwin":
        font = ImageFont.truetype('/Library/Fonts/SourceCodePro-Bold.ttf', size)
    else:
        font = ImageFont.truetype('simsun.ttc', size)
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=color)
    return np.asarray(pil)
    #find {/System,}/Library/Fonts -name \*ttf
        # working_with_filename(resultX2,imagePath, rawImage, x, y, w, h)
        # plt.imshow(rawImage)
        # plt.title("output_rawImage")
        # plt.show()


directory_modified(att_dir_input, 5)