import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time




cap = cv2.VideoCapture(0)
detectorH = HandDetector(maxHands=1)
classifier = Classifier("data/model/keras_model.h5","data/model/labels.txt")
offset = 20
imgSize = 300
folder = "data/N"
counter = 0
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","1","2","3","4","5","6","7","8","9"]

while True:
    succes, img = cap.read()
    imgOutPut = img.copy()
    hands , img = detectorH.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y- offset:y + h+offset , x-offset:x + w+offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h/w
        classifier.getPrediction(img)

        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
            print(prediction,index)

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[ hGap:hCal+hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)


        cv2.putText(imgOutPut, labels[index], (x,y-offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255),2)
        # cv2.rectangle(imgOutPut, (x -offset,y-offset),(x+w+offset, y+h+offset), (255,0,255),4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutPut)
    key = cv2.waitKey(1)












