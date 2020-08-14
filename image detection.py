# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:04:02 2020

@author: admin1
"""
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
#qimport argparse
import imutils
import time
import cv2
import datetime
import os

def detect_and_predict_image(frame,faceNet,imageNet):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0, 177.0, 123.0))
    
    #pass the blob through the network
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
    locs = []
    preds = []
    
    for i in range(0,detections.shape[2]):
        #detect the confidence for the inout blob image
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)   #img_to_array is used for converting PIL image to array
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
    if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
        preds = imageNet.predict(faces)

	# return a 2-tuple of the face locations and their corresponding
	# locations
    return (locs, preds)

#loading face detector
print("[INFO] loading face detector model...")
prototxt_path = 'Face Detector/res10_300x300_ssd_iter_140000.caffemodel'
weights_path = 'Face Detector/deploy.prototxt'
faceNet = cv2.dnn.readNet(prototxt_path,weights_path)

#loading face mask detector from disk
imageNet = load_model('Output/img_classifier.h5')

#initialize video stream
vs = cv2.VideoCapture()
time.sleep(2.0)

while True :
    ret,frame = vs.read()
    frame = frame.resize(frame,(400,400))
    (locs,pred) = detect_and_predict_image(frame,faceNet,imageNet)
    
    for (box,pred) in zip(locs,preds):
        
    


        
    
    