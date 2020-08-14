# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:55:03 2020

@author: admin1
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

import cv2

img = load_img('test2.jpg',target_size = (224,224))
img = img_to_array(img)
img = np.expand_dims(img, axis = 0) #expand dims is used for 4 input tupples
#img = img_to_array(img)
#img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])

maskNet = load_model('Output/img_classifier_w_TF.h5')

#img = img.resize()
pred = maskNet.predict_classes(img)

#decode the class