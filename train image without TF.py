# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:45:26 2020

@author: admin1
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten, Input

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
#for the next trial, please change the MobileNetV2 with VGG16
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras.utils import plot_model




datagen = ImageDataGenerator(rescale=1/255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

#train data
train_data= datagen.flow_from_directory('data/train',class_mode = 'binary', batch_size = 64, target_size = (224,224))
test_data= datagen.flow_from_directory('data/test',class_mode = 'binary', batch_size = 64, target_size = (224,224))
validation_data = datagen.flow_from_directory('data/validation',class_mode = 'binary', batch_size = 64,target_size = (224,224))

#define the model (with sequential model)
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape =(224,224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))  #why just one?
model.add(Activation('sigmoid'))


#model using functional API model

#inp = Input(shape=(224,224,3))
#headModel = Conv2D(32,(2,2),activation = 'relu')(inp)
#headModel = MaxPooling2D(pool_size=(2,2))(headModel)
#headModel = Conv2D(32,(2,2),activation = 'relu')(headModel)
#headModel = MaxPooling2D(pool_size=(2,2))(headModel)
#headModel = Conv2D(64,(2,2),activation = 'relu')(headModel)
#headModel = MaxPooling2D(pool_size=(2,2))(headModel)
#headModel = Flatten()(headModel)
#headModel = Dense(64,activation = 'relu')(headModel)
#headModel = Dropout(0.5)(headModel)
#headModel = Dense(1, activation = 'relu')(headModel)

#model = Model(inputs = inp, outputs = headModel, name = 'mask_model')

#model compiling

model.compile(loss= 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

model.fit_generator(train_data,steps_per_epoch = 32, validation_data = validation_data,validation_steps = 8)
#loss = model.evaluate_generator(test_data,steps = 24)

model.save('Output/img_classifier_w_TF.h5')

print(model.summary())


