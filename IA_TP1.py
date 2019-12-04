import matplotlib.pyplot as plt
#import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras import optimizers
from keras import models

from model import *

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
#plotImages(sample_training_images[:5])

model.compile(
              loss='binary_crossentropy',
              optimizer= 'sgd' ,
              metrics= ['accuracy']
               )

model.fit_generator(
        train_generator,
        steps_per_epoch=42,
        epochs=7,
        validation_data=validation_generator,
        validation_steps=200)

sample_training_images, class_ = next(train_generator)

model.save('train.h5')







        
