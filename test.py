import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
import sys
from model import *

model.load_weights('train.h5')

print(sys.argv)

nom_fichier = sys.argv[1]

test_image = image.load_img('data/prediction/voiture_ou_avion.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image = test_image/255)
if result[0][0] == 1:
    prediction = 'avion'
else:
	prediction = 'voiture'

print(prediction)
