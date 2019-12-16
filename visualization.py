from net_model.linearstack_layers import KerasSequential
import matplotlib
import cv2
import argparse
import os
import glob
import random

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.preprocessing.image import img_to_array

from keras.utils import plot_model

import matplotlib.pyplot as pyplot

import numpy as np

cmd_arguments = argparse.ArgumentParser()
cmd_arguments.add_argument("-datasett", "--dataset", required=True, help=" path to dataset required")

parameters = cmd_arguments.parse_args()
dimensions_of_image = (96,96,3)
gen_types_list = []
data_list=[]

train_images = []


for i in glob.glob(parameters.dataset + "/**/*", recursive=True):
    if not os.path.isdir(i):
        train_images.append(i)



random.shuffle(train_images)


for im_instance in train_images:
    image = cv2.imread(im_instance)
    resized_image = cv2.resize(image, (dimensions_of_image[0], dimensions_of_image[1]))

    resized_image_array = img_to_array(resized_image)

    data_list.append(resized_image_array)

    gen_type = im_instance.split(os.path.sep)[-2]
    if gen_type =="man":
        gen_type=0
    else:
        gen_type=1

    gen_types_list.append([gen_type])



full_data = np.array(data_list, dtype="float") /255.0

gen_types_list = np.array(gen_types_list)

(XTrain, XTest, YTrain, YTest) =train_test_split(full_data, gen_types_list, test_size=0.2,
                                                  random_state=42)



YTest = to_categorical(YTest, num_classes=2)
YTrain = to_categorical(YTrain, num_classes=2)


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(XTrain[i])
pyplot.title("XTrain Dataset Example", loc="left")
pyplot.show()



for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(XTest[i])
pyplot.title("XTest Dataset Example", loc="left")
pyplot.show()


