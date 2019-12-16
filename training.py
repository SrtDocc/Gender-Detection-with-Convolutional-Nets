from net_model.linearstack_layers import KerasSequential
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
import matplotlib.pyplot as pyplt
import numpy as np


cmd_arguments = argparse.ArgumentParser()
cmd_arguments.add_argument("-dset", "--dataset", required=True, help=" path to dataset required")

cmd_arguments.add_argument("-plt_accuracy", "--plot_acc", required=False, default="accuracy.png")
cmd_arguments.add_argument("-plt_loss", "--plot_loss", required= False, default="loss.png")

cmd_arguments.add_argument("-model_save", "--trained_model", type=str,
                      default="trained_detection_model.model")

parameters = cmd_arguments.parse_args()

train_images = []


for i in glob.glob(parameters.dataset + "/**/*", recursive=True):
    if not os.path.isdir(i):
        train_images.append(i)


random.seed(42)
random.shuffle(train_images)

dimensions_of_image = (96,96,3)

gen_types_list = []
data_list=[]




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



detection_model = KerasSequential.model(width=dimensions_of_image[0], height=dimensions_of_image[1], depth=dimensions_of_image[2],classes=2)



epoc = 100
batch = 64
training_model = detection_model.fit(x=XTrain, y=YTrain, batch_size=batch, epochs=epoc, verbose=1, validation_data=(XTest, YTest))
detection_model.save(parameters.trained_model)


pyplt.plot(training_model.history['accuracy'])
pyplt.plot(training_model.history['val_accuracy'])

pyplt.title('model accuracy')
pyplt.ylabel('accuracy')
pyplt.xlabel('epoch')
pyplt.legend(['train', 'test'], loc='upper left')
pyplt.savefig(parameters.plot_acc)
pyplt.show()


pyplt.plot(training_model.history['loss'])
pyplt.plot(training_model.history['val_loss'])
pyplt.title('model loss')
pyplt.ylabel('loss')
pyplt.xlabel('epoch')
pyplt.legend(['train', 'test'], loc='upper left')
pyplt.savefig(parameters.plot_loss)
pyplt.show()




















