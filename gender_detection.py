from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cvlib as cv

import os
import cv2

import argparse

from keras.utils import get_file

line_arg = argparse.ArgumentParser()
line_arg.add_argument("-im_file","--input_image", required=True, help="image path required")

arguments = line_arg.parse_args()


model_path = "/Users/MRDOCC/PycharmProjects/ComputerVisionFinalProject/trained_detection_model.model"

image = cv2.imread(arguments.input_image)
cv_model = load_model(model_path)

shape, score = cv.detect_face(image) #imface, conf


gender_class = ["man", "woman"]

for i , z in enumerate(shape):
    (Xbegin , Ybegin) = z[0], z[1]
    (Xend, Yend) = z[2], z[3]


    cv2.rectangle(image, (Xbegin , Ybegin), (Xend, Yend), (0,255,0), 2)
    crop_imface = np.copy(image[Ybegin:Yend, Xbegin:Xend])

    crop_imface = cv2.resize(crop_imface, (96,96))
    resized = crop_imface.astype("float") / 255.0

    numpyed_img = img_to_array(resized)

    final_face = np.expand_dims(numpyed_img, axis=0)

    confidence_int = cv_model.predict(final_face)[0]

    print(confidence_int)
    print(gender_class)

    Y_end = Ybegin -10 if Ybegin-10 >10 else Ybegin +10

    i = np.argmax(confidence_int)

    gen_type = gender_class[i]

    gen_type = "{}- {:.2f}%".format(gen_type, confidence_int[i]*100)

    cv2.putText(image, gen_type, (Xbegin, Y_end), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                0.8, (0,255,0),2)



cv2.imshow("detect gender", image)
cv2.waitKey()
cv2.imwrite("gender.jpg", image)
cv2.destroyAllWindows()

