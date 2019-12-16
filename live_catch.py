import cv2
import cvlib
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model


path = "/Users/MRDOCC/PycharmProjects/ComputerVisionFinalProject/trained_detection_model.model"

loaded_m = load_model(path)

gender_class = ["man",  "woman"]

live_stream = cv2.VideoCapture()

while live_stream.isOpened():
    pos, cam = live_stream.read()

    shape, score = cvlib.detect_face(cam)

    for i, z in enumerate(shape):
        (Xbegin, Ybegin) = z[0], z[1]
        (Xend, Yend) = z[2], z[3]

        cv2.rectangle(cam, (Xbegin, Ybegin), (Xend, Yend), (0, 255, 0), 2)
        crop_imface = np.copy(cam[Ybegin:Yend, Xbegin:Xend])

        crop_imface = cv2.resize(crop_imface, (96, 96))
        resized = crop_imface.astype("float") / 255.0

        numpyed_img = img_to_array(resized)

        final_face = np.expand_dims(numpyed_img, axis=0)

        confidence_int = loaded_m.predict(final_face)[0]

        Y_end = Ybegin - 10 if Ybegin - 10 > 10 else Ybegin + 10

        i = np.argmax(confidence_int)

        gen_type = gender_class[i]

        gen_type = "{}- {:.2f}%".format(gen_type, confidence_int[i] * 100)

        cv2.putText(cam, gen_type, (Xbegin, Y_end), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("detected gender", cam)