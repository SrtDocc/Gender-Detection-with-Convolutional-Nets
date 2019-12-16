from keras.models import Sequential as Seq_model
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
from keras.layers.normalization import *
from keras.layers.convolutional import MaxPooling2D
from keras import backend as keras_back
from keras.layers.core import Flatten
from keras.optimizers import Adam

class KerasSequential:
    @staticmethod
    def model(width, height, depth, classes):
        image_shape = (depth, height, width)
        ch_dimension = 1

        if keras_back.image_data_format() == "channels_last":
            image_shape = (height, width, depth)
            ch_dimension = -1

        # Network Creation Notes
        """
        conv2d args: 32= number of filters, (x,x) = strides, padding = i/o size dealer,
                    input_shape = input shape of a given instance or new instance,
                    Activation = Activation function from keras.layer cores,
                    MaxPooling2D = re-size the input image, dropout = applies dropout to the input.

        """

        net_model = Seq_model([Conv2D(32, (3,3), padding="same", input_shape=image_shape),
                               Activation("relu"), BatchNormalization(axis=ch_dimension),
                               MaxPooling2D(pool_size=(2, 2)), Dropout(0.25),

                               Conv2D(64, (3,3), padding="same"), Activation("relu"),
                               BatchNormalization(axis=ch_dimension),
                               Conv2D(64, (3,3), padding="same"), Activation("relu"),
                               BatchNormalization(axis=ch_dimension),
                               MaxPooling2D(pool_size=(2, 2)), Dropout(0.25),

                               Conv2D(128, (3,3), padding="same"), Activation("relu"),
                               BatchNormalization(axis=ch_dimension),
                               Conv2D(128, (3,3), padding="same"), Activation("relu"),
                               BatchNormalization(axis=ch_dimension),
                               MaxPooling2D(pool_size=(2, 2)), Dropout(0.25),


                               Flatten(), Dense(1024), Activation("relu"), BatchNormalization(),
                               Dropout(0.5), Dense(classes), Activation("sigmoid")
                               ])
        learning_rate = 1e-3
        epoc = 100
        optimizer = Adam(lr=learning_rate, decay=learning_rate / epoc)
        net_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        return net_model