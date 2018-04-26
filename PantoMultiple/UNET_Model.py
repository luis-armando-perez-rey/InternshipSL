import keras
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, merge, concatenate, UpSampling2D, BatchNormalization
from keras.layers.core import SpatialDropout2D
from keras.models import Model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from Asbestos_Utils import name_list, load_image
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
from os.path import isfile, isdir, join, exists



def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    smooth = 1e-12;
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f)
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    # sum_ = K.sum(y_true_f)+K.sum(y_pred_f)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def dice_coef(y_true, y_pred):
    smooth = 1e-12
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def UNET(channels, classes , initial_features=32, num_layers=5, loss="binary_crossentropy", optimizer=Adam(),
                  metrics=[jaccard_coef, dice_coef], batch_normalization=False, dropout_type=0,
                  dropout_p=1.0):
    """
    This UNET implements the standard model
    :param classes: Number of classes used
    :param initial_features: Number of features used initially for the convolutions
    :param num_layers: Number of down and up sampling blocks
    :param loss: The type of loss used
    :param optimizer: The optimizer used for the gradient descent
    :param metrics: The metrics used to assess the performance
    :param channels: Number of channels of the input image
    :return: Returns a UNET model
    """
    # Initialize a list which will store some of the layers in the model
    layers = []
    layers.append(Input((None, None, channels)))
    # DOWNDSAMPLING
    for i in range(num_layers):

        # LAYER BLOCK
        if i != (num_layers - 1):
            # Define two convolutions and a max pooling
            layers.append(Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            # Add batch normalization after convolution
            if batch_normalization:
                layers[-1] = BatchNormalization(axis=-1)(layers[-1])
            layers[-1] = Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1])
            # BATCH NORMALIZATION
            if batch_normalization:
                layers[-1] = BatchNormalization(axis=-1)(layers[-1])
            layers.append(MaxPooling2D(pool_size=(2, 2))(layers[-1]))

        # FINAL LAYER
        else:
            # Define just two convolutions for the final layer
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            # BATCH NORMALIZATION
            if batch_normalization:
                layers[-1] = BatchNormalization(axis=-1)(layers[-1])
            layers[-1] = Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1])
            # BATCH NORMALIZATION
            if batch_normalization:
                layers[-1] = BatchNormalization(axis=-1)(layers[-1])
            # DROPOUT TYPE 1
            if dropout_type == 1:
                layers[-1] = SpatialDropout2D(dropout_p)(layers[-1])

    # UPSAMPLING
    for j in range(num_layers - 1):
        # Concatenate with the appropriate output of a previous layer and the upsample
        layers.append(
            concatenate([UpSampling2D(size=(2, 2))(layers[-1]), layers[2 * num_layers - (2 * j) - 3]], axis=-1))
        # DROPOUT TYPE 2
        if dropout_type == 2:
            layers[-1] = SpatialDropout2D(dropout_p)(layers[-1])
        # Add two extra convolutions
        layers.append(
            Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu', padding='same')(
                layers[-1]))
        # BATCH NORMALIZATION
        if batch_normalization:
            layers[-1] = BatchNormalization(axis=-1)(layers[-1])
        layers[-1] = Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu',
                            padding='same')(layers[-1])
        # BATCH NORMALIZATION
        if batch_normalization:
            layers[-1] = BatchNormalization(axis=-1)(layers[-1])

    # Add the final sigmoid output
    layers.append(Conv2D(classes - 1, (1, 1), activation='sigmoid')(layers[-1]))

    # Compile the model
    print(layers)
    model = Model(inputs=layers[0], outputs=layers[-1])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
