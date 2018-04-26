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
    smooth = 1e-12
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


def UNET_normalized(classes=2, initial_features=32, num_layers=5, loss="binary_crossentropy", optimizer=Adam(),
                    metrics=[jaccard_coef, dice_coef], channels=1):
    """
    This UNET implements the batch normalization after each of the convolutions
    :param classes: Number of classes used
    :param initial_features: Number of features used initially for the convolutions
    :param num_layers: Number of down and up sampling blocks
    :param loss: The type of loss used
    :param optimizer: The optimizer used for the gradient descent
    :param metrics: The metrics used to assess the performance
    :param channels: Number of channels of the input image
    :return: Returns a UNET model
    """
    layers = []
    layers.append(Input((None, None, channels)))
    # Downsampling
    for i in range(num_layers):
        if i != (num_layers - 1):
            # Define two convolutions and a max pooling
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            layers[-1] = BatchNormalization(axis=-1)(layers[-1])
            # layers[-1] = Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1])
            layers.append(MaxPooling2D(pool_size=(2, 2))(layers[-1]))
        else:
            # Define just two convolutions for the final layer
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            layers[-1] = BatchNormalization(axis=-1)(layers[-1])
            # layers[-1] = Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1])

    # Upsampling
    for j in range(num_layers - 1):
        # Concatenate with the appropriate output of a previous layer and the upsample
        layers.append(
            concatenate([UpSampling2D(size=(2, 2))(layers[-1]), layers[2 * num_layers - (2 * j) - 3]], axis=-1))
        # Add two extra convolutions
        layers.append(
            Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu', padding='same')(
                layers[-1]))
        layers[-1] = BatchNormalization(axis=-1)(layers[-1])

    # layers[-1] = Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu',
    #                        padding='same')(layers[-1])

    # Add the final sigmoid output
    layers.append(Conv2D(classes - 1, (1, 1), activation='sigmoid')(layers[-1]))

    # Compile the model
    model = Model(inputs=layers[0], outputs=layers[-1])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def UNET_standard(classes=2, initial_features=32, num_layers=5, loss="binary_crossentropy", optimizer=Adam(),
                    metrics=[jaccard_coef, dice_coef], channels=1):
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
    layers = []
    layers.append(Input((None, None, channels)))
    # Downsampling
    for i in range(num_layers):
        if i != (num_layers - 1):
            # Define two convolutions and a max pooling
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            layers[-1] = Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1])
            layers.append(MaxPooling2D(pool_size=(2, 2))(layers[-1]))
        else:
            # Define just two convolutions for the final layer
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            layers[-1] = Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1])

    # Upsampling
    for j in range(num_layers - 1):
        # Concatenate with the appropriate output of a previous layer and the upsample
        layers.append(
            concatenate([UpSampling2D(size=(2, 2))(layers[-1]), layers[2 * num_layers - (2 * j) - 3]], axis=-1))
        # Add two extra convolutions
        layers.append(
            Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu', padding='same')(
                layers[-1]))
        layers[-1] = Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu',
                            padding='same')(layers[-1])

    # Add the final sigmoid output
    layers.append(Conv2D(classes - 1, (1, 1), activation='sigmoid')(layers[-1]))

    # Compile the model
    print(layers)
    model = Model(inputs=layers[0], outputs=layers[-1])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def UNET_dropout(classes=2, initial_features=32, num_layers=5, loss="binary_crossentropy", optimizer=Adam(),
                    metrics=[jaccard_coef, dice_coef], channels=1,dropout_p = 1):
    """
    This UNET implements the model with dropout of the upsampling layers
    :param classes: Number of classes used
    :param initial_features: Number of features used initially for the convolutions
    :param num_layers: Number of down and up sampling blocks
    :param loss: The type of loss used
    :param optimizer: The optimizer used for the gradient descent
    :param metrics: The metrics used to assess the performance
    :param channels: Number of channels of the input image
    :return: Returns a UNET model
    """
    layers = []
    dropout = []
    layers.append(Input((None, None, channels)))
    # Downsampling
    for i in range(num_layers):
        if i != (num_layers - 1):
            # Define two convolutions and a max pooling
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            layers[-1] = Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1])
            layers.append(MaxPooling2D(pool_size=(2, 2))(layers[-1]))
        else:
            # Define just two convolutions for the final layer
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            layers[-1] = Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1])

    # Upsampling
    # Do a first dropout after the last maxpooling
    dropout.append(SpatialDropout2D(dropout_p)(layers[-1]))
    for j in range(num_layers - 1):
        # Concatenate with the appropriate output of a previous layer and the upsample
        layers.append(
            concatenate([UpSampling2D(size=(2, 2))(layers[-1]), layers[2 * num_layers - (2 * j) - 3]], axis=-1))
        # Dropout after concatenating
        dropout.append(SpatialDropout2D(dropout_p)(layers[-1]))
        # Add two extra convolutions
        layers.append(
            Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu', padding='same')(
                dropout[-1]))
        layers[-1] = Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu',
                            padding='same')(layers[-1])

    # Add the final sigmoid output
    layers.append(Conv2D(classes - 1, (1, 1), activation='sigmoid')(layers[-1]))

    # Compile the model
    model = Model(inputs=layers[0], outputs=layers[-1])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def UNET_norm_drop(classes=2, initial_features=32, num_layers=5, loss="binary_crossentropy", optimizer=Adam(),
         metrics=[jaccard_coef, dice_coef], channels=1,dropout_p = 1):
    """
        This UNET implements the model with dropout of the upsampling layers
        :param classes: Number of classes used
        :param initial_features: Number of features used initially for the convolutions
        :param num_layers: Number of down and up sampling blocks
        :param loss: The type of loss used
        :param optimizer: The optimizer used for the gradient descent
        :param metrics: The metrics used to assess the performance
        :param channels: Number of channels of the input image
        :return: Returns a UNET model
        """
    layers = []
    dropout = []
    layers.append(Input((None, None, channels)))
    # Downsampling
    for i in range(num_layers):
        if i != (num_layers - 1):
            # Define two convolutions and a max pooling
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            layers[-1] = BatchNormalization(axis=-1)(layers[-1])
            layers.append(MaxPooling2D(pool_size=(2, 2))(layers[-1]))
        else:
            # Define just two convolutions for the final layer
            layers.append(
                Conv2D(initial_features * (2 ** i), (3, 3), activation='relu', padding='same')(layers[-1]))
            layers[-1] = BatchNormalization(axis=-1)(layers[-1])
    # Upsampling
    # Do a first dropout after the last maxpooling
    dropout.append(SpatialDropout2D(dropout_p)(layers[-1]))
    for j in range(num_layers - 1):
        # Concatenate with the appropriate output of a previous layer and the upsample
        layers.append(
            concatenate([UpSampling2D(size=(2, 2))(layers[-1]), layers[2 * num_layers - (2 * j) - 3]], axis=-1))
        # Dropout after concatenating
        dropout.append(SpatialDropout2D(dropout_p)(layers[-1]))
        # Add two extra convolutions
        layers.append(
            Conv2D(initial_features * 2 ** (num_layers - 2 - j), (3, 3), activation='relu', padding='same')(
                dropout[-1]))
        layers[-1] = BatchNormalization(axis=-1)(layers[-1])

    # Add the final sigmoid output
    layers.append(Conv2D(classes - 1, (1, 1), activation='sigmoid')(layers[-1]))

    # Compile the model
    model = Model(inputs=layers[0], outputs=layers[-1])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
def UNET(classes=2, initial_features=32, num_layers=5, loss="binary_crossentropy", optimizer=Adam(),
         metrics=[jaccard_coef, dice_coef], channels=1,normalization = False,dropout_p = 1):
    # Initialize the proper list of layers
    if (normalization and dropout_p==1):
       """
       Case where we normalize the activations after each convolution and don't apply any dropout of the activations
       """
       model = UNET_normalized(classes,initial_features,num_layers,loss,optimizer,metrics,channels)
       return model
    if not(normalization) and (dropout_p ==1):
       # Case where we don't apply the normalization and we don't apply any dropout of the activations
       model = UNET_standard(classes,initial_features,num_layers,loss,optimizer,metrics,channels)
       return model
    if (not(normalization) and dropout_p!=1):
       # Case where we don't apply normalization but we only apply dropout in the upsampling
       model = UNET_dropout(classes,initial_features,num_layers,loss,optimizer,metrics,channels,dropout_p)
       return model
    if normalization and dropout_p!=1:
       model = UNET_norm_drop(classes,initial_features,num_layers,loss,optimizer,metrics,channels,dropout_p)
       return model



