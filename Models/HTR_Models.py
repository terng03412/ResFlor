
import tensorflow as tf
import numpy as np

import cv2
import string
import h5py

keras = tf.keras
print(tf.__version__)


class FullGatedConv2D(keras.layers.Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""
        output = super(FullGatedConv2D, self).call(inputs)
        linear = keras.layers.Activation("linear")(
            output[:, :, :, :self.nb_filters])
        sigmoid = keras.layers.Activation("sigmoid")(
            output[:, :, :, self.nb_filters:])

        return keras.layers.Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""
        output_shape = super(
            FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters,)

    def get_config(self):
        """Return the config of the layer"""
        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config


def FlorHTR(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)
    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def ExtendFlorHTR(input_shape, output_shape):
    input_data = keras.layers.Input(name="input", shape=input_shape)
    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = FullGatedConv2D(filters=64, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    cnn = keras.layers.Conv2D(filters=72, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)


def FlorResAcHTR(input_shape, output_shape):

    # https://arxiv.org/pdf/1512.03385.pdf
    input_data = keras.layers.Input(name="input", shape=input_shape)
    cnn = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(
        2, 2), padding="same", kernel_initializer="he_uniform")(input_data)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn1 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn1)
    res1 = keras.layers.add([cnn1, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res1)

    cnn = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn2 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn2)
    res2 = keras.layers.add([cnn2, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res2)

    cnn = keras.layers.Conv2D(filters=40, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn3 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn3)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res3 = keras.layers.add([cnn3, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res3)

    cnn = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn4 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn4)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res4 = keras.layers.add([cnn4, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res4)

    cnn = keras.layers.Conv2D(filters=56, kernel_size=(2, 4), strides=(
        2, 4), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn5 = keras.layers.BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(
        3, 3), padding="same", kernel_constraint=keras.constraints.MaxNorm(4, [0, 1, 2]))(cnn5)
    cnn = keras.layers.Dropout(rate=0.2)(cnn)
    res5 = keras.layers.add([cnn5, cnn])

    rac = keras.layers.PReLU(shared_axes=[1, 2])(res5)

    cnn = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(
        1, 1), padding="same", kernel_initializer="he_uniform")(rac)
    cnn = keras.layers.PReLU(shared_axes=[1, 2])(cnn)
    cnn = keras.layers.BatchNormalization(renorm=True)(cnn)

    cnn = keras.layers.MaxPooling2D(pool_size=(
        1, 2), strides=(1, 2), padding="valid")(cnn)

    shape = cnn.get_shape()
    bgru = keras.layers.Reshape((shape[1], shape[2] * shape[3]))(cnn)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    bgru = keras.layers.TimeDistributed(keras.layers.Dense(units=128))(bgru)

    bgru = keras.layers.Bidirectional(keras.layers.GRU(
        units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = keras.layers.TimeDistributed(
        keras.layers.Dense(units=output_shape, activation="softmax"))(bgru)
    return (input_data, output_data)
