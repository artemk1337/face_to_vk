from facenet_pytorch import MTCNN
import torch

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, PReLU, Flatten, Softmax
from tensorflow.keras.models import Model
import tensorflow as tf

import numpy as np


def build_pnet(weights, input_shape=None):
    if input_shape is None:
        input_shape = (None, None, 3)

    p_inp = Input(input_shape)

    init_kernel, init_bias = weights[0]
    p_layer = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(p_inp)
    init_kernel = weights[1]
    p_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(p_layer)
    p_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(p_layer)

    init_kernel, init_bias = weights[2]
    p_layer = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(p_layer)
    init_kernel = weights[3]
    p_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(p_layer)

    init_kernel, init_bias = weights[4]
    p_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(p_layer)
    init_kernel = weights[5]
    p_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(p_layer)

    init_kernel, init_bias = weights[6]
    p_layer_out1 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1),
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(p_layer)
    p_layer_out1 = Softmax(axis=3)(p_layer_out1)

    init_kernel, init_bias = weights[7]
    p_layer_out2 = Conv2D(4, kernel_size=(1, 1), strides=(1, 1),
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(p_layer)

    p_net = Model(p_inp, [p_layer_out2, p_layer_out1])

    return p_net


def build_rnet(weights, input_shape=None):
    if input_shape is None:
        input_shape = (24, 24, 3)

    r_inp = Input(input_shape)

    init_kernel, init_bias = weights[0]
    r_layer = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(r_inp)
    init_kernel = weights[1]
    r_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(r_layer)
    r_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(r_layer)

    init_kernel, init_bias = weights[2]
    r_layer = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(r_layer)
    init_kernel = weights[3]
    r_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(r_layer)
    r_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(r_layer)

    init_kernel, init_bias = weights[4]
    r_layer = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(r_layer)
    init_kernel = weights[5]
    r_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(r_layer)

    r_layer = Flatten()(r_layer)

    init_kernel, init_bias = weights[6]
    r_layer = Dense(128,
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(r_layer)
    init_kernel = weights[7]
    r_layer = PReLU(alpha_initializer=tf.keras.initializers.constant(init_kernel))(r_layer)

    init_kernel, init_bias = weights[8]
    r_layer_out1 = Dense(2,
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(r_layer)
    r_layer_out1 = Softmax(axis=1)(r_layer_out1)

    init_kernel, init_bias = weights[9]
    r_layer_out2 = Dense(4,
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(r_layer)

    r_net = Model(r_inp, [r_layer_out2, r_layer_out1])

    return r_net


def build_onet(weights, input_shape=None):
    if input_shape is None:
        input_shape = (48, 48, 3)

    o_inp = Input(input_shape)

    init_kernel, init_bias = weights[0]
    o_layer = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(o_inp)
    init_kernel = weights[1]
    o_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(o_layer)
    o_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(o_layer)

    init_kernel, init_bias = weights[2]
    o_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(o_layer)
    init_kernel = weights[3]
    o_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(o_layer)
    o_layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(o_layer)

    init_kernel, init_bias = weights[4]
    o_layer = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(o_layer)
    init_kernel = weights[5]
    o_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(o_layer)
    o_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(o_layer)

    init_kernel, init_bias = weights[6]
    o_layer = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), padding="valid",
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(o_layer)
    init_kernel = weights[7]
    o_layer = PReLU(shared_axes=[1, 2], alpha_initializer=tf.keras.initializers.constant(init_kernel))(o_layer)

    o_layer = Flatten()(o_layer)

    init_kernel, init_bias = weights[8]
    o_layer = Dense(256,
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(o_layer)
    init_kernel = weights[9]
    o_layer = PReLU(alpha_initializer=tf.keras.initializers.constant(init_kernel))(o_layer)

    init_kernel, init_bias = weights[10]
    o_layer_out1 = Dense(2,
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(o_layer)
    o_layer_out1 = Softmax(axis=1)(o_layer_out1)

    init_kernel, init_bias = weights[11]
    o_layer_out2 = Dense(4,
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(o_layer)

    init_kernel, init_bias = weights[12]
    o_layer_out3 = Dense(10,
                     kernel_initializer=tf.keras.initializers.constant(init_kernel),
                     bias_initializer=tf.keras.initializers.constant(init_bias))(o_layer)

    o_net = Model(o_inp, [o_layer_out2, o_layer_out3, o_layer_out1])
    return o_net


def transform_net(torch_model):

    # for i, layer in enumerate(tf_model.layers):
    #     print(i, layer.name, tuple(sublayer.shape for sublayer in layer.weights))

    print("Torch model:")
    for i, (k, v) in enumerate(torch_model.state_dict().items()):
        print(i, k, v.shape)

    torch_new_weights = []
    for i, (k, v) in enumerate(torch_model.state_dict().items()):
        if "conv" in k or "dense" in k:
            if "weight" in k:
                torch_new_weights += [[np.moveaxis(np.array(v), [0, 1], [-1, -2])]]
            elif "bias" in k:
                torch_new_weights[-1] += [np.array(v)]
        elif "prelu" in k:
            torch_new_weights += [np.expand_dims(np.array(v), axis=(0, 1))]

    print("\nTorch model transformed:")
    for i, v in enumerate(torch_new_weights):
        print(i, tuple(val.shape for val in v) if isinstance(v, list) else v.shape)

    return torch_new_weights


if __name__ == "__main__":
    torch_model = MTCNN(image_size=160)
    torch_pnet = torch_model.pnet
    torch_rnet = torch_model.rnet
    torch_onet = torch_model.onet

    torch_new_weights = transform_net(torch_pnet)
    tf_pnet = build_pnet(torch_new_weights)

    torch_new_weights = transform_net(torch_rnet)
    tf_rnet = build_rnet(torch_new_weights)

    torch_new_weights = transform_net(torch_onet)
    tf_onet = build_onet(torch_new_weights)
    # print()
    # for i, layer in enumerate(tf_rnet.layers):
    #     print(i, layer.name, tuple(sublayer.shape for sublayer in layer.weights))

