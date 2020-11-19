import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Input, Dense, Conv2D, Bidirectional, UpSampling2D, UpSampling3D, concatenate, LSTM
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, ReLU, Reshape, Permute, Lambda, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import he_normal, glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard


def AV_model():
    def UpSampling2DBilinear(size):
        return Lambda(lambda x: tf.image.resize(x, size))

    def sliced(x, index):
        return x[:, :, :, index]

    # --------------------------- AUDIO MODEL ---------------------------
    audio_input = Input(shape=(298, 257, 2))
    as_conv1 = Conv2D(96, kernel_size=(1, 7), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv1')(audio_input)
    as_conv1 = BatchNormalization()(as_conv1)
    as_conv1 = ReLU()(as_conv1)

    as_conv2 = Conv2D(96, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv2')(as_conv1)
    as_conv2 = BatchNormalization()(as_conv2)
    as_conv2 = ReLU()(as_conv2)

    as_conv3 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv3')(as_conv2)
    as_conv3 = BatchNormalization()(as_conv3)
    as_conv3 = ReLU()(as_conv3)

    as_conv4 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='as_conv4')(as_conv3)
    as_conv4 = BatchNormalization()(as_conv4)
    as_conv4 = ReLU()(as_conv4)

    as_conv5 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='as_conv5')(as_conv4)
    as_conv5 = BatchNormalization()(as_conv5)
    as_conv5 = ReLU()(as_conv5)

    as_conv6 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='as_conv6')(as_conv5)
    as_conv6 = BatchNormalization()(as_conv6)
    as_conv6 = ReLU()(as_conv6)

    as_conv7 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='as_conv7')(as_conv6)
    as_conv7 = BatchNormalization()(as_conv7)
    as_conv7 = ReLU()(as_conv7)

    as_conv8 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 1), name='as_conv8')(as_conv7)
    as_conv8 = BatchNormalization()(as_conv8)
    as_conv8 = ReLU()(as_conv8)

    as_conv9 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv9')(as_conv8)
    as_conv9 = BatchNormalization()(as_conv9)
    as_conv9 = ReLU()(as_conv9)

    as_conv10 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(2, 2), name='as_conv10')(as_conv9)
    as_conv10 = BatchNormalization()(as_conv10)
    as_conv10 = ReLU()(as_conv10)

    as_conv11 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(4, 4), name='as_conv11')(as_conv10)
    as_conv11 = BatchNormalization()(as_conv11)
    as_conv11 = ReLU()(as_conv11)

    as_conv12 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(8, 8), name='as_conv12')(as_conv11)
    as_conv12 = BatchNormalization()(as_conv12)
    as_conv12 = ReLU()(as_conv12)

    as_conv13 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(16, 16), name='as_conv13')(as_conv12)
    as_conv13 = BatchNormalization()(as_conv13)
    as_conv13 = ReLU()(as_conv13)

    as_conv14 = Conv2D(96, kernel_size=(5, 5), strides=(1, 1), padding='same', dilation_rate=(32, 32), name='as_conv14')(as_conv13)
    as_conv14 = BatchNormalization()(as_conv14)
    as_conv14 = ReLU()(as_conv14)

    as_conv15 = Conv2D(8, kernel_size=(1, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='as_conv15')(as_conv14)
    as_conv15 = BatchNormalization()(as_conv15)
    as_conv15 = ReLU()(as_conv15)

    AS_out = Reshape((298, 8 * 257))(as_conv15)

    # --------------------------- VS_model start ---------------------------
    VS_model = Sequential()
    VS_model.add(Conv2D(256, kernel_size=(7, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv1'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Conv2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(1, 1), name='vs_conv2'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Conv2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(2, 1), name='vs_conv3'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Conv2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(4, 1), name='vs_conv4'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Conv2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(8, 1), name='vs_conv5'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Conv2D(256, kernel_size=(5, 1), strides=(1, 1), padding='same', dilation_rate=(16, 1), name='vs_conv6'))
    VS_model.add(BatchNormalization())
    VS_model.add(ReLU())
    VS_model.add(Reshape((75, 256, 1)))
    VS_model.add(UpSampling2DBilinear((298, 256)))
    VS_model.add(Reshape((298, 256)))
    # --------------------------- VS_model end ---------------------------

    video_input = Input(shape=(75, 1, 1792, 2))
    AVfusion_list = [AS_out]
    for i in range(2):
        single_input = Lambda(sliced, arguments={'index': i})(video_input)
        VS_out = VS_model(single_input)
        AVfusion_list.append(VS_out)

    AVfusion = concatenate(AVfusion_list, axis=2)
    AVfusion = TimeDistributed(Flatten())(AVfusion)

    lstm = Bidirectional(LSTM(400, input_shape=(298, 8 * 257), return_sequences=True), merge_mode='sum')(AVfusion)

    fc1 = Dense(600, name="fc1", activation='relu', kernel_initializer=he_normal(seed=27))(lstm)
    fc2 = Dense(600, name="fc2", activation='relu', kernel_initializer=he_normal(seed=42))(fc1)
    fc3 = Dense(600, name="fc3", activation='relu', kernel_initializer=he_normal(seed=65))(fc2)

    complex_mask = Dense(257 * 2 * 2, name="complex_mask", kernel_initializer=glorot_uniform(seed=87))(fc3)

    complex_mask_out = Reshape((298, 257, 2, 2))(complex_mask)
    
    AV_model = Model(inputs=[audio_input, video_input], outputs=complex_mask_out)

    return AV_model