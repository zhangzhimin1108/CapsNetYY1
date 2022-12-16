from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.layers import Dropout, Activation
K.set_image_data_format('channels_last')
from capsulelayers import CapsuleLayer, CapsuleLayer_nogradient_stop, PrimaryCap, Length, Mask
from keras.engine.topology import Layer

class Extract_outputs(Layer):
    def __init__(self, outputdim, **kwargs):
        # self.input_spec = [InputSpec(ndim='3+')]
        self.outputdim = outputdim
        super(Extract_outputs, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[1], self.outputdim])

    def call(self, x, mask=None):
        x = x[:, :, :self.outputdim]
        # return K.batch_flatten(x)
        return x


def KNFC_model():
    K.clear_session()
    X1 = Input(shape=(256, 1))
    X2 = Input(shape=(256, 1))
    x1 = Conv1D(filters=128, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal',
                activation='relu')(X1)
    x1 = MaxPooling1D(pool_size=2, strides=2)(x1)
    x1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                activation='relu')(x1)
    x1 = MaxPooling1D(pool_size=2, strides=2)(x1)
    x2 = Conv1D(filters=128, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal',
                activation='relu')(X2)
    x2 = MaxPooling1D(pool_size=2, strides=2)(x2)
    x2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                activation='relu')(x2)
    x2 = MaxPooling1D(pool_size=2, strides=2)(x2)
    merge_layer = Concatenate(axis=1)([x1, x2])
    merge_layer = Dropout(0.5)(merge_layer)
    conv1 = Conv1D(filters=200, kernel_size=9, strides=1, padding='valid', kernel_initializer='he_normal',
                   activation='relu', name='conv1')(merge_layer)
    dt1 = Dropout(0.5)(conv1)
    primarycaps = PrimaryCap(dt1, dim_capsule=8, n_channels=6, kernel_size=20, kernel_initializer='he_normal',
                             strides=1, padding='valid', dropout=0.2)
    digitcaps_c = CapsuleLayer_nogradient_stop(num_capsule=2, dim_capsule=10, num_routing=3, name='digitcaps',
                                               kernel_initializer='he_normal', dropout=0.1)(primarycaps)
    digitcaps = Extract_outputs(10)(digitcaps_c)
    out_caps = Length()(digitcaps)
    model = Model([X1, X2], out_caps)
    return model
