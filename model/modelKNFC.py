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


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def get_model():
    K.clear_session()
    X1 = Input(shape=(1024, 1))
    X2 = Input(shape=(1024, 1))
    x1 = Conv1D(filters=128, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal',
                activation='relu')(X1)
    x1 = MaxPooling1D(pool_size=2, strides=2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                activation='relu')(x1)
    x1 = MaxPooling1D(pool_size=2, strides=2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    x2 = Conv1D(filters=128, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal',
                activation='relu')(X2)
    x2 = MaxPooling1D(pool_size=2, strides=2)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                activation='relu')(x2)
    x2 = MaxPooling1D(pool_size=2, strides=2)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    merge_layer = Concatenate(axis=1)([x1, x2])
    conv1 = Conv1D(filters=200, kernel_size=9, strides=1, padding='valid', kernel_initializer='he_normal',
                   activation='relu', name='conv1')(merge_layer)
    dt1 = Dropout(0.2)(conv1)
    primarycaps = PrimaryCap(dt1, dim_capsule=8, n_channels=6, kernel_size=20, kernel_initializer='he_normal',
                             strides=1, padding='valid', dropout=0.2)
    digitcaps_c = CapsuleLayer_nogradient_stop(num_capsule=2, dim_capsule=10, num_routing=3, name='digitcaps',
                                               kernel_initializer='he_normal', dropout=0.1)(primarycaps)
    digitcaps = Extract_outputs(10)(digitcaps_c)
    out_caps = Length()(digitcaps)
    model = Model([X1, X2], out_caps)
    model.compile(optimizer=optimizers.Adam(lr=0.0005), loss=[margin_loss],
                  metrics=['accuracy'])
    return model
