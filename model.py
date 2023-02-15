from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.layers import Dropout, Activation
from keras.models import Model
import numpy as np
from keras import layers, optimizers
from layer import CapsuleLayer, PrimaryCap, Length

def model1():
    X1 = Input(shape=(506, 4))
    x1_1 = Conv1D(filters=32, kernel_size=7, strides=2, padding='same', activation='relu')(X1)
    x1_2 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu')(X1)
    x1_3 = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(X1)
    x1 = Concatenate(axis=-1)([x1_1, x1_2, x1_3])
    x1 = Conv1D(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu')(x1)
    X2 = Input(shape=(506, 4))
    x2_1 = Conv1D(filters=32, kernel_size=7, strides=2, padding='same', activation='relu')(X2)
    x2_2 = Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu')(X2)
    x2_3 = Conv1D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(X2)
    x2 = Concatenate(axis=-1)([x2_1, x2_2, x2_3])
    x2 = Conv1D(filters=32, kernel_size=5, strides=2, padding='valid', activation='relu')(x2)
    merge_layer = Concatenate(axis=1)([x1, x2])
    dt = Dropout(0.5)(merge_layer)
    dt = Bidirectional(GRU(32, return_sequences=True))(dt)
    primarycaps = PrimaryCap(dt, dim_vector=8, n_channels=16, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=2, dim_vector=8, num_routing=3, name='digitcaps')(primarycaps)
    out_caps = Length()(digitcaps)
    model = Model([X1, X2], out_caps)
    return model