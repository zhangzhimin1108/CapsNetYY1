from keras.layers import *
import pandas as pd
from keras.models import *
from keras import backend as K
from keras.layers import Dropout, Activation
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from sklearn import metrics
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,auc
from keras import utils
import matplotlib.pyplot as plt
K.set_image_data_format('channels_last')
from capsulelayers import CapsuleLayer, CapsuleLayer_nogradient_stop, PrimaryCap, Length, Mask

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

auc_savepath = 'D:/PycharmProjects/pythonProject/CapsNetYY1/roc.png'
coloer_list = ['b','orange','fuchsia','green','orange','khaki','mediumslateblue','c','pink','rosybrown']
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
names = ['HCT116',  'K562']
name=names[0]
m=pd.read_csv('D:/PycharmProjects/pythonProject/CapsNetYY1/feature encoding/MSA/HCT116_train.csv',header=None)
data=np.array(m).reshape(8384,675,1)
X_1=data[0:4192]
X_2=data[4192:8384]
y_tra=pd.read_csv('D:/PycharmProjects/pythonProject/CapsNetYY1/feature encoding/HCT116labeltrain.txt',header=None)
y_tra = utils.to_categorical(y_tra, 2)
fpr_list = []
tpr_list = []
plt.figure(figsize=(8,8))
acc_score = []
auc_score = []

for i,(train, test) in enumerate(kfold.split(y_tra)):
    print('\n\n%d'%i)
    #print(i,(train,test))
    path = 'D:/PycharmProjects/pythonProject/CapsNetYY1/MSA_h5/%sModel%d.h5' % (name, i)
    checkpoint = ModelCheckpoint(filepath=path,monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto')
    def get_model():
        K.clear_session()
        X1 = Input(shape=(675,1))
        X2 = Input(shape=(675,1))
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
        model.compile(optimizer=optimizers.Adam(lr=0.0005), loss=[margin_loss],
                      metrics=['accuracy'])
        return model

    print('Train...')

    callbacks_list = checkpoint
    back = EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='auto')
    model = None
    model = get_model()
    model.summary()
    history = model.fit([X_1[train], X_2[train]], y_tra[train], validation_data=([X_1[test], X_2[test]], y_tra[test]), epochs=200,
                        batch_size=128,callbacks=[checkpoint, back], verbose=2)
    model1 = get_model()
    model1.load_weights(path)
    '''for i in prd_acc:
            pre_acc2.append(i[0])'''

    pre_test_y = model1.predict([X_1[test], X_2[test]])
    '''prd_acc = model.predict([X_en_tra[test], X_pr_tra[test]])
    pre_acc2 = []'''
    test_auc = metrics.roc_auc_score(y_tra[test], pre_test_y)
    auc_score.append(test_auc)

    fpr, tpr, threshold = roc_curve(y_tra[test][:, -1], pre_test_y[:, -1])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_roc = metrics.roc_auc_score(y_tra[test], pre_test_y)
    print('auc_roc', auc_roc, 'len(fpr)', len(fpr), 'tpr', tpr)
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 22,
            }
    lw = 1.5

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(fpr, tpr, color=coloer_list[i], lw=lw, label='ROC fold' + str(i + 1) + '(AUC=%0.4f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tick_params(labelsize=20)
    plt.xlabel('1-Specificity', font)
    plt.ylabel('Sensitivity', font)
    plt.title('HCT116', font)
    plt.legend(loc="lower right")

    # plotROC(y_tra[test].argmax(axis=1),pre_test_y.argmax(axis=1),auc_path)
    score, acc = model1.evaluate([X_1[test], X_2[test]], y_tra[test])
    acc_score.append(acc)
    print('Test accuracy:', acc)
    print("test_auc: ", test_auc)
fpr_mean_list = []
tpr_mean_list = []
fpr_tpr_len = []
# print('fpr_list',fpr_list,'/n','fpr_list[0]',fpr_list[0])
for j in range(0, 10, 1):
    len_j = len(fpr_list[j])
    fpr_tpr_len.append(len_j)
length1 = np.min(np.array(fpr_tpr_len))
print('length1', length1)
for i in range(length1):
    fpr_mean = np.mean(np.array((fpr_list[0][i], fpr_list[1][i], fpr_list[2][i], fpr_list[3][i], fpr_list[4][i],
                                 fpr_list[5][i], fpr_list[6][i], fpr_list[7][i], fpr_list[8][i], fpr_list[9][i])))
    fpr_mean_list.append(fpr_mean)
for i in range(length1):
    tpr_mean = np.mean(np.array((tpr_list[0][i], tpr_list[1][i], tpr_list[2][i], tpr_list[3][i], tpr_list[4][i],
                                 tpr_list[5][i], tpr_list[6][i], tpr_list[7][i], tpr_list[8][i], tpr_list[9][i])))
    tpr_mean_list.append(tpr_mean)
fpr_mean = np.array(fpr_mean_list)
tpr_mean = np.array(tpr_mean_list)
auc_mean = np.mean(np.array(auc_score))

plt.plot(fpr_mean, tpr_mean, color='r', lw=lw, label='Mean ROC(AUC=%0.4f)' % auc_mean)
plt.legend(loc="lower right")
plt.savefig(auc_savepath, dpi=350)
plt.show()

print('***********************print final result*****************************')
print(auc_score)
mean_acc = np.mean(acc_score)
mean_auc = np.mean(auc_score)
# line = 'acc\tauc:\n%.2f\t%.4f' % (100 * mean_acc, mean_auc)
line = 'acc\tauc:\n%.2f\t%.4f' % (100 * mean_acc, mean_auc)
print('10-fold result:\n' + line)
