import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from modelPseEIIP import get_model
import numpy as np
import pandas as pd
from keras import utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,matthews_corrcoef,accuracy_score,auc
from sklearn.metrics import average_precision_score, f1_score,recall_score,precision_recall_curve
from sklearn import metrics

fpr_list = []
tpr_list = []
plt.figure(figsize=(8, 8))
acc_score = []
auc_score = []
model=get_model()
model.load_weights("D:/PycharmProjects/pythonProject/CapsNetYY1/mode_weight/K562_weight/K562Model_PseEIIP.h5")
names = ['HCT116']
auc_savepath = 'D:/PycharmProjects/pythonProject/CapsNetYY1/roc.png'
coloer_list = ['r','orange','fuchsia','green','orange','khaki','mediumslateblue','c','pink','rosybrown']

for name in names:
    names = ['HCT116']
    name = names[0]
    m = pd.read_csv('D:/PycharmProjects/pythonProject/CapsNetYY1/feature encoding/PseEIIP/K562seq1test.csv', header=None)
    n = pd.read_csv('D:/PycharmProjects/pythonProject/CapsNetYY1/feature encoding/PseEIIP/K562seq2test.csv', header=None)
    Xtes_1 = np.array(m).reshape(3314, 64, 1)
    Xtes_2 = np.array(n).reshape(3314, 64, 1)
    y_tes_1 = pd.read_csv('D:/PycharmProjects/pythonProject/CapsNetYY1/feature encoding/k562labeltest.txt', header=None)
    y_tes= utils.to_categorical(y_tes_1, 2)
    pre_test_y = model.predict([Xtes_1, Xtes_2])
    test_auc = metrics.roc_auc_score(y_tes, pre_test_y)
    auc_score.append(test_auc)

    fpr, tpr, threshold = roc_curve(y_tes[:, -1], pre_test_y[:, -1])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_roc = metrics.roc_auc_score(y_tes, pre_test_y)
    #print('auc_roc', auc_roc, 'len(fpr)', len(fpr), 'tpr', tpr)
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 22,
            }
    lw = 1.5

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(fpr, tpr, color=coloer_list[0], lw=lw, label='EIIP' + ' (AUC=%0.4f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tick_params(labelsize=20)
    plt.xlabel('1-Specificity', font)
    plt.ylabel('Sensitivity', font)
    plt.title('HCT116', font)
    plt.legend(loc="lower right")
    plt.savefig(auc_savepath, dpi=350)

    y_pred = model.predict([Xtes_1, Xtes_2])
    Auc = roc_auc_score(y_tes, y_pred)
    pre, rec, _ = precision_recall_curve(y_tes[:, -1], y_pred[:, -1])
    aupr = auc(rec, pre)
    y_pred_1 = model.predict([Xtes_1, Xtes_2]).argmax(axis=1)
    #print(np.round(y_pred.reshape(-1)))
    #print(y_pred)
    #print(y_tes)
    acc = accuracy_score(y_tes_1, y_pred_1)
    sn = recall_score(y_tes_1, y_pred_1)
    mcc = matthews_corrcoef(y_tes_1, y_pred_1)
    tn, fp, fn, tp = confusion_matrix(y_tes_1, y_pred_1).ravel()
    sp = tn / (tn + fp)
    # print(y_tes)
    f1 = f1_score(y_tes.reshape(-1), np.round(y_pred.reshape(-1)))
    print('predict result')
    print("AUC : ", Auc)
    print("AUPR : ", aupr)
    print("f1_score", f1)
    print("Accuracy", acc)
    print("SN : ", sn)
    print("SP : ", sp)
    print("MCC : ", mcc)

