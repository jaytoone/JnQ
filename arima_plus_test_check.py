import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time

import keras
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# print(tf.__version__)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.keras.backend.set_session(tf.Session(config=tf_config))

#           GPU Set         #
tf.device('/device:XLA_GPU:0')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import confusion_matrix

x_test = np.load('test_set/data/x_test.npy')
y_test = np.load('test_set/data/y_test.npy')
pr_test = np.load('test_set/data/pr_test.npy')

ckpt_path = "./test_set/model/"
model_name = "classifier_5_min_re.h5"  # <-- specifying model name
model_path = r'C:\Users\Lenovo\PycharmProjects\Project_System_Trading\Rapid_Ascend\test_set\model\classifier_45_min_pr_re.h5'

# model = keras.models.load_model(ckpt_path + model_name)
model = keras.models.load_model(model_path)

start_t = time.time()
test_result = model.predict(x_test)
# test_result = model.predict(test_set)

print('elapsed time : ', time.time() - start_t)

print('test_result.shape :', test_result.shape)
# print('pr_val.shape :', pr_val.shape)

y_score = test_result[:, [1]]
print('y_test[:5] :', y_test.reshape(-1, )[:5])
# print('np.unique(y_test) :', np.unique(y_test, return_counts=True))
print('y_score[:5] :', y_score[:5])
# print('np.unique(y_score) :', np.unique(y_score, return_counts=True))

print('y_test.shape :', y_test.shape)
print('y_score.shape :', y_score.shape)

print('len(y_test) :', len(y_test))

#     precision recall curve   #
precision, recall, threshold = precision_recall_curve(y_test, y_score)
precision, recall = precision[:-1], recall[:-1]

plt.plot(threshold, precision, label='precision')
plt.plot(threshold, recall, label='recall')
plt.legend()
plt.title('precision recall')
plt.show()
# print(y_pred)

# thresh = 0.19

# threshold = [thresh]
# print('threshold :', threshold)

acc_pr_bythr = []
for thresh in threshold:

    y_pred = np.where(y_score[:, -1] > thresh, 1, 0)
    print('y_pred.shape :', y_pred.shape)
    # print('y_pred :', y_pred)

    #     compare precision     #

    print('precision :', precision_score(y_test, y_pred))
    print('recall :', recall_score(y_test, y_pred))
    print()

    print('np.isnan(np.sum(x_test)) :', np.isnan(np.sum(x_test)))
    print('np.isnan(np.sum(y_test)) :', np.isnan(np.sum(y_test)))

    # plot_confusion_matrix(best_model, x_test, y_test, normalize=None)
    # plt.show()
    print()

    #     check win-ratio improvement     #
    cmat = confusion_matrix(y_test, y_pred)
    # print(cmat)
    # print(np.sum(cmat, axis=1))

    test_size = len(y_test)
    test_pr_list = pr_test
    print('origin ac_pr :', np.cumprod(test_pr_list)[-1])

    org_wr = np.sum(cmat, axis=1)[-1] / sum(np.sum(cmat, axis=1))
    ml_wr = cmat[1][1] / np.sum(cmat, axis=0)[-1]
    print('win ratio improvement %.2f --> %.2f' % (org_wr, ml_wr))

    # print('pr_test.shape :', pr_test.shape)

    # print(y_pred)
    # print(test_pr_list)
    pred_pr_list = np.where(y_pred == 1, test_pr_list.reshape(-1, ), 1.0)
    # print('pred_pr_list.shape :', pred_pr_list.shape)

    if np.cumprod(test_pr_list)[-1] < np.cumprod(pred_pr_list)[-1]:
        print('accum_pr increased ! : %.3f --> %.3f' % (np.cumprod(test_pr_list)[-1], np.cumprod(pred_pr_list)[-1]))
        print('thresh :', thresh)

        # # if len(threshold) == 1:
        # plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # plt.plot(np.cumprod(test_pr_list))
        # plt.title('%.3f' % (np.cumprod(test_pr_list)[-1]))
        # # plt.show()
        #
        # plt.subplot(122)
        # plt.plot(np.cumprod(pred_pr_list))
        # plt.title('%.3f' % (np.cumprod(pred_pr_list)[-1]))
        # plt.show()

    acc_pr_bythr.append(np.cumprod(pred_pr_list)[-1])

print('acc_pr_bythr :', acc_pr_bythr)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(threshold, precision, label='precision')
plt.plot(threshold, recall, label='recall')
plt.legend()
plt.title('precision recall')
# plt.show()
plt.subplot(122)
plt.plot(threshold, acc_pr_bythr)
plt.axhline(np.cumprod(test_pr_list)[-1], linestyle='--', color='r')
plt.show()