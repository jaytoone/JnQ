import pandas as pd
import numpy as np
import os


import keras
import tensorflow as tf


# tf_config = tf.ConfigProto()
# tf_config.gpu_options.allow_growth = True
# # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
# tf.keras.backend.set_session(tf.Session(config=tf_config))
#
# #           GPU Set         #
# tf.device('/device:XLA_GPU:0')

current_path = os.getcwd()
# print(current_path)
# quit()

#       load data       #
period = 45
x_save_path = os.path.join(current_path, 'npy', '%s_close_updown_x.npy' % period)
# print(x_save_path)
# quit()
y_save_path = os.path.join(current_path, 'npy', '%s_close_updown_y.npy' % period)
# y_save_path = current_path + 'npy/' + '%s_close_updown_y.npy' % period

total_x = np.load(x_save_path)
total_pr = np.load(y_save_path)

_, row, col, _ = total_x.shape
print("row, col :", row, col)

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight


seed = 1
random_state = 20
np.random.seed(seed)


#         train / test split      #
# x_train, x_test_, pr_train, pr_test_, ud_train, ud_test_ = train_test_split(re_total_x, total_pr, total_ud, test_size=0.4, shuffle=True, random_state=random_state)
x_train, x_test_, pr_train, pr_test_ = train_test_split(total_x, total_pr, test_size=0.4, shuffle=True, random_state=random_state)
x_test, x_val, pr_test, pr_val = train_test_split(x_test_, pr_test_, test_size=0.5, shuffle=True, random_state=random_state)
# break
#         pr label   #
y_train = np.where(pr_train > 1, 1, 0)
y_test = np.where(pr_test > 1, 1, 0)
y_val = np.where(pr_val > 1, 1, 0)

del total_x

print('pr_train[:5] :', pr_train[:5])
# print('ud_train[:5] :', ud_train[:5])
print('y_train[:5] :', y_train[:5])
print('y_train.dtype :', y_train.dtype)

print('x_train.shape :', x_train.shape)
print('x_test.shape :', x_test.shape)
print('x_val.shape :', x_val.shape)
print('y_train.shape :', y_train.shape)
print('y_test.shape :', y_test.shape)
print('y_val.shape :', y_val.shape)


def class_ratio(in_list):
    return in_list / in_list[1]


print('np.unique(y_train, return_counts=True :', np.unique(y_train, return_counts=True),
      class_ratio(np.unique(y_train, return_counts=True)[1]))
print('np.unique(y_val, return_counts=True :', np.unique(y_val, return_counts=True),
      class_ratio(np.unique(y_val, return_counts=True)[1]))
print('np.unique(y_test, return_counts=True :', np.unique(y_test, return_counts=True),
      class_ratio(np.unique(y_test, return_counts=True)[1]))

label = y_train.reshape(-1, )
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(label),
                                                  y=label)
class_weights = dict(enumerate(class_weights))
print('class_weights :', class_weights)

# sample_weight = np.ones(shape=(len(y_train),))
# sample_weight[(y_train == 1).reshape(-1,)] = 1.5
# print('sample_weight[:20] :', sample_weight[:20])

num_classes = 2

print('np.isnan(np.sum(x_train)) :', np.isnan(np.sum(x_train)))
print('np.isnan(np.sum(x_val)) :', np.isnan(np.sum(x_val)))
print('np.isnan(np.sum(x_test)) :', np.isnan(np.sum(x_test)))

print('np.isnan(np.sum(y_train)) :', np.isnan(np.sum(y_train)))
print('np.isnan(np.sum(y_val)) :', np.isnan(np.sum(y_val)))
print('np.isnan(np.sum(y_test)) :', np.isnan(np.sum(y_test)))

y_train_ohe = np_utils.to_categorical(y_train, num_classes)
y_val_ohe = np_utils.to_categorical(y_val, num_classes)
y_test_ohe = np_utils.to_categorical(y_test, num_classes)
print('y_train_ohe.shape :', y_train_ohe.shape)
print('y_val_ohe.shape :', y_val_ohe.shape)
print('y_test_ohe.shape :', y_test_ohe.shape)

datagen = ImageDataGenerator(
    rotation_range=45,
    # zoom_range = 0.5,
    # shear_range = 0.5,
    # horizontal_flip = True,
    # vertical_flip = True,
    # width_shift_range=0.5,
    # height_shift_range=0.5,
    # fill_mode = 'nearest'
)

valgen = ImageDataGenerator(
)

datagen.fit(x_train)
valgen.fit(x_val)

batch_size = 256
#
# for x_batch, _ in datagen.flow(x_train, y_train_ohe, batch_size=9):
#
#     plt.suptitle("train x_batch")
#
#     for i in range(0, 9):
#         plt.subplot(330 + 1 + i)
#         # resized = cv2.resize(x_batch[i].reshape(row, col), (row * 2, col * 10))
#         # cmapped = plt.cm.Set1(resized)
#         # plt.imshow(cmapped)
#         # plt.imshow(x_batch[i].reshape(row, col))
#         plt.imshow(x_batch[i])
#         plt.axis('off')
#     plt.show()
#     break
#
# for x_batch, _ in valgen.flow(x_val, y_val_ohe, batch_size=9):
#
#     plt.suptitle("val x_batch")
#
#     for i in range(0, 9):
#         plt.subplot(330 + 1 + i)
#         # resized = cv2.resize(x_batch[i].reshape(row, col), (row * 2, col * 10))
#         # cmapped = plt.cm.Set1(resized)
#         # plt.imshow(cmapped)
#         # plt.imshow(x_batch[i].reshape(row, col))
#         plt.imshow(x_batch[i])
#         plt.axis('off')
#     plt.show()
#     break

train_flow = datagen.flow(x_train, y_train_ohe, batch_size=batch_size)
val_flow = valgen.flow(x_val, y_val_ohe, batch_size=batch_size)
# break
print('train & val flow successfully made !')





from keras.optimizers import Adam, SGD
# from keras.regularizers import l1, l2

from msg_model import msg_model

(_, row, col, _) = x_train.shape

model = msg_model(input_shape=(row, col, 3))
opt = Adam(lr=0.00001, decay=0.000005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

ckpt_path = os.path.join(current_path, 'ckpt/')
board_path = os.path.join(current_path, 'graph/')
model_name = 'classifier_%s_close_updown_pr_500k.h5' % period

checkpoint = ModelCheckpoint(ckpt_path + model_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
checkpoint2 = TensorBoard(log_dir=board_path,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
checkpoint3 = EarlyStopping(monitor='val_loss', patience=40)
# callbacks_list = [checkpoint, checkpoint2, checkpoint3]
callbacks_list = [checkpoint, checkpoint2]

# keras.callbacks.Callback 로 부터 log 를 받아와 history log 를 작성할 수 있다.

# we iterate 200 times over the entire training set
num_epochs = 1000
history = model.fit_generator(train_flow,
                              steps_per_epoch=len(x_train) / batch_size,
                              epochs=num_epochs,
                              verbose=2,
                              callbacks=callbacks_list,
                              class_weight=class_weights,
                              validation_data=val_flow,
                              validation_steps=len(x_val) / batch_size,
                              shuffle=False)