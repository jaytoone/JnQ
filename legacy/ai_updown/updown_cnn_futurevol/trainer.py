from updown_rnn.utils import *
from updown_rnn.model import lstm_model
# from binance_futures_concat_candlestick import concat_candlestick
from funcs.olds.funcs_indicator_candlescore import *
from funcs.funcs_trader import min_max_scaler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from sklearn.utils import class_weight
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_curve
from datetime import datetime

import keras.backend as K

import time

seed = 1
random_state = 201
np.random.seed(seed)

num_classes = 2


def class_ratio(in_list):
    return in_list / in_list[1]


#       goal : filling empty pred_result in stacked_df(ohlcv)      #
def retrain_and_pred(stacked_df, symbol, interval=30, period=45, lb_period=100, leverage=5, long_index=0):

    #       len(ohlcv) should be more than 8000 + lb_period (some rows would be removed by period)     #
    ohlcv = stacked_df
    # ohlcv = pd.read_excel(df_path, index_col=0)

    predictions = ohlcv['close'].shift(1).values
    err_ranges = np.zeros_like(predictions)

    try:
        print('-------------- %s --------------' % symbol)
        result = get_back_result(ohlcv, predictions, err_ranges, tp=0, leverage=leverage, show_plot=False,
                                 reverse_short=False, show_detail=False)
        # temp_ap_list.append(result[2])
        # temp_pr_list.append(result[3])

        # if round(leverage) == 1:
        #   temp_pr_list = result[3]
        pr_list = result[3][long_index]

    except Exception as e:
        print(e)
        return

    #       scale with price    #
    # time_index = ohlcv.index[-len(predictions):]
    sliced_ohlcv = ohlcv[-len(predictions):]

    ohlc = ohlcv.iloc[-len(predictions):, :4]
    sar = lucid_sar(sliced_ohlcv)
    senkou1, senkou2 = ichimoku(sliced_ohlcv)

    data_x, data_pr = [], []
    pred_data_x = []
    plotting = True

    #       stack 여부에 상관없이 new label 을 수집하기 위해
    #       we need last row data for prediction       #
    for i in range(period, len(ohlc) + 1):

        #   pr_list != 1 인 데이터만 사용
        if pr_list[i] != 1:
            pass
        else:
            if i == len(ohlc):  # last row handicap
                pass
            else:
                continue

        #   prediction 을 제외한 이전 데이터를 사용해야한다
        temp_ohlc = ohlc.iloc[i - period: i].values
        temp_sar = sar.iloc[i - period: i].values.reshape(-1, 1)
        temp_senkou1 = senkou1.iloc[i - period: i].values.reshape(-1, 1)
        temp_senkou2 = senkou2.iloc[i - period: i].values.reshape(-1, 1)

        price_data = np.hstack((temp_ohlc, temp_sar, temp_senkou1, temp_senkou2))

        if np.isnan(np.sum(price_data)):
            continue

        if plotting:
            plt.plot(price_data)
            plt.show()

        temp_price_data = min_max_scaler(price_data)
        # temp_price_data = (price_data - np.min(price_data)) / (np.max(price_data) - np.min(price_data))

        temp_ohlc = temp_price_data[:, :4]

        if plotting:
            plt.plot(temp_price_data)
            plt.show()

            plotting = False

        #                   feature selection                   #
        temp_data = temp_ohlc[:, [3]]

        if np.isnan(np.sum(temp_data)):
            continue

        if i != len(ohlc):
            data_x.append(temp_data)
            data_pr.append(pr_list[i])
        else:
            pred_data_x.append(temp_data)

    print('np.array(data_x).shape :', np.array(data_x).shape)
    # print(data_x[0])

    #       Reshape data for image deep - learning     #
    _, row, col = np.array(data_x).shape

    input_x = np.array(data_x).reshape(-1, row, col).astype(np.float32)
    pred_x = np.array(pred_data_x).reshape(-1, row, col).astype(np.float32)
    input_pr = np.array(data_pr).reshape(-1, 1).astype(np.float32)
    print('input_x.shape :', input_x.shape)
    print('input_x.dtype :', input_x.dtype)
    print('input_pr.shape :', input_pr.shape)

    total_x = input_x
    total_pr = input_pr

    _, row, col = input_x.shape

    model_name = 'classifier_%s_lstm_close_updown_pr_retrain_%s_timesplit.h5' % (period, symbol)

    t_result_path = 'npy/' + '%s_rnn_close_updown_t_result_%s.npy' % (period, symbol)
    pr_list_path = 'npy/' + '%s_rnn_close_updown_pr_list_%s.npy' % (period, symbol)

    #       ----------- 여기까지 non_stack / on_stack 동일하게 진행함  -----------     #
    #       stack file 이 없거나, update 시기가 오래된 경우 --> non_stack
    non_stack = False
    try:
        with open('updown_rnn/%s_ai_survey_log.txt' % symbol, 'r') as log_file:
            logged_timestamp = float(log_file.readline())
            # print(float(log_file.readline()))
    except Exception as e:
        print("Error in load timestamp log :", e)
        non_stack = True

    if datetime.now().timestamp() - logged_timestamp > interval * 60 * 2:
        non_stack = True

    #       1. get_best_thr 을 위한, total_result 형성       #
    if non_stack:
        non_stack_retrain(8000, lb_period, total_x, total_pr, model_name, t_result_path, pr_list_path)

    return on_stack_retrain(total_x, total_pr, pred_x, model_name)


def non_stack_retrain(tv_length, lb_period, total_x, total_pr, model_name, t_result_path, pr_list_path):
    start_time = time.time()

    x_train = total_x[-tv_length - lb_period:-lb_period]
    pr_train = total_pr[-tv_length - lb_period:-lb_period]
    x_test = total_x[-lb_period:]
    pr_test = total_pr[-lb_period:]

    total_result = []
    pr_list = []

    _, row, col, = x_train.shape

    for r_i in tqdm(range(len(x_test))):

        if r_i < len(total_result):
            continue

        new_x_test = x_test[[r_i]]
        new_pr_test = pr_test[[r_i]]

        ind_x_train = np.vstack((x_train[r_i:], x_test[:r_i]))
        ind_pr_train = np.vstack((pr_train[r_i:], pr_test[:r_i]))

        #     add & remove test set   #
        new_x_train, new_x_val, new_pr_train, new_pr_val = train_test_split(ind_x_train, ind_pr_train, test_size=0.25,
                                                                            shuffle=True, random_state=random_state)

        new_y_train = np.where(new_pr_train > 1, 1, 0)
        new_y_val = np.where(new_pr_val > 1, 1, 0)
        new_y_test = np.where(new_pr_test > 1, 1, 0)

        print('new_x_train.shape :', new_x_train.shape)
        print('new_x_test.shape :', new_x_test.shape)
        print('new_x_val.shape :', new_x_val.shape)
        print('new_y_train.shape :', new_y_train.shape)
        print('new_y_test.shape :', new_y_test.shape)
        print('new_y_val.shape :', new_y_val.shape)

        print('np.unique(y_train, return_counts=True :', np.unique(new_y_train, return_counts=True),
              class_ratio(np.unique(new_y_train, return_counts=True)[1]))
        print('np.unique(new_y_val, return_counts=True :', np.unique(new_y_val, return_counts=True),
              class_ratio(np.unique(new_y_val, return_counts=True)[1]))
        # print('np.unique(new_y_test, return_counts=True :', np.unique(new_y_test, return_counts=True),
        #       class_ratio(np.unique(new_y_test, return_counts=True)[1]))

        label = new_y_train.reshape(-1, )
        class_weights = class_weight.compute_class_weight('balanced',
                                                          classes=np.unique(label),
                                                          y=label)
        class_weights = dict(enumerate(class_weights))
        print('class_weights :', class_weights)

        print('np.isnan(np.sum(x_train)) :', np.isnan(np.sum(new_x_train)))
        print('np.isnan(np.sum(new_x_val)) :', np.isnan(np.sum(new_x_val)))
        print('np.isnan(np.sum(new_x_test)) :', np.isnan(np.sum(new_x_test)))

        print('np.isnan(np.sum(new_y_train)) :', np.isnan(np.sum(new_y_train)))
        print('np.isnan(np.sum(new_y_val)) :', np.isnan(np.sum(new_y_val)))
        print('np.isnan(np.sum(new_y_test)) :', np.isnan(np.sum(new_y_test)))

        new_y_train_ohe = np_utils.to_categorical(new_y_train, num_classes)
        new_y_val_ohe = np_utils.to_categorical(new_y_val, num_classes)
        new_y_test_ohe = np_utils.to_categorical(new_y_test, num_classes)
        print('new_y_train_ohe.shape :', new_y_train_ohe.shape)
        print('new_y_val_ohe.shape :', new_y_val_ohe.shape)
        print('new_y_test_ohe.shape :', new_y_test_ohe.shape)

        ckpt_path = 'ckpt/'
        # board_path = 'graph/'

        #       retraining    #
        if r_i != 0:

            K.clear_session()  # added for training delation

            try:
                model = load_model(ckpt_path + model_name)
                print("----------------------- model loaded ! -----------------------")

                for l_i, layer in enumerate(model.layers):

                    if l_i != len(model.layers) - 1:
                        layer.trainable = False

            except Exception as e:
                print('Error in load_model :', e)
                model = lstm_model(input_shape=(row, col))

        else:
            model = lstm_model(input_shape=(row, col))

        batch_size = 512
        # print("row, col :", row, col)

        opt = Adam(lr=0.00001, decay=0.000005)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        checkpoint = ModelCheckpoint(ckpt_path + model_name, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='auto')
        # checkpoint2 = TensorBoard(log_dir=board_path,
        #                           histogram_freq=0,
        #                           write_graph=True,
        #                           write_images=True)
        checkpoint3 = EarlyStopping(monitor='val_loss', patience=50)
        callbacks_list = [checkpoint, checkpoint3]
        # callbacks_list = [checkpoint, checkpoint2]

        num_epochs = 1000
        history = model.fit(new_x_train, new_y_train_ohe,
                            steps_per_epoch=int(len(new_x_train) / batch_size),
                            epochs=num_epochs,
                            verbose=2,
                            callbacks=callbacks_list,
                            class_weight=class_weights,
                            validation_data=(new_x_val, new_y_val_ohe),
                            validation_steps=int(len(new_x_val) / batch_size),
                            shuffle=False)

        model = load_model(ckpt_path + model_name)
        test_result = model.predict(new_x_test)
        print('test_result.shape :', test_result.shape)

        total_result.append(test_result)
        pr_list.append(new_pr_test)

        np.save(t_result_path, np.array(total_result))
        np.save(pr_list_path, np.array(pr_list))
        print('total_result.shape :', np.array(total_result).shape)
        print('pr_list.shape :', np.array(pr_list).shape)

    print("stacking time consumed :", time.time() - start_time)

    return


def on_stack_retrain(total_x, total_pr, pred_x, model_name):
    start_time = time.time()

    new_x_train, new_x_val, new_pr_train, new_pr_val = train_test_split(total_x, total_pr, test_size=0.25,
                                                                        shuffle=True, random_state=random_state)

    new_y_train = np.where(new_pr_train > 1, 1, 0)
    new_y_val = np.where(new_pr_val > 1, 1, 0)

    print('new_x_train.shape :', new_x_train.shape)
    print('new_x_val.shape :', new_x_val.shape)
    print('new_y_train.shape :', new_y_train.shape)
    print('new_y_val.shape :', new_y_val.shape)

    print('np.unique(y_train, return_counts=True :', np.unique(new_y_train, return_counts=True),
          class_ratio(np.unique(new_y_train, return_counts=True)[1]))
    print('np.unique(new_y_val, return_counts=True :', np.unique(new_y_val, return_counts=True),
          class_ratio(np.unique(new_y_val, return_counts=True)[1]))

    label = new_y_train.reshape(-1, )
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(label),
                                                      y=label)
    class_weights = dict(enumerate(class_weights))
    print('class_weights :', class_weights)

    print('np.isnan(np.sum(x_train)) :', np.isnan(np.sum(new_x_train)))
    print('np.isnan(np.sum(new_x_val)) :', np.isnan(np.sum(new_x_val)))

    print('np.isnan(np.sum(new_y_train)) :', np.isnan(np.sum(new_y_train)))
    print('np.isnan(np.sum(new_y_val)) :', np.isnan(np.sum(new_y_val)))

    new_y_train_ohe = np_utils.to_categorical(new_y_train, num_classes)
    new_y_val_ohe = np_utils.to_categorical(new_y_val, num_classes)
    print('new_y_train_ohe.shape :', new_y_train_ohe.shape)
    print('new_y_val_ohe.shape :', new_y_val_ohe.shape)

    ckpt_path = 'ckpt/'
    # board_path = 'graph/'

    #       retraining      #
    K.clear_session()  # added for training delation

    try:
        model = load_model(ckpt_path + model_name)
        print("----------------------- model loaded ! -----------------------")

        #     model 을 load 한 경우에만, freeze 해야한다, initial_model 은 feature_extractor 가 학습되지 않았으니까   #
        for l_i, layer in enumerate(model.layers):

            if l_i != len(model.layers) - 1:
                layer.trainable = False

        print("----------------------- layers freezed without last layer ! -----------------------")

    except Exception as e:
        print('Error in load_model :', e)

    batch_size = 512

    opt = Adam(lr=0.00001, decay=0.000005)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(ckpt_path + model_name, monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='auto')
    # checkpoint2 = TensorBoard(log_dir=board_path,
    #                           histogram_freq=0,
    #                           write_graph=True,
    #                           write_images=True)
    checkpoint3 = EarlyStopping(monitor='val_loss', patience=50)
    callbacks_list = [checkpoint, checkpoint3]
    # callbacks_list = [checkpoint, checkpoint2]

    num_epochs = 1000
    history = model.fit(new_x_train, new_y_train_ohe,
                        steps_per_epoch=int(len(new_x_train) / batch_size),
                        epochs=num_epochs,
                        verbose=2,
                        callbacks=callbacks_list,
                        class_weight=class_weights,
                        validation_data=(new_x_val, new_y_val_ohe),
                        validation_steps=int(len(new_x_val) / batch_size),
                        shuffle=False)

    print("retrain time consumed :", time.time() - start_time)

    #       inference phase     #
    model = load_model(ckpt_path + model_name)

    test_result = model.predict(pred_x)
    print('test_result :', test_result)

    return test_result


def get_best_thr(t_result_path, pr_list_path, lb_period=100):
    total_result_ = np.load(t_result_path)
    pr_list_ = np.load(pr_list_path)
    # total_result = list(total_result)
    # pr_list = list(pr_list)

    #       original back_test tested only on pr_list !=1 dataset       #
    #       so, pop out pr_list == 1 condition      #
    pop_index = np.argwhere(pr_list_ == 1)

    total_result = np.delete(total_result_, pop_index)
    pr_list = np.delete(pr_list_, pop_index)

    assert len(total_result) >= lb_period, "len(total_result) < lb_period !"
    assert len(total_result) == len(pr_list), "len(total_result) != len(pr_list) !"

    y_score = total_result[:, [1]]
    y_test = np.where(pr_list > 1, 1, 0)
    pr_test = pr_list

    print("y_score.shape :", y_score.shape)
    print("pr_test.shape :", pr_test.shape)

    min_thr = 0.55
    # best_thr_list = []
    # new_pr_list = []
    # for i in tqdm(range(lb_period, len(total_result))):

    precision, recall, threshold = precision_recall_curve(y_test[-lb_period:], y_score[-lb_period:])
    # precision, recall = precision[:-1], recall[:-1]

    acc_pr_bythr = []
    new_thresh = []
    for thresh in threshold:

        if thresh < 0.5:
            continue

        #   lookback previous range   #
        y_pred = np.where(y_score[-lb_period:, -1] > thresh, 1, 0)

        # test_size = len(y_test[:len(y_score)][i - lb_period:i])
        test_pr_list = pr_test[-lb_period:]

        pred_pr_list = np.where(y_pred == 1, test_pr_list.reshape(-1, ), 1.0)
        pred_pr_list = np.where(np.isnan(pred_pr_list), 1.0, pred_pr_list)
        pred_pr_list = np.where(pred_pr_list == 0.0, 1.0, pred_pr_list)

        acc_pr_bythr.append(np.cumprod(pred_pr_list)[-1])
        new_thresh.append(thresh)

    #       replace best_thr only max_pr > 1      #
    # if np.max(acc_pr_bythr) > 1.10:
    best_thr = new_thresh[np.argmax(acc_pr_bythr)]

    if best_thr < min_thr:
        best_thr = min_thr

    return best_thr
