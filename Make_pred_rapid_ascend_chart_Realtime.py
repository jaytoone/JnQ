import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Make_X_rapid_ascend_chart import made_x
import pybithumb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)

if __name__ == '__main__':

    #           PARAMS           #
    input_data_length = 30
    model_num = '64'
    crop_size = input_data_length
    check_span = 40

    #       Make folder      #
    try:
        os.mkdir('./pred_ohlcv/%s_%s/' % (input_data_length, model_num))
    except Exception as e:
        pass

    try:
        os.mkdir('./Figure_pred/%s_%s/' % (input_data_length, model_num))
    except Exception as e:
        pass

    except_list = os.listdir('./pred_ohlcv/%s_%s' % (input_data_length, model_num))

    TopCoin = ['BOA']

    #       LOAD MODEL      #
    model = load_model('./model/rapid_ascending %s_%s_chart.hdf5' % (input_data_length, model_num))

    for Coin in TopCoin:

        #         if file in except_list:
        #             continue

        print('loading %s' % Coin)

        try:
            # X_test = np.load('./Made_Chart_to_np/%s_%s/%s %s.npy' % (input_data_length, model_num,
            #                                                               Date, Coin)).astype(np.float32) / 255.
            file = pybithumb.get_ohlcv(Coin, interval='minute1')
            result = made_x(file, input_data_length, model_num, check_span=check_span, get_fig=0, crop_size=input_data_length)
            X_test = np.array(result[0])
            y = np.array(result[1]).astype(np.uint8)
            # print(X_test.shape)
            # print(y[1])
            # quit()

            row = X_test.shape[1]
            col = X_test.shape[2]

        except Exception as e:
            print('Error in getting data from made_x :', e)
            continue

        if X_test is not None:
            if len(X_test) != 0:

                Y_pred_ = model.predict(X_test, verbose=1)

                # min_value = np.min(Y_pred_, axis=0)
                # print(min_value)
                # print(Y_pred_)
                Y_pred = np.argmax(Y_pred_, axis=1)

                for i in range(len(X_test)):
                    plt.title('%s %s' % (y[i], Y_pred[i]))
                    plt.imshow(X_test[i])
                    plt.show()

                # Y_pred = np.zeros(len(Y_pred_))
                # for i in range(len(Y_pred_)):
                #     if Y_pred_[i][0] < 1.14e-15:
                #         Y_pred[i] = 1
                #     if Y_pred_out_[i][0] < 1.14e-15:
                #         Y_pred[i] = 2

                # spanlist_low = []
                # spanlist_high = []
                #
                # for m in range(len(Y_pred)):
                #     if (Y_pred[m] > 0.5) and (Y_pred[m] < 1.5):
                #         if m + 1 < len(Y_pred):
                #             spanlist_low.append((m, m + 1))
                #         else:
                #             spanlist_low.append((m - 1, m))
                #
                # for m in range(len(Y_pred)):
                #     if (Y_pred[m] > 1.5) and (Y_pred[m] < 2.5):
                #         if m + 1 < len(Y_pred):
                #             spanlist_high.append((m, m + 1))
                #         else:
                #             spanlist_high.append((m - 1, m))
                #
                # ohlcv_excel = pd.read_excel(dir + Coin, index_col=0)
                # ohlcv_excel['MA60'] = ohlcv_excel['close'].rolling(60).mean()
                # ohlcv_data = ohlcv_excel.values[sum(ohlcv_excel.MA60.isna()): -check_span].astype(np.float)[crop_size:]

                # plt.subplot(211)
                # # plt.subplot(313)
                # plt.plot((ohlcv_data[:, [1]]), 'gold', label='close')
                # plt.plot((ohlcv_data[:, [5]]), 'b', label='ma')
                # plt.legend(loc='upper right')
                # for i in range(len(spanlist_low)):
                #     plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='c', alpha=0.7)
                #
                # plt.subplot(212)
                # # plt.subplot(313)
                # plt.plot((ohlcv_data[:, [1]]), 'gold', label='close')
                # plt.plot((ohlcv_data[:, [5]]), 'b', label='ma')
                # plt.legend(loc='upper right')
                # for i in range(len(spanlist_high)):
                #     plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='m', alpha=0.7)
                #
                # Date = Coin.split()[0]
                # Coin = file.split()[1].split('.')[0]
                # plt.savefig('./Figure_pred/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
                # plt.close()





