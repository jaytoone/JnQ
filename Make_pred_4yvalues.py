import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Make_X3 import made_x
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)

if __name__ == '__main__':

    input_data_length = 54
    # model_num = input('Press model num : ')
    model_num = 20

    #           PARAMS           #
    check_span = 30
    Range_fluc = 1.035
    get_fig = 1

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

    #       LOAD MODEL      #
    model = load_model('./model/rapid_ascending %s_%s.hdf5' % (input_data_length, model_num))

    for file in ohlcv_list:

        #         if file in except_list:
        #             continue

        print('loading %s' % file)

        try:
            X_test, Y_test, sliced_ohlcv = made_x(file, input_data_length, model_num, check_span, Range_fluc, 0)

        except Exception as e:
            print('Error in getting data from made_x :', e)
            continue

        closeprice = np.roll(np.array(list(map(lambda x: x[-1][[1]][0], X_test))), -1)
        OBV = np.roll(np.array(list(map(lambda x: x[-1][[5]][0], X_test))), -1)

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if len(X_test) != 0:

            X_test = np.array(X_test)

            row = X_test.shape[1]
            col = X_test.shape[2]

            X_test = X_test.astype('float32').reshape(-1, row, col, 1)
            # print(X_test.shape)

            Y_pred_ = model.predict(X_test, verbose=1)
            max_value = np.max(Y_pred_, axis=0)
            # print(Y_pred_)
            # print(max_value)
            # quit()


            def classifier(Y_pred_):
                Y_pred_[0] = Y_pred_[0] / max_value[0]
                Y_pred_[1] = Y_pred_[1] / max_value[1]
                Y_pred_[2] = Y_pred_[2] / max_value[2]
                Y_pred_[3] = Y_pred_[3] / max_value[3]

                return Y_pred_

            # limit_line = 0.9
            Y_pred = np.array(list(map(classifier, Y_pred_)))
            #   0.6573004  0.17771776 0.11235251 0.05262928
            print(Y_pred[:10])
            quit()

            #       Save Pred_ohlcv      #
            #   기존에 pybithumb 을 통해서 제공되던 ohlcv 와는 조금 다르다 >> 이전 데이터와 현재 y 데이터 행이 같다.
            sliced_Y = Y_pred.reshape(-1, 1)
            pred_ohlcv = np.concatenate((sliced_ohlcv, sliced_Y), axis=1)  # axis=1 가로로 합친다

            #   col 이 7이 아닌 데이터 걸러주기
            try:
                pred_ohlcv_df = pd.DataFrame(pred_ohlcv,
                                             columns=['open', 'close', 'high', 'low', 'volume', 'OBV', 'trade_state'])

            except Exception as e:
                print(e)
                continue
            # print(pred_ohlcv_df.tail(20))
            # quit()
            pred_ohlcv_df.to_excel('./pred_ohlcv/%s_%s/%s' % (input_data_length, model_num, file))

            # Y_pred 에 1 이 존재하면, Plot Comparing
            if get_fig == 1:

                spanlist = []
                spanlist_low = []
                spanlist_high = []

                for m in range(len(Y_pred)):
                    if (Y_pred[m] > 0.5) and (Y_pred[m] < 1.5):
                        if m + 1 < len(Y_pred):
                            spanlist.append((m, m + 1))
                        else:
                            spanlist.append((m - 1, m))

                for m in range(len(Y_pred)):
                    if (Y_pred[m] > 1.5) and (Y_pred[m] < 2.5):
                        if m + 1 < len(Y_pred):
                            spanlist_low.append((m, m + 1))
                        else:
                            spanlist_low.append((m - 1, m))

                for m in range(len(Y_pred)):
                    if (Y_pred[m] > 2.5) and (Y_pred[m] < 3.5):
                        if m + 1 < len(Y_pred):
                            spanlist_high.append((m, m + 1))
                        else:
                            spanlist_high.append((m - 1, m))

                plt.subplot(311)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist)):
                    plt.axvspan(spanlist[i][0], spanlist[i][1], facecolor='g', alpha=0.5)

                plt.subplot(312)
                # plt.subplot(313)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_low)):
                    plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='m', alpha=0.5)

                plt.subplot(313)
                # plt.subplot(313)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_high)):
                    plt.axvspan(spanlist_high[i][0], spanlist_high[i][1], facecolor='c', alpha=0.5)

                Date = file.split()[0]
                Coin = file.split()[1].split('.')[0]
                plt.savefig('./Figure_pred/%s_%s/%s %s.png' % (input_data_length, model_num, Date, Coin), dpi=500)
                plt.close()





