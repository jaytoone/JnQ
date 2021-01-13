import numpy as np
import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt
import os
from Make_X_total import made_x, min_max_scaler
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)


def dividing(Y_test):
    if Y_test.shape[1] == 1:

        np_one = np.ones((len(Y_test), 1))
        np_zeros = np.zeros((len(Y_test), 1))

        if np.sum(Y_test) != 0:
            Y_test = np.concatenate((np_zeros, np_one), axis=1)

        else:
            Y_test = np.concatenate((np_one, np_zeros), axis=1)

    return Y_test


if __name__ == '__main__':
    
    # input_data_length = int(input("Input Data Length : "))
    input_data_length = 54
    # model_num = input('Press model number : ')
    model_num = 16

    #       Make folder      # 
    try:
        os.mkdir('./pred_ohlcv/%s_%s/' % (input_data_length, model_num))
    except Exception as e:
        print('Error in making folder :', e)
    try:
        os.mkdir('./Figure_pred/%s_%s/' % (input_data_length, model_num))        
    except Exception as e:
        print('Error in making folder :', e)

    except_list = os.listdir('pred_ohlcv/%s_%s' % (input_data_length, model_num))

    #           PARAMS           #
    check_span = 30
    get_fig = 1

    #       LOAD MODEL      #
    model_low = load_model('model/rapid_ascending_low %s_%s.hdf5' % (input_data_length, model_num))
    model_high = load_model('model/rapid_ascending_high %s_%s.hdf5' % (input_data_length, model_num))

    process_rate = 0
    for file in ohlcv_list:

        #   if file in except_list:
        #     continue
        # file = '2019-10-05 CMT ohlcv.xlsx'

        #   print('loading %s' % file)
        process_rate += 1
        print('Processing rate : %.2f' % (process_rate / len(ohlcv_list)))

        try:
            X_test, Y_test_low, Y_test_high, sliced_ohlcv = made_x(file, input_data_length, model_num, check_span, 0)

            if len(sliced_ohlcv) < 100:
                continue

        except Exception as e:
            print('Error in getting data from made_x :', e)
            continue

        closeprice = min_max_scaler(sliced_ohlcv[:, [1]])
        OBV = min_max_scaler(sliced_ohlcv[:, [-1]])
#         print(closeprice)
#         print(closeprice.shape, len(X_test))
#         quit()

        # dataX 에 담겨있는 value 에 [-1] : 바로 이전의 행 x[-1][:].shape = (1, 6)
        # sliced_ohlcv = np.array(list(map(lambda x: x[-1][:], X_test)))
        # print(sliced_ohlcv)
        # quit()

        if len(X_test) != 0:

            X_test = np.array(X_test)

            row = X_test.shape[1]
            col = X_test.shape[2]

            X_test = X_test.astype('float32').reshape(-1, row, col, 1)
#             print(X_test.shape)
#             quit()

            #       Data Preprocessing      #
            Y_pred_low_ = model_low.predict(X_test, verbose=3)
            Y_pred_high_ = model_high.predict(X_test, verbose=3)
            # print(Y_pred_low_.shape)

            max_value_low = np.max(Y_pred_low_, axis=0)
            max_value_high = np.max(Y_pred_high_, axis=0)
            # print(max_value)
            Y_pred_low_[:, [0]] = Y_pred_low_[:, [0]] / max_value_low[0]
            Y_pred_low_[:, [1]] = Y_pred_low_[:, [1]] / max_value_low[1]

            Y_pred_high_[:, [0]] = Y_pred_high_[:, [0]] / max_value_high[0]
            Y_pred_high_[:, [1]] = Y_pred_high_[:, [1]] / max_value_high[1]
            # print(Y_pred_low_[:2])
            # quit()
            Y_pred_low = np.argmax(Y_pred_low_, axis=1)
            Y_pred_high = np.argmax(Y_pred_high_, axis=1)

            # max_value = np.max(Y_pred_[:, [-1]])

            # limit_line = 0.9
            # limit_line_low = 0.9
            # limit_line_high = 0.9
            # Y_pred_low = np.where(Y_pred_low_[:, [-1]] > max_value_low * limit_line_low, 1, 0)
            # Y_pred_high = np.where(Y_pred_high_[:, [-1]] > max_value_high * limit_line_high, 1, 0)

            #       Save Pred_ohlcv      #
            # 기존에 pybithumb 을 통해서 제공되던 ohlcv 와는 조금 다르다 >> 이전 데이터와 현재 y 데이터 행이 같다.
            sliced_Y_low = Y_pred_low.reshape(-1, 1)
            sliced_Y_high = Y_pred_high.reshape(-1, 1)
            pred_ohlcv = np.concatenate((sliced_ohlcv, sliced_Y_low, sliced_Y_high), axis=1)
        
            # col 이 7이 아닌 데이터 걸러주기
            try:
                pred_ohlcv_df = pd.DataFrame(pred_ohlcv, columns=['open', 'close', 'high', 'low', 'volume', 'OBV',
                                                                  'low_check', 'high_check'])
                # pred_ohlcv_df = pd.DataFrame(pred_ohlcv,
                #                              columns=['open', 'close', 'high', 'low', 'volume',
                #                                       'OBV',
                #                                       'low_check', 'high_check'])

            except Exception as e:
                print('Error in making dataframe :', e)
                continue
            # print(pred_ohlcv_df.tail(20))
            # quit()
            pred_ohlcv_df.to_excel('pred_ohlcv/%s_%s/%s' % (input_data_length, model_num, file))

            if get_fig == 1:
                spanlist_low = []
                spanlist_high = []

                for m in range(len(Y_pred_low)):
                    if Y_pred_low[m] > 0.5:
                        if m + 1 < len(Y_pred_low):
                            spanlist_low.append((m, m + 1))
                        else:
                            spanlist_low.append((m - 1, m))

                for m in range(len(Y_pred_high)):
                    if Y_pred_high[m] > 0.5:
                        if m + 1 < len(Y_pred_high):
                            spanlist_high.append((m, m + 1))
                        else:
                            spanlist_high.append((m - 1, m))

                # ----------- 인덱스 초기화 됨 -----------#

                plt.subplot(211)
                # plt.subplot(312)
                plt.plot(closeprice, 'r', label='close')
                plt.plot(OBV, 'b', label='OBV')
                plt.legend(loc='upper right')
                for i in range(len(spanlist_low)):
                    plt.axvspan(spanlist_low[i][0], spanlist_low[i][1], facecolor='m', alpha=0.5)

                plt.subplot(212)
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





