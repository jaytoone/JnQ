import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import pybithumb
import Funcs
import Coin_Rank
import random
import time

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
tf.set_random_seed(777)


# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차원


# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


# ----- 하이퍼파라미터 -----#
input_data_column_cnt = 6  # 입력데이터의 컬럼 개수(Variable 개수)
output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

seq_length = 54  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 128  # 각 셀의 (hidden)출력 크기
forget_bias = 1.0  # 망각편향(기본값 1.0)
num_stacked_layers = 1  # stacked LSTM layers 개수
keep_prob = 1.0  # dropout할 때 keep할 비율

epoch_num = 1000  # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
learning_rate = 0.01  # 학습률


# 모델(LSTM 네트워크) 생성
def lstm_cell():
    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  to the biases of the forget gate
    #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


def clear_datatime(datetime_index):

    datetime_type = type(datetime_index)
    string_hour = str(datetime_index).split()[1].split(':')[0]
    string_min = str(datetime_index).split()[1].split(':')[1]
    string_second = str(datetime_index).split()[1].split(':')[2]

    cleared_min = int(string_min) // 10 * 10
    datetime_index = str(datetime_index).split()[0] + ' %s:%02d:%s' % (string_hour, cleared_min, string_second)

    return datetime_type(datetime_index)


if __name__ == '__main__':

    #           Load Data           #

    input_data_length = int(input('input_data_length : '))

    Made_X = np.load('./Made_X/Made_X %s.npy' % input_data_length)
    Made_Y = np.load('./Made_X/Made_Y %s.npy' % input_data_length)

    # 학습용/테스트용 데이터 생성
    # 전체 70%를 학습용 데이터로 사용, 나머지(30%)를 테스트용 데이터로 사용
    total_len = len(Made_X)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    test_len = total_len - (train_len + val_len)

    X_train = Made_X[:train_len].astype('float32').reshape(-1, input_data_length, col, 1)
    X_val = Made_X[train_len:train_len + val_len].astype('float32').reshape(-1, input_data_length, col, 1)
    X_test = Made_X[train_len + val_len:].astype('float32').reshape(-1, input_data_length, col, 1)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

        # ----- 텐서플로우 플레이스홀더 생성 -----#
        # 입력 X, 출력 Y를 생성한다
        X = tf.placeholder(tf.float64, [None, seq_length, input_data_column_cnt])
        Y = tf.placeholder(tf.float64, [None, 1])

        # 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
        targets = tf.placeholder(tf.float64, [None, 1])
        predictions = tf.placeholder(tf.float64, [None, 1])

        # num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
        stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

        # RNN Cell(여기서는 LSTM셀임)들을 연결
        hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float64)

        # [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
        # 과거 여러 거래일의 주가를 이용해서 다음날의 주가 1개를 예측하기때문에 MANY-TO-ONE형태이다
        hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

        # 손실함수로 평균제곱오차를 사용한다
        loss = tf.reduce_sum(tf.square(hypothesis - Y))
        # 최적화함수로 AdamOptimizer를 사용한다
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate) # LSTM과 궁합 별로임

        train = optimizer.minimize(loss)

        # RMSE(Root Mean Square Error)
        # 제곱오차의 평균을 구하고 다시 제곱근을 구하면 평균 오차가 나온다
        # rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions))) # 아래 코드와 같다
        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

        train_error_summary = []  # 학습용 데이터의 오류를 중간 중간 기록한다
        test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
        test_predict = ''  # 테스트용데이터로 예측한 결과

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # 학습한다
        start_time = datetime.now()  # 시작시간을 기록한다
        print('학습 시작!')
        for epoch in range(epoch_num):
            _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            if ((epoch + 1) % 10 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
                # 학습용데이터로 rmse오차를 구한다
                train_predict = sess.run(hypothesis, feed_dict={X: trainX})
                train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
                train_error_summary.append(train_error)

                # 테스트용데이터로 rmse오차를 구한다
                test_predict = sess.run(hypothesis, feed_dict={X: testX})
                test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
                test_error_summary.append(test_error)

                # 현재 오류를 출력한다
                if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):
                    print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1, train_error, test_error,
                                                                                         test_error - train_error))
                # break

        end_time = datetime.now()  # 종료시간을 기록한다
        elapsed_time = end_time - start_time  # 경과시간을 구한다
        print('elapsed_time:', elapsed_time)

        print('train_error:', train_error_summary[-1], end='')
        print(' test_error:', test_error_summary[-1], end='')
        print(' min_test_error:', np.min(test_error_summary))
        print("학습 완료!")
        print()

        # ----- 결과 그래프 -----#
        plt.figure(1)
        plt.plot(train_error_summary, 'gold')
        plt.plot(test_error_summary, 'b')
        plt.xlabel('Epoch(x100)')
        plt.ylabel('Root Mean Square Error')

        plt.figure(2)
        plt.plot(testY, 'r')
        plt.plot(test_predict, 'b')
        plt.xlabel('Time Period')
        plt.ylabel('Stock Price')
        plt.show()
        # plt.savefig("./Chart/%s.png" % Coin)
        # plt.close()
        break
        # ----- 결과 그래프 완성-----#


        # 학습된 모델 저장
        saver = tf.train.Saver()
        save_path = saver.save(sess, './Educated_model/%s model.ckpt' % str(datetime.now()).split()[0])

        # Dataframe 저장
        real_test_predict = reverse_min_max_scaling(price, test_predict)

        # 예상 종가열을 기존 데이터프레임에 추가
        timestamp_list = list(timestamp[-len(real_test_predict):])
        real_test_predict = pd.DataFrame(index=timestamp_list, data=real_test_predict, columns=['pred_price_close'])
        lstm_dataframe = pd.concat([input_dataframe[-len(real_test_predict):], real_test_predict], axis=1)
        lstm_dataframe.to_excel("./Data_LSTM/%s %s.xlsx" % (str(datetime.now()).split()[0], Coin))

        # print("---------------------- num_stacked_layers = %d ----------------------" % num_stacked_layers)
        # num_stacked_layers += 1
        # if num_stacked_layers > 5:

        sess.close()  # 세션을 닫을 경우 저장되어있는 데이터들을 해제한다. (재학습 시키는 경우만 사용하도록하자)
        break

    # -------------------------------- 학습 완료 --------------------------------#










