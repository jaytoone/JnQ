from futures_modules import *
from futures_concat_candlestick import concat_candlestick
from datetime import datetime
import numpy as np


symbol_ = 'ALGOUSDT'
prev_minute = datetime.now().minute
time_list = list()
data_list = list()
second_stack = list()
while 1:

    current_datetime = datetime.now()

    if current_datetime.second >= 55:  # approximate close

        #          Just Take Realtime price and watch SL condition      #
        try:
            realtime_price = get_market_price(symbol_)
        except Exception as e:
            print('Error in get_market_price :', e)
            continue

        # print(current_datetime)
        # print('realtime_price :', realtime_price)
        time_list.append(current_datetime)
        data_list.append(realtime_price)

    #           Check close data completion time            #
    elif current_datetime.minute != prev_minute:

        if current_datetime.second < 5:

            first_df, _ = concat_candlestick(symbol_, '1m', days=1)
            # print()
            minute = str(first_df.index[-2]).split(':')[1]
            # print(minute, current_datetime.minute)
            if int(minute) == current_datetime.minute - 1:
                # print('close data completion time :', current_datetime)
                prev_minute = current_datetime.minute
                # print('close :', first_df['close'].iloc[-2])

                #       Check closest data building time        #
                # print(np.array(data_list) - first_df['close'].iloc[-2])
                # print(np.argmin(np.array(data_list) - first_df['close'].iloc[-2]))
                # print(time_list[np.argmin(np.array(data_list) - first_df['close'].iloc[-2])])
                second = str(time_list[np.argmin(np.array(data_list) - first_df['close'].iloc[-2])]).split(':')[-1].split('.')[0]
                print(second)
                second_stack.append(second)

                if len(second_stack) % 10 == 0:
                    np.save('second_stack.npy', np.array(second_stack))

                time_list = list()
                data_list = list()
