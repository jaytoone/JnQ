from binance_futures_concat_candlestick import concat_candlestick, a_day
from binance_futures_bot_config import LabelType
from funcs.funcs_for_trade import intmin
# import numpy as np
import pandas as pd

days = 1
end_date = '2020-12-09'
# end_date = '2020-08-24'
# end_date = '2020-08-27'
# end_date = '2021-01-22'
end_date = None
symbol = 'ADAUSDT'
label_type = LabelType.PROFIT

first_df, _ = concat_candlestick(symbol, '1m', days, end_date=end_date, show_process=True)
second_df, _ = concat_candlestick(symbol, '15m', days, end_date=end_date, show_process=True)

first_df_copy = first_df.copy()

interval = 15


def to_higher_ohlc(first_df, interval):

    assert interval < 60, 'Current fuction is only for below 1h interval'

    for i in range(len(first_df)):

        roll_i = intmin(first_df.index[i]) % interval + 1
        first_df_copy['open'].iloc[i] = first_df['open'].iloc[i - (roll_i - 1)]
        first_df_copy['high'].iloc[i] = first_df['high'].rolling(roll_i).max().iloc[i]
        first_df_copy['low'].iloc[i] = first_df['low'].rolling(roll_i).min().iloc[i]

    # print(pd.concat([first_df, first_df_copy], axis=1))

    return first_df_copy


print(to_higher_ohlc(first_df, interval).tail(20))
print(second_df.tail(5))
