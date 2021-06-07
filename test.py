import pandas as pd
from datetime import datetime
import numpy as np

symbol = 'SXPUSDT'
# with open('updown_rnn/survey_logger/%s.txt' % symbol, 'w') as log_file:
#     log_file.write(str(datetime.now().timestamp()))

with open('updown_rnn/survey_logger/%s.txt' % symbol, 'r') as log_file:
    # print(type(log_file.readline()))
    print(float(log_file.readline()))

# import time
# st = datetime.now().timestamp()
# print(st)
# time.sleep(5)
# print(datetime.now().timestamp() - st)



# if 1 and not pd.isna(1):
#     print('break')
# price = 1.0074
#
# print('something new', end=' ')
# print('something')
# tlist = [1, 2, 3, 4]
# a, b = tlist[:2]
# print(a, b)
# tlist = [tlist, tlist]
# print(list(map(lambda x: sum(x), tlist)))
#
# import sys
# import inspect
#
# clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
# print(clsmembers)

# error_msg = ('ExecuteError', '[Executing] -4003: Quantity less than zero.')
# print('Quantity less than zero' in str(error_msg))
# try:
#     precision = len(str(price).split('.')[1])
# except:
#     precision = 0
#
# print('precision :', precision)
#
# for back_i in range(5, 0, -1):
#     pass
# print(back_i)

# print(datetime.fromtimestamp(1613759400.004374))
# print(int('08'))
# import os

# tlist = [2, 2, 1, 4, 2]
# print(np.cumprod(tlist))
# # print(tlist.pop(2))
# # print(sum(tlist))
# print(tlist[0:1])
# print(10 ** -2)
