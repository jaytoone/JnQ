import pandas as pd
from datetime import datetime
import numpy as np

# if 1 and not pd.isna(1):
#     print('break')
# price = 1.0074
#
# print('something new', end=' ')
# print('something')
tlist = [1, 2, 3, 4]
tlist = [tlist, tlist]
print(list(map(lambda x: sum(x), tlist)))

import sys
import inspect

clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
print(clsmembers)
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

# print(datetime.fromtimestamp(1610197130835/ 1000))
# print(int('08'))
# import os

# tlist = [2, 2, 1, 4, 2]
# print(np.cumprod(tlist))
# # print(tlist.pop(2))
# # print(sum(tlist))
# print(tlist[0:1])
# print(10 ** -2)
