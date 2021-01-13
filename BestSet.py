import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.width', 400)
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


# total_df = pd.DataFrame(
#     columns=['short', 'long', 'signal', 'C', 'gamma', 'epsilon', 'total_profit_avg', 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
#              'min_profit_avg'])
total_df = pd.DataFrame(
    columns=['short', 'long', 'signal', 'total_profit_avg', 'plus_profit_avg', 'minus_profit_avg', 'avg_profit_avg',
             'min_profit_avg', 'std_profit_avg'])

# file_list = os.listdir('BestSet')
# print(file_list)
# txt_list = list()
# for file in file_list:
#     try:
#         if file.split('.')[1] == 'txt':
#             txt_list.append(file)
#     except:
#         pass
#
# for txt in txt_list:
#     with open("BestSet/%s" % txt) as f:
#         lines = f.readlines()
#         for index, value in enumerate(lines):
#             factors = value.split()
#             # print(factors)
#             if len(factors) == 0:
#                 continue
#             if factors[0] != txt.split()[1].split('.')[0]:
#                 continue
#             short = int(factors[0])
#             long = int(factors[1])
#             signal = int(factors[2])
#             total_profit_avg = float(factors[3])
#
#             result_df = pd.DataFrame(data=[[short, long, signal, total_profit_avg]],
#                                      columns=['short', 'long', 'signal', 'total_profit_avg'])
#             total_df = total_df.append(result_df)
#             # print(total_df)
#             # quit()
#     total_df.to_excel('./BestSet/total_df %s.xlsx' % short)

file_list = os.listdir('BestSet/macd_set/')
df_list = list()
for file in file_list:
    try:
        if file.split('.')[1] == 'xlsx':
            df_list.append(file)
    except:
        pass

# print(df_list)
# quit()
for df in df_list:
    result_df = pd.read_excel('./BestSet/macd_set/%s' % df, index_col=0)
    total_df = total_df.append(result_df)

sorted_df = total_df.sort_values(by='min_profit_avg', ascending=False)
aligned_df = sorted_df.reset_index(drop=True)
print(aligned_df.head(50).drop_duplicates())
quit()

aligned_df = aligned_df.head(50)
plt.subplot(311)
plt.scatter(aligned_df['total_profit_avg'], aligned_df['short'], )
plt.subplot(312)
plt.scatter(aligned_df['total_profit_avg'], aligned_df['long'], )
plt.subplot(313)
plt.scatter(aligned_df['total_profit_avg'], aligned_df['signal'], )
plt.show()