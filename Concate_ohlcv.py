

#       DataFrame Concatenate  from 3.1 to 3.25        #
#   3.1의 코인 리스트를 뽑아서, 25일 데이터까지 concat
CoinBot_path = 'C:/Users/Lenovo/OneDrive/CoinBot/'
ohlc_list = os.listdir(CoinBot_path + 'ohlcv')
Coinlist = list()
for file in ohlc_list:
    if file.split()[0] == '2020-03-01':
        Coinlist.append(file.split()[1])
print(Coinlist)

for Coin in Coinlist:

    df_sum = pd.DataFrame(columns=['open', 'close', 'high', 'low', 'volume'])
    for day in range(1, 26):

        try:
            df = pd.read_excel(CoinBot_path + 'ohlcv/' + '2020-03-%02d %s ohlcv.xlsx' % (day, Coin), index_col=0)
            df_sum = df_sum.append(df)
            print('2020-03-%02d %s ohlcv.xlsx' % (day, Coin), 'appended')

        except Exception as e:
            print(e)

    df_sum.to_excel(CoinBot_path + 'ohlcv_concat/' + '2020-03-%02d %s ohlcv.xlsx' % (day, Coin))
    # print(df_sum)
    # quit()