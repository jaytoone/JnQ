import os
import pandas as pd
import pickle

home_dir = os.path.expanduser('~')
dir = home_dir + '/OneDrive/CoinBot/ohlcv/'
ohlcv_list = os.listdir(dir)


total_top_list = list()
for day in range(14):

    top_list = list()
    top_fluc = list()

    for Coin in ohlcv_list:
        if Coin.endswith('.xlsx'):
            if Coin.split('-')[1] not in ['08'] or Coin.split('-')[2].split()[0] not in ['%02d' % day]:
                continue

        df = pd.read_excel(dir + '%s' % Coin, index_col=0)
        chart_gap = (df['high'].max() / df['low'].min())

        top_list.append(Coin)
        top_fluc.append(chart_gap)
        print(Coin)

    top_df = pd.DataFrame(index=top_list, data=top_fluc, columns=['chart_gap'])
    top20 = top_df.sort_values(by='chart_gap', ascending=False).head(20)
    try:
        total_top_list += (list(top20.index.values))
    except Exception as e:
        print(e)
    # print(total_top_list)

    with open('top20.txt', 'wb') as f:
        pickle.dump(total_top_list, f)
    print('top20.txt saved')

    # top_df.to_excel('%s top_df.xlsx' % Coin.split()[0])
    # quit()



