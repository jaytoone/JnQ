from funcs_binance.binance_futures_modules import *
import pickle
from datetime import datetime

tickers = []
result = request_client.get_exchange_information()

for data in result.symbols:
    symbol = data.symbol
    print(symbol)
    tickers.append(symbol)
    # quit()

# for ticker_ in tickers:
#     print(ticker_)
print(type(tickers))

save_date = str(datetime.now()).split(" ")[0].replace("-", "")
save_path = "../ticker_list/binance_futures_{}.pkl".format(save_date)
with open(save_path, 'wb') as f:
    pickle.dump(tickers, f)
    print(tickers)
    print(save_path, 'saved !')