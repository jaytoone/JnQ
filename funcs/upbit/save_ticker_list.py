import pyupbit
import pickle
from datetime import datetime

tickers = pyupbit.get_tickers(fiat="KRW")

# for ticker_ in tickers:
#     print(ticker_)
print(type(tickers))

save_date = str(datetime.now()).split(" ")[0].replace("-", "")
save_path = "../ticker_list/upbit_{}.pkl".format(save_date)
with open(save_path, 'wb') as f:
    pickle.dump(tickers, f)
    print(tickers)
    print(save_path, 'saved !')