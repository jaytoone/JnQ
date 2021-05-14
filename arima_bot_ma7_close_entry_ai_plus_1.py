from arima_bot_long_ma7_close_entry_ai_plus import *

if __name__=='__main__':

    arima_bot = ARIMA_Bot(symbol='ETHUSDT', interval='30m', threshold=0.5451, tp=0, leverage_=5, initial_asset=180,
                          stacked_df_on=True)
    arima_bot.run()