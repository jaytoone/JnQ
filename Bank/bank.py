
# Library for Futures.
from funcs.binance_f.model import *
from funcs.binance.um_futures import UMFutures
from funcs.binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
# from funcs.binance.error import ClientError


# Library for Bank.
# from funcs.binance.shield_module_v4_1 import ShieldModule
from funcs.public.constant import *
from funcs.public.indicator import *
from funcs.public.broker import *


# Library for Bank.
# from funcs.binance.futures_module_v4 import FuturesModule
# from funcs.binance.futures_concat_candlestick_ftr_v2 import concat_candlestick
# from funcs.binance.futures_concat_candlestick_ftr_v2 import concat_candlestick_v2


import pandas as pd
import numpy as np
import math

import time
from datetime import datetime
from pandas import to_datetime

import pickle
import logging

import logging.config
from easydict import EasyDict
import json
from ast import literal_eval

import shutil
import importlib
import os

import telegram
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters

from IPython.display import clear_output

pd.set_option('mode.chained_assignment',  None)
pd.set_option('display.max_columns', 100)




class Bank(UMFutures):

    """
    v4.1    
        allow server_time.
        remove read_and_write_config

        modify set_logger(), rm console handler
    v4.2  
        modify to rotateHandler.
        append echo & get_messenger_bot in Class Bank.
            caused by echo 'context missing' error.
        
    last confirmed at, 20240528 1039
    """

    def __init__(self, **kwargs):
        
        api_key, secret_key = self.load_key(kwargs['path_api'])
        
        UMFutures.__init__(self, key=api_key, secret=secret_key)

        self.websocket_client = UMFuturesWebsocketClient()
        self.websocket_client.start()
        self.price_market = {}
        
        self.save_path_log = kwargs['save_path_log']
        self.set_logger()
        
        self.path_config = kwargs['path_config']
        with open(self.path_config, 'r') as f:
            self.config = EasyDict(json.load(f))
        
        # add messegner
        self.chat_id = kwargs['chat_id']
        self.token = kwargs['token'] # "7156517117:AAF_Qsz3pPTHSHWY-ZKb8LSKP6RyHSMTupo"        
        self.msg_bot = None
        self.get_messenger_bot()
        # self.push_msg(self, "messenger connection established.")
        
        
        # inital set.
            # balance
        self.balance_available = self.config.trader_set.initial_asset
        self.balance_over = None
        self.balance_min = 5.0  # USDT, not used in Stock.

            # income
        self.income = 0.0
        self.income_accumulated = self.config.trader_set.income_accumulated
        
            # profit
        self.profit = 0.0
        self.profit_accumulated = self.config.trader_set.profit_accumulated

                # get previous trade's accumulated profit info.
        # if self.config.trader_set.profit_mode == "SUM":
        #     self.profit_real_accumulated = self.config.trader_set.acc_profit_sum
        # else:
        #     self.profit_real_accumulated = self.config.trader_set.acc_profit_prod
        # self.profit_ideal_accumulated = self.profit_real_accumulated
        
        
    @staticmethod
    def load_key(key_abspath):
        with open(key_abspath, 'rb') as f:
            return pickle.load(f)            

    
    @staticmethod
    def get_precision_by_price(price):
        try:
            precision = len(str(price).split('.')[1])
        except Exception as e:
            precision = 0
        return precision

    @staticmethod
    def calc_with_precision(data, precision_data, def_type='floor'):

        if not pd.isna(data):
            if precision_data > 0:
                if def_type == 'floor':
                    data = math.floor(data * (10 ** precision_data)) / (10 ** precision_data)
                elif def_type == 'round':
                    data = float(round(data, precision_data))
                else:
                    data = math.ceil(data * (10 ** precision_data)) / (10 ** precision_data)
            else:
                data = int(data)

        return data


    def agg_trade_message_handler(self, message):
        """
        1. websocket streaming method 를 이용한 get_price_realtime method.
            a. try --> received data = None 일 경우를 대비한다.
        """
        try:
            self.price_market[message['s']] = float(message['p'])
        except Exception as e:
            pass

    def set_logger(self,):

        """
        v1.0
            add RotatingFileHandler

        last confirmed at, 20240520 0610.
        """
        
        simple_formatter = logging.Formatter("[%(name)s] %(message)s")
        complex_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s")
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.DEBUG)
    
        # file_handler = logging.FileHandler(self.save_path_log)
        file_handler = logging.handlers.RotatingFileHandler(self.save_path_log, maxBytes=100000000, backupCount=10)
        file_handler.setFormatter(complex_formatter)
        file_handler.setLevel(logging.DEBUG)        
    
        self.sys_log = logging.getLogger('Bank')
    
        self.sys_log.handlers.clear()
        self.sys_log.addHandler(console_handler)
        self.sys_log.addHandler(file_handler)
        self.sys_log.setLevel(logging.DEBUG)      

    def echo(self, update, context):    
        self.user_text = update.message.text

    def get_messenger_bot(self, ):

        """
        v2.0
            use token from self directly.
        v3.0
            remove self.msg_bot exist condition.
    
        last confirmed at, 20240528 1257.
        """
            
        # init
        self.msg_bot = telegram.Bot(token=self.token)    
        self.user_text = None # messenger buffer.
        
        # get updater & set handler
        self.updater = Updater(token=self.token, use_context=True)
        dispatcher = self.updater.dispatcher
        
        echo_handler = MessageHandler(Filters.text & (~Filters.command), self.echo)
        dispatcher.add_handler(echo_handler)
        
        # start_polling.
        self.updater.start_polling()
        
        self.sys_log.debug("msg_bot {} assigned.".format(self.token))
        


def push_msg(self, msg):    
    try:
        self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
    except Exception as e:
        self.sys_log.error(e)
        

def read_and_write_config(self, mode='r', config_edited=None):
    try:
        file_config = open(self.path_config, mode)

        if mode == 'r':
            self.config = EasyDict(json.load(file_config))
            
        elif mode == 'w':
            assert config_edited is not None, "assert config_edited is not None"
            json.dump(config_edited, file_config, indent=2)
        else:
            assert mode in ['r', 'w'], "assert mode in ['r', 'w']"

        # 1. opened files should be closed.
        file_config.close()

    except Exception as e:
        msg = "error in read_and_write_config : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
        # self.kill_proc()


def get_tables(self, ):

    """
    v1.0
        we need this func. only once (program start, loading saved data.)
        replace csv to excel : preserving dtypes.
    v2.0
        path_table_trade replace to feather.
        add restoration logic.
        
    last confirmed at, 20240518 2355.
    """
    
    self.table_condition = pd.read_excel(self.path_table_condition)
    # self.table_trade = pd.read_feather(self.path_table_trade)
    self.table_trade = pd.read_pickle(self.path_table_trade)
    try:
        self.table_log = pd.read_excel(self.path_table_log)
    except:
        self.table_log = pd.read_excel(self.path_table_log.replace(".xlsx", "_bk.xlsx"))
    

def set_tables(self, mode='ALL'):

    """
    v1.0
        we need this func. only once (program start, loading saved data.)        
        replace csv to excel : preserving dtypes.
    v2.0
        path_table_trade replace to feather.
    v2.1
        divide save target.
        save as pickle
            table_trade
                feather doesnt support 'something'.
    v2.3
        save as excel
            table condition, log
                for user interfacing.
        
    last confirmed at, 20240524 1946.
    """    

    if mode in ['ALL', 'CONDITION']:  
        self.table_condition.to_excel(self.path_table_condition, index=False)
        self.table_condition.to_excel(self.path_table_condition.replace(".xlsx", "_bk.xlsx"), index=False)
        
    if mode in ['ALL', 'TRADE']:            
        # if len(self.table_trade.columns) > 46:
        #     self.table_trade = self.table_trade.iloc[:, 1:]
            
        self.table_trade.reset_index(drop=True).to_pickle(self.path_table_trade)
        self.table_trade.reset_index(drop=True).to_pickle(self.path_table_trade.replace(".pkl", "_bk.pkl"))
    
    if mode in ['ALL', 'LOG']:
        self.table_log.to_excel(self.path_table_log, index=False)
        self.table_log.to_excel(self.path_table_log.replace(".xlsx", "_bk.xlsx"), index=False)


def read_and_write_config(self, mode='r', config_edited=None):
    try:
        file_config = open(self.path_config, mode)

        if mode == 'r':
            self.config = EasyDict(json.load(file_config))
            
        elif mode == 'w':
            assert config_edited is not None, "assert config_edited is not None"
            json.dump(config_edited, file_config, indent=2)
        else:
            assert mode in ['r', 'w'], "assert mode in ['r', 'w']"

        # 1. opened files should be closed.
        file_config.close()

    except Exception as e:
        msg = "error in read_and_write_config : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
        # self.kill_proc()


def get_tickers(self, ):
    
    self.tickers_available = [info['symbol'] for info in self.exchange_info()['symbols']]
                    

    
def get_streamer(self):
    
    row_list = literal_eval(self.config.trader_set.row_list)
    itv_list = literal_eval(self.config.trader_set.itv_list)
    rec_row_list = literal_eval(self.config.trader_set.rec_row_list)
    use_rows, days = calc_rows_and_days(itv_list, row_list, rec_row_list)

    back_df = pd.read_feather(self.config.trader_set.back_data_path, columns=None, use_threads=True).set_index("index")

    if self.config.trader_set.start_datetime != "None":
        target_datetime = pd.to_datetime(self.config.trader_set.start_datetime)
        if target_datetime in back_df.index:
            start_idx = np.argwhere(back_df.index == target_datetime).item()
        else:
            start_idx = use_rows
    else:
        start_idx = use_rows

    # 시작하는 idx 는 필요로하는 rows 보다 커야함.
    assert start_idx >= use_rows, "more dataframe rows required"

    for i in range(start_idx + 1, len(back_df)):  # +1 for i inclusion
        yield back_df.iloc[i - use_rows:i]


def concat_candlestick(self, interval, days, limit, end_date=None, show_process=False, timesleep=None):

    """
    v2.0
        aggregate dataframe by timestamp_unit.
        add error solution. between valid response.

    last confirmed at, 20240605 2019.
    """

    if show_process:
        self.sys_log.debug("symbol : {}".format(self.symbol))

    limit_kline = 1500
    assert limit <= limit_kline, "assert limit < limit_kline ({})".format(limit_kline)

    if end_date is None:
        end_date = str(datetime.now()).split(' ')[0]
        

    timestamp_end = int(datetime.timestamp(pd.to_datetime('{} 23:59:59'.format(end_date))) * 1000)
    timestamp_days = days * 86399000
    timestamp_start = timestamp_end - timestamp_days

    timestamp_unit = (itv_to_number(interval) * 60000 * limit)


    time_arr_start = np.arange(timestamp_start, timestamp_end, timestamp_unit)
    time_arr_end = time_arr_start + timestamp_unit

    df_list = []
    for index_time, (time_start, time_end) in enumerate(zip(time_arr_start, time_arr_end)):
        try:
            response = self.klines(symbol=self.symbol,
                                    interval=interval,
                                    startTime=int(time_start),
                                    endTime=int(time_end),
                                    limit=limit)

            # data validation.
            if len(response) == 0:
                # data skip
                if len(df_list) != 0:
                    return None, ''
                # no data.
                else:
                    continue

            df = pd.DataFrame(np.array(response)).set_index(0).iloc[:, :5]
            df.index = list(map(lambda x: datetime.fromtimestamp(int(x) / 1000), df.index)) # modify to datetime format.
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)

            # display(df.tail())

            if show_process:
                self.sys_log.debug("{} --> {}".format(df.index[0], df.index[-1]))

            df_list.append(df)
            
        except Exception as e:
            self.push_msg(self, "error in klines : {}".format(str(e)))
            time.sleep(timesleep)
            
        else:
            if timesleep is not None:
                time.sleep(timesleep)

    sum_df = pd.concat(df_list)

    return sum_df[~sum_df.index.duplicated(keep='last')], end_date
    


def get_df_new_by_streamer(self, ):
    
    try:
        self.df_res = next(self.streamer)  # Todo, 무결성 검증 미진행
    except Exception as e:
        msg = "error in get_df_new_by_streamer : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
        # self.kill_proc()  # error 처리할 필요없이, backtrader 프로그램 종료
        # return None, None   # output len 유지
    # else:
    #     return df_res




def get_df_new(self, interval='1m', days=2, limit=1500, timesleep=None):

    """
    v2.0
        add self.config.trader_set.get_df_new_timeout
        add df_res = None to all return phase.
    v3.0
        replace concat_candlestick to v2.0
    v3.1
        replace to bank.concat_candlestick

    last confirmed at, 20240604 1814.
    """
    
    # interval = '15m'
    # days = 2 # at least 2 days for 15T II value to be synced.
    # limit = 1500 # fixed in Binance.
    end_date = None
    
    while 1:
        try:
            df_new, _ = self.concat_candlestick(self,
                                              interval,
                                              days,
                                              limit=limit,
                                              end_date=end_date,
                                              show_process=True,
                                              timesleep=timesleep) # timesleep automatically adjust, when days > 1

        except Exception as e:
            msg = "error in concat_candlestick : {}".format(e)
            self.sys_log.error(msg)
            if 'sum_df' not in msg:
                self.push_msg(self, msg)
                
            # appropriate term for retries.
            time.sleep(self.config.trader_set.api_term)
        else:
            self.df_res = df_new
            return



def get_df_mtf(self, mode="OPEN", row_slice=True):

    """
    v2.0   
        remove not used phase.
        Class mode.

    last confirmed at, 20240412 2005.
    """

    try:
        make_itv_list = [m_itv.replace('m', 'T') for m_itv in literal_eval(self.config.trader_set.itv_list)]
        row_list = literal_eval(self.config.trader_set.row_list)
        rec_row_list = literal_eval(self.config.trader_set.rec_row_list)
        offset_list = literal_eval(self.config.trader_set.offset_list)

        assert len(make_itv_list) == len(offset_list), "length of itv & offset_list should be equal"
        htf_df_list = [to_htf(self.df_res, itv=itv_, offset=offset_) for itv_idx, (itv_, offset_) in enumerate(zip(make_itv_list, offset_list)) if itv_idx != 0]
        htf_df_list.insert(0, self.df_res)
        

        # 1. row slicing considered latecy for indicator gerenation.
        if row_slice:
            df, df_3T, df_5T, df_15T, df_30T, df_H, df_4H = [df_s.iloc[-row_list[row_idx]:].copy() for row_idx, df_s in enumerate(htf_df_list)]
            rec_df, rec_df_3T, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_H, rec_df_4H = [df_s.iloc[-rec_row_list[row_idx]:].copy() for row_idx, df_s in enumerate(htf_df_list)]
        else:
            df, df_3T, df_5T, df_15T, df_30T, df_H, df_4H = htf_df_list
            rec_df, rec_df_3T, rec_df_5T, rec_df_15T, rec_df_30T, rec_df_H, rec_df_4H = htf_df_list

    except Exception as e:
        self.sys_log.error("error in sync_check :", e)
    # else:
    #     return df



def get_wave_info(self, mode="OPEN"):

    """
    v2.0
        Class mode.
    
    last confirmed at, 20240412 2025
    """
    
    roll_hl_cnt = 3
    
    try:
        # 1. self.config.tr_set.wave_period1
        if itv_to_number(self.config.tr_set.wave_itv1) > 1:
            offset = '1h' if self.config.tr_set.wave_itv1 != 'D' else '9h'
            htf_df_ = to_htf(self.df_res, self.config.tr_set.wave_itv1, offset=offset)  # to_htf 는 ohlc, 4개의 col 만 존재 (현재까지)
            htf_df = htf_df_[~pd.isnull(htf_df_.close)]

            htf_df = wave_range_cci_v4_1(htf_df, self.config.tr_set.wave_period1, itv=self.config.tr_set.wave_itv1)

            cols = list(htf_df.columns[4:])  # 15T_ohlc 를 제외한 wave_range_cci_v4 로 추가된 cols, 다 넣어버리기 (추후 혼란 방지)

            valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr = roll_wave_hl_idx_v5(htf_df,
                                                                                                           self.config.tr_set.wave_itv1,
                                                                                                           self.config.tr_set.wave_period1,
                                                                                                           roll_hl_cnt=roll_hl_cnt)

            """ 
            1. wave_bb 의 경우 roll_hl 의 기준이 co <-> cu 변경됨 (cci 와 비교)
            2. wave_bb : high_fill_ -> cu_prime_idx 사용
            """
            htf_df = get_roll_wave_data_v2(htf_df, valid_co_prime_idx, roll_co_idx_arr, 'wave_high_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1), roll_hl_cnt)
            cols += list(htf_df.columns[-roll_hl_cnt:])

            htf_df = get_roll_wave_data_v2(htf_df, valid_cu_prime_idx, roll_cu_idx_arr, 'wave_low_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1), roll_hl_cnt)
            cols += list(htf_df.columns[-roll_hl_cnt:])

            htf_df = wave_range_ratio_v4_2(htf_df, self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1, roll_hl_cnt=roll_hl_cnt)
            cols += list(htf_df.columns[-4:])

            self.df_res = get_wave_length(self.df_res, valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr, self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1, roll_hl_cnt=roll_hl_cnt)
            cols += list(htf_df.columns[-4:])

            # ------ 필요한 cols 만 join (htf's idx 정보는 ltf 와 sync. 가 맞지 않음 - join 불가함) ------ #
            self.df_res.drop(cols, inplace=True, axis=1, errors='ignore')
            self.df_res = self.df_res.join(to_lower_tf_v4(self.df_res, htf_df, cols, backing_i=0, ltf_itv='T'), how='inner')

        else:
            self.df_res = wave_range_cci_v4_1(self.df_res, self.config.tr_set.wave_period1, itv=self.config.tr_set.wave_itv1)

            valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr = roll_wave_hl_idx_v5(self.df_res,
                                                                                                           self.config.tr_set.wave_itv1,
                                                                                                           self.config.tr_set.wave_period1,
                                                                                                           roll_hl_cnt=roll_hl_cnt)

            self.df_res = get_roll_wave_data_v2(self.df_res, valid_co_prime_idx, roll_co_idx_arr, 'wave_high_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1), roll_hl_cnt)
            self.df_res = get_roll_wave_data_v2(self.df_res, valid_cu_prime_idx, roll_cu_idx_arr, 'wave_low_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1), roll_hl_cnt)

            self.df_res = wave_range_ratio_v4_2(self.df_res, self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1, roll_hl_cnt=roll_hl_cnt)

            self.df_res = get_wave_length(self.df_res, valid_co_prime_idx, valid_cu_prime_idx, roll_co_idx_arr, roll_cu_idx_arr, self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1, roll_hl_cnt=roll_hl_cnt)

                
    except Exception as e:
        self.sys_log.error("error in get_wave_info : {}".format(e))


def set_price_and_open_signal(self, mode='OPEN', env='BANK'):
    
    """
    v2.0
        Class mode.
    		use self, as parameter.
            modulize self.short_open_res1 *= phase.
            include np.timeidx
    v3.0
        turn off adj_wave_point. (messenger version)
        modify 'IDEP' to 'RIP'
    v3.1
        modify to TableCondition_v0.3
    
    last confirmed at, 20240610 1557.
    """
    
    self.len_df = len(self.df_res)
    self.np_timeidx = np.array([intmin_np(date_) for date_ in self.df_res.index.to_numpy()])     
    open_, high, low, close = [self.df_res[col_].to_numpy() for col_ in ['open', 'high', 'low', 'close']]
    
    self.set_price_box(self, )
    self.set_price(self, close)       
    self.get_reward_risk_ratio(self, ) # default
    
    if env == 'RIP':
        self.adj_price_unit(self,)

    # set_price_box need only for OPEN.
    # if mode == 'OPEN':        
        # self.get_open_res(self, )
        # self.adj_wave_point(self, )
        # self.adj_wave_update_hl_rejection(self, )
        # self.validate_price(self, close)    






def set_price_box(self, ):

    """
    v2.0
        use table_condition & messenger.

    last confirmed at, 20240529 0929.
    """  
    
    self.df_res['short_tp_1_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['short_tp_0_{}'.format(self.config.selection_id)] = self.price_stop_loss
    self.df_res['long_tp_1_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['long_tp_0_{}'.format(self.config.selection_id)] = self.price_stop_loss

    self.df_res['short_ep1_1_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['short_ep1_0_{}'.format(self.config.selection_id)] = self.price_stop_loss
    self.df_res['long_ep1_1_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['long_ep1_0_{}'.format(self.config.selection_id)] = self.price_stop_loss

    # --> p2's self.price_entry use p1's self.price_entry
    self.df_res['short_ep2_1_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['short_ep2_0_{}'.format(self.config.selection_id)] = self.price_stop_loss
    self.df_res['long_ep2_1_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['long_ep2_0_{}'.format(self.config.selection_id)] = self.price_stop_loss

    # --> self.price_stop_loss use p1's low, (allow prev_low as self.price_stop_loss for p1_hhm only)
    self.df_res['short_out_1_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['short_out_0_{}'.format(self.config.selection_id)] = self.price_stop_loss
    self.df_res['long_out_1_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['long_out_0_{}'.format(self.config.selection_id)] = self.price_stop_loss

    
    # gap
    self.df_res['short_tp_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['short_tp_1_{}'.format(self.config.selection_id)] - self.df_res['short_tp_0_{}'.format(self.config.selection_id)])
    self.df_res['long_tp_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['long_tp_1_{}'.format(self.config.selection_id)] - self.df_res['long_tp_0_{}'.format(self.config.selection_id)])
    self.df_res['short_ep1_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['short_ep1_1_{}'.format(self.config.selection_id)] - self.df_res['short_ep1_0_{}'.format(self.config.selection_id)])
    self.df_res['long_ep1_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['long_ep1_1_{}'.format(self.config.selection_id)] - self.df_res['long_ep1_0_{}'.format(self.config.selection_id)])

    self.df_res['short_out_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['short_out_1_{}'.format(self.config.selection_id)] - self.df_res['short_out_0_{}'.format(self.config.selection_id)])
    self.df_res['long_out_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['long_out_1_{}'.format(self.config.selection_id)] - self.df_res['long_out_0_{}'.format(self.config.selection_id)])
    self.df_res['short_ep2_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['short_ep2_1_{}'.format(self.config.selection_id)] - self.df_res['short_ep2_0_{}'.format(self.config.selection_id)])
    self.df_res['long_ep2_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['long_ep2_1_{}'.format(self.config.selection_id)] - self.df_res['long_ep2_0_{}'.format(self.config.selection_id)])

    
    # gap
    self.df_res['short_tp_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['short_tp_1_{}'.format(self.config.selection_id)] - self.df_res['short_tp_0_{}'.format(self.config.selection_id)])
    self.df_res['long_tp_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['long_tp_1_{}'.format(self.config.selection_id)] - self.df_res['long_tp_0_{}'.format(self.config.selection_id)])
    self.df_res['short_ep1_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['short_ep1_1_{}'.format(self.config.selection_id)] - self.df_res['short_ep1_0_{}'.format(self.config.selection_id)])
    self.df_res['long_ep1_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['long_ep1_1_{}'.format(self.config.selection_id)] - self.df_res['long_ep1_0_{}'.format(self.config.selection_id)])

    self.df_res['short_out_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['short_out_1_{}'.format(self.config.selection_id)] - self.df_res['short_out_0_{}'.format(self.config.selection_id)])
    self.df_res['long_out_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['long_out_1_{}'.format(self.config.selection_id)] - self.df_res['long_out_0_{}'.format(self.config.selection_id)])
    self.df_res['short_ep2_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['short_ep2_1_{}'.format(self.config.selection_id)] - self.df_res['short_ep2_0_{}'.format(self.config.selection_id)])
    self.df_res['long_ep2_gap_{}'.format(self.config.selection_id)] = abs(self.df_res['long_ep2_1_{}'.format(self.config.selection_id)] - self.df_res['long_ep2_0_{}'.format(self.config.selection_id)])

    
    # spread
    self.df_res['short_spread_{}'.format(self.config.selection_id)] = (self.df_res['short_tp_0_{}'.format(self.config.selection_id)].to_numpy() / self.df_res['short_tp_1_{}'.format(self.config.selection_id)].to_numpy() - 1) / 2
    self.df_res['long_spread_{}'.format(self.config.selection_id)] = (self.df_res['long_tp_1_{}'.format(self.config.selection_id)].to_numpy() / self.df_res['long_tp_0_{}'.format(self.config.selection_id)].to_numpy() - 1) / 2

    
    # # lvrg_needed is used in get_price v2.0
    # if not pd.isnull(self.table_condition_row.lvrg_k):        
    #     self.df_res['short_lvrg_needed_{}'.format(self.config.selection_id)] = (1 / self.df_res['short_spread_{}'.format(self.config.selection_id)]) * self.table_condition_row.lvrg_k
    #     self.df_res['long_lvrg_needed_{}'.format(self.config.selection_id)] = (1 / self.df_res['long_spread_{}'.format(self.config.selection_id)]) * self.table_condition_row.lvrg_k

    #     if not pd.isnull(self.table_condition_row.lvrg_ceiling):            
    #         if self.table_condition_row.lvrg_ceiling:                
    #             self.df_res['short_lvrg_needed_{}'.format(self.config.selection_id)][self.df_res['short_lvrg_needed_{}'.format(self.config.selection_id)] > self.table_condition_row.lvrg_max_short] = self.table_condition_row.lvrg_max_short
    #             self.df_res['long_lvrg_needed_{}'.format(self.config.selection_id)][self.df_res['long_lvrg_needed_{}'.format(self.config.selection_id)] > self.table_condition_row.lvrg_max_long] = self.table_condition_row.lvrg_max_long
                



def set_price(self, close):    
    
    """
    v2.0
        use table_condition & messenger.

    last confirmed at, 20240528 1422.
    """        
    
    # price_take_profit 
        # i. 기준 : balance_available_1, gap : balance_available_box
    self.df_res['short_tp_{}'.format(self.config.selection_id)] = self.price_take_profit
    self.df_res['long_tp_{}'.format(self.config.selection_id)] = self.price_take_profit

    
    # price_entry
        # limit
    if self.config.ep_set.entry_type == "LIMIT": 
            # 1. 기준 : ep1_0, gap : ep1_box
        self.df_res['short_ep1_{}'.format(self.config.selection_id)] = self.price_entry
        self.df_res['long_ep1_{}'.format(self.config.selection_id)] = self.price_entry
        
        # market
    else:
        self.df_res['short_ep1_{}'.format(self.config.selection_id)] = close
        self.df_res['long_ep1_{}'.format(self.config.selection_id)] = close    
    
    
    # price_entry2
        # limit
    if self.config.ep_set.point2.entry_type == "LIMIT":
            # 1. 기준 : ep2_0, gap : ep2_box
        self.df_res['short_ep2_{}'.format(self.config.selection_id)] = self.df_res['short_ep2_1_{}'.format(self.config.selection_id)].to_numpy() + self.df_res['short_ep2_gap_{}'.format(self.config.selection_id)].to_numpy() * self.config.tr_set.ep2_gap
        self.df_res['long_ep2_{}'.format(self.config.selection_id)] = self.df_res['long_ep2_1_{}'.format(self.config.selection_id)].to_numpy() - self.df_res['long_ep2_gap_{}'.format(self.config.selection_id)].to_numpy() * self.config.tr_set.ep2_gap        
        
        # market
    else:
        self.df_res['short_ep2_{}'.format(self.config.selection_id)] = close
        self.df_res['long_ep2_{}'.format(self.config.selection_id)] = close

    
    # price_stop_loss
    if self.config.tr_set.check_hlm == 0:
        # ii. 기준 : out_0, gap : out_box
        self.df_res['short_out_{}'.format(self.config.selection_id)] = self.price_stop_loss
        self.df_res['long_out_{}'.format(self.config.selection_id)] = self.price_stop_loss

    elif self.config.tr_set.check_hlm == 1:
        self.df_res['short_out_{}'.format(self.config.selection_id)] = self.df_res['short_ep1_0_{}'.format(self.config.selection_id)].to_numpy() + self.df_res['short_ep1_gap_{}'.format(self.config.selection_id)].to_numpy() * self.config.tr_set.out_gap  
        self.df_res['long_out_{}'.format(self.config.selection_id)] = self.df_res['long_ep1_0_{}'.format(self.config.selection_id)].to_numpy() - self.df_res['long_ep1_gap_{}'.format(self.config.selection_id)].to_numpy() * self.config.tr_set.out_gap  

    else:  # p2_hlm
        # ii. 기준 : ep2_0, gap : ep2_box
        self.df_res['short_out_{}'.format(self.config.selection_id)] = self.df_res['short_ep2_1_{}'.format(self.config.selection_id)].to_numpy() + self.df_res['short_ep2_gap_{}'.format(self.config.selection_id)].to_numpy() * self.config.tr_set.ep2_gap  # * 5
        self.df_res['long_out_{}'.format(self.config.selection_id)] = self.df_res['long_ep2_1_{}'.format(self.config.selection_id)].to_numpy() - self.df_res['long_ep2_gap_{}'.format(self.config.selection_id)].to_numpy() * self.config.tr_set.ep2_gap  # * 5




def adj_price_unit(self):   
    
    calc_with_hoga_unit_vecto = np.vectorize(calc_with_precision)
    self.df_res['short_tp_{}'.format(self.config.selection_id)] = calc_with_hoga_unit_vecto(self.df_res['short_tp_{}'.format(self.config.selection_id)].to_numpy(), 2)
    self.df_res['long_tp_{}'.format(self.config.selection_id)] = calc_with_hoga_unit_vecto(self.df_res['long_tp_{}'.format(self.config.selection_id)].to_numpy(), 2)
    self.df_res['short_ep1_{}'.format(self.config.selection_id)] = calc_with_hoga_unit_vecto(self.df_res['short_ep1_{}'.format(self.config.selection_id)].to_numpy(), 2)
    self.df_res['long_ep1_{}'.format(self.config.selection_id)] = calc_with_hoga_unit_vecto(self.df_res['long_ep1_{}'.format(self.config.selection_id)].to_numpy(), 2)
    self.df_res['short_ep2_{}'.format(self.config.selection_id)] = calc_with_hoga_unit_vecto(self.df_res['short_ep2_{}'.format(self.config.selection_id)].to_numpy(), 2)
    self.df_res['long_ep2_{}'.format(self.config.selection_id)] = calc_with_hoga_unit_vecto(self.df_res['long_ep2_{}'.format(self.config.selection_id)].to_numpy(), 2)
    self.df_res['short_out_{}'.format(self.config.selection_id)] = calc_with_hoga_unit_vecto(self.df_res['short_out_{}'.format(self.config.selection_id)].to_numpy(), 2)
    self.df_res['long_out_{}'.format(self.config.selection_id)] = calc_with_hoga_unit_vecto(self.df_res['long_out_{}'.format(self.config.selection_id)].to_numpy(), 2)




def get_open_res(self, ):
    
    #    a. init open_res
    self.short_open_res1 = np.ones(self.len_df)  # .astype(object)
    self.long_open_res1 = np.ones(self.len_df)  # .astype(object)
    self.short_open_res2 = np.ones(self.len_df)  # .astype(object)
    self.long_open_res2 = np.ones(self.len_df)  # .astype(object)

    if self.config.show_detail:
        self.sys_log.info("init open_res")
        self.sys_log.info("np.sum(self.short_open_res1 == 1) : {}".format(np.sum(self.short_open_res1 == 1)))
        self.sys_log.info("np.sum(self.long_open_res1 == 1) : {}".format(np.sum(self.long_open_res1 == 1)))




def adj_wave_point(self, ):
    
    #    a. wave_point
    # notnan_short_tc = ~pd.isnull(self.df_res['short_tc_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy())  # isnull for object
    # notnan_long_tc = ~pd.isnull(self.df_res['long_tc_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy())  # isnull for object

    notnan_cu = ~pd.isnull(self.df_res['wave_cu_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy())  # isnull for object
    notnan_co = ~pd.isnull(self.df_res['wave_co_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy())
    # notnan_cu2 = ~pd.isnull(self.df_res['wave_cu_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy())  # isnull for object
    # notnan_co2 = ~pd.isnull(self.df_res['wave_co_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy())

    self.short_open_res1 *= self.df_res['wave_cu_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy().astype(bool) * notnan_cu  # object로 변환되는 경우에 대응해, bool 로 재정의
    self.long_open_res1 *= self.df_res['wave_co_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy().astype(bool) * notnan_co  # np.nan = bool type 으로 True 임..
    # self.short_open_res1 *= self.df_res['short_tc_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy().astype(bool) * notnan_short_tc
    # self.long_open_res1 *= self.df_res['long_tc_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy().astype(bool) * notnan_long_tc

    # self.short_open_res2 *= self.df_res['wave_cu_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy().astype(bool) * notnan_cu2  # object로 변환되는 경우에 대응해, bool 로 재정의
    # self.long_open_res2 *= self.df_res['wave_co_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy().astype(bool) * notnan_co2  # np.nan = bool type 으로 True 임..
    # self.short_open_res2 *= self.df_res['short_tc_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.tc_period)].to_numpy()
    # self.long_open_res2 *= self.df_res['long_tc_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.tc_period)].to_numpy()

    if self.config.show_detail:
        self.sys_log.info("wave_point")
        self.sys_log.info("np.sum(self.short_open_res1 == 1) : {}".format(np.sum(self.short_open_res1 == 1)))
        self.sys_log.info("np.sum(self.long_open_res1 == 1) : {}".format(np.sum(self.long_open_res1 == 1)))
        # self.sys_log.info("np.sum(self.short_open_res2 == 1) : {}".format(np.sum(self.short_open_res2 == 1)))
        # self.sys_log.info("np.sum(self.long_open_res2 == 1) : {}".format(np.sum(self.long_open_res2 == 1)))





def adj_wave_update_hl_rejection(self, ):
    
    notnan_update_low_cu = ~pd.isnull(self.df_res['wave_update_low_cu_bool_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy())
    notnan_update_high_co = ~pd.isnull(self.df_res['wave_update_high_co_bool_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy())
    # notnan_update_low_cu2 = ~pd.isnull(self.df_res['wave_update_low_cu_bool_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy())
    # notnan_update_high_co2 = ~pd.isnull(self.df_res['wave_update_high_co_bool_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy())

    self.short_open_res1 *= ~(self.df_res['wave_update_low_cu_bool_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy().astype(bool)) * notnan_update_low_cu
    self.long_open_res1 *= ~(self.df_res['wave_update_high_co_bool_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy().astype(bool)) * notnan_update_high_co
    # self.short_open_res2 *= ~(self.df_res['wave_update_low_cu_bool_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy().astype(bool)) * notnan_update_low_cu2
    # self.long_open_res2 *= ~(self.df_res['wave_update_high_co_bool_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy().astype(bool)) * notnan_update_high_co2

    if self.config.show_detail:
        self.sys_log.info("reject update_hl")
        self.sys_log.info("np.sum(self.short_open_res1 == 1) : {}".format(np.sum(self.short_open_res1 == 1)))
        self.sys_log.info("np.sum(self.long_open_res1 == 1) : {}".format(np.sum(self.long_open_res1 == 1)))
        # self.sys_log.info("np.sum(self.short_open_res2 == 1) : {}".format(np.sum(self.short_open_res2 == 1)))
        # self.sys_log.info("np.sum(self.long_open_res2 == 1) : {}".format(np.sum(self.long_open_res2 == 1)))



def adj_wave_itv(self, ):
    
    if self.config.tr_set.wave_itv1 != 'T':
        wave_itv1_num = itv_to_number(self.config.tr_set.wave_itv1)
        self.short_open_res1 *= self.np_timeidx % wave_itv1_num == (wave_itv1_num - 1)
        self.long_open_res1 *= self.np_timeidx % wave_itv1_num == (wave_itv1_num - 1)

        if self.config.show_detail:
            self.sys_log.info("self.config.tr_set.wave_itv1")
            self.sys_log.info("np.sum(self.short_open_res1 == 1) : {}".format(np.sum(self.short_open_res1 == 1)))
            self.sys_log.info("np.sum(self.long_open_res1 == 1) : {}".format(np.sum(self.long_open_res1 == 1)))

    if self.config.tr_set.wave_itv2 != 'T':
        wave_itv2_num = itv_to_number(self.config.tr_set.wave_itv2)
        self.short_open_res2 *= self.np_timeidx % wave_itv2_num == (wave_itv2_num - 1)
        self.long_open_res2 *= self.np_timeidx % wave_itv2_num == (wave_itv2_num - 1)

        if self.config.show_detail:
            self.sys_log.info("self.config.tr_set.wave_itv2")
            self.sys_log.info("np.sum(self.short_open_res2 == 1) : {}".format(np.sum(self.short_open_res2 == 1)))
            self.sys_log.info("np.sum(self.long_open_res2 == 1) : {}".format(np.sum(self.long_open_res2 == 1)))



def adj_wave_term(self, ):
    
    wave_high_terms_cnt_fill1_ = self.df_res['wave_high_terms_cnt_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
    wave_low_terms_cnt_fill1_ = self.df_res['wave_low_terms_cnt_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()

    self.short_open_res1 *= (wave_high_terms_cnt_fill1_ > self.config.tr_set.wave_greater2) & (wave_low_terms_cnt_fill1_ > self.config.tr_set.wave_greater1)
    self.long_open_res1 *= (wave_low_terms_cnt_fill1_ > self.config.tr_set.wave_greater2) & (wave_high_terms_cnt_fill1_ > self.config.tr_set.wave_greater1)

    # wave_high_terms_cnt_fill2_ = self.df_res['wave_high_terms_cnt_fill_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy()
    # wave_low_terms_cnt_fill2_ = self.df_res['wave_low_terms_cnt_fill_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy()

    # self.short_open_res2 *= (wave_high_terms_cnt_fill2_ > self.config.tr_set.wave_greater2) & (wave_low_terms_cnt_fill2_ > self.config.tr_set.wave_greater1)
    # self.long_open_res2 *= (wave_low_terms_cnt_fill2_ > self.config.tr_set.wave_greater2) & (wave_high_terms_cnt_fill2_ > self.config.tr_set.wave_greater1)

    if self.config.show_detail:
        self.sys_log.info("wave_term")
        self.sys_log.info("np.sum(self.short_open_res1 == 1) : {}".format(np.sum(self.short_open_res1 == 1)))
        self.sys_log.info("np.sum(self.long_open_res1 == 1) : {}".format(np.sum(self.long_open_res1 == 1)))
        # self.sys_log.info("np.sum(self.short_open_res2 == 1) : {}".format(np.sum(self.short_open_res2 == 1)))
        # self.sys_log.info("np.sum(self.long_open_res2 == 1) : {}".format(np.sum(self.long_open_res2 == 1)))



def adj_wave_length(self, ):
    
    if self.config.tr_set.wave_length_min_short1 != "None":
        short_wave_length_fill_ = self.df_res['short_wave_length_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
        self.short_open_res1 *= short_wave_length_fill_ >= self.config.tr_set.wave_length_min_short1
        if self.config.show_detail:
            self.sys_log.info("wave_length_min_short1")
            self.sys_log.info("np.sum(self.short_open_res1 == 1) : {}".format(np.sum(self.short_open_res1 == 1)))

    if self.config.tr_set.wave_length_max_short1 != "None":
        short_wave_length_fill_ = self.df_res['short_wave_length_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
        self.short_open_res1 *= short_wave_length_fill_ <= self.config.tr_set.wave_length_max_short1
        if self.config.show_detail:
            self.sys_log.info("wave_length_max_short1")
            self.sys_log.info("np.sum(self.short_open_res1 == 1) : {}".format(np.sum(self.short_open_res1 == 1)))

    if self.config.tr_set.wave_length_min_long1 != "None":
        long_wave_length_fill_ = self.df_res['long_wave_length_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
        self.long_open_res1 *= long_wave_length_fill_ >= self.config.tr_set.wave_length_min_long1
        if self.config.show_detail:
            self.sys_log.info("wave_length_min_long1")
            self.sys_log.info("np.sum(self.long_open_res1 == 1) : {}".format(np.sum(self.long_open_res1 == 1)))

    if self.config.tr_set.wave_length_max_long1 != "None":
        long_wave_length_fill_ = self.df_res['long_wave_length_fill_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
        self.long_open_res1 *= long_wave_length_fill_ <= self.config.tr_set.wave_length_max_long1
        if self.config.show_detail:
            self.sys_log.info("wave_length_max_long1")
            self.sys_log.info("np.sum(self.long_open_res1 == 1) : {}".format(np.sum(self.long_open_res1 == 1)))



def validate_price(self, close):

    short_tp_ = self.df_res['short_tp_{}'.format(self.config.selection_id)].to_numpy()
    short_ep1_ = self.df_res['short_ep1_{}'.format(self.config.selection_id)].to_numpy()
    short_ep2_ = self.df_res['short_ep2_{}'.format(self.config.selection_id)].to_numpy()
    short_out_ = self.df_res['short_out_{}'.format(self.config.selection_id)].to_numpy()

    long_tp_ = self.df_res['long_tp_{}'.format(self.config.selection_id)].to_numpy()
    long_ep1_ = self.df_res['long_ep1_{}'.format(self.config.selection_id)].to_numpy()
    long_ep2_ = self.df_res['long_ep2_{}'.format(self.config.selection_id)].to_numpy()
    long_out_ = self.df_res['long_out_{}'.format(self.config.selection_id)].to_numpy()

    
    #     a. p1 validate_price
    #             i. tr_set validation reject nan data
    #             ii. 정상 거래 위한 self.price_take_profit > self.price_entry           
    
    self.short_open_res1 *= (short_tp_ < short_ep1_) & (short_ep1_ < short_out_)
    #             iii. reject hl_out exec_openution -> close always < ep1_0 at wave_p1
    self.short_open_res1 *= close < short_out_  # self.df_res['short_ep1_0_{}'.format(self.config.selection_id)].to_numpy()
    # self.short_open_res1 *= close < short_ep1_   # reject entry exec_openution
    # short_out_  self.df_res['short_tp_0_{}'.format(self.config.selection_id)].to_numpy() self.df_res['short_ep1_0_{}'.format(self.config.selection_id)].to_numpy()

    self.long_open_res1 *= (long_tp_ > long_ep1_) & (long_ep1_ > long_out_)  # (long_tp_ > long_ep_)
    self.long_open_res1 *= close > long_out_  # self.df_res['long_ep1_0_{}'.format(self.config.selection_id)].to_numpy()
    # self.long_open_res1 *= close > long_ep1_  # reject entry exec_openution
    # long_out_ self.df_res['long_tp_0_{}'.format(self.config.selection_id)].to_numpy() self.df_res['long_ep1_0_{}'.format(self.config.selection_id)].to_numpy()

    
    #     b. p2 validate_price => deprecated => now, executed in en_ex_pairing() function.
    # self.short_open_res2 *= (short_ep2_ < short_out_) # tr_set validation (short_tp_ < short_ep_) # --> p2_box location (cannot be vectorized)
    # self.short_open_res2 *= close < short_out_    # reject hl_out exec_openution
    # self.long_open_res2 *= (long_ep2_ > long_out_)  # tr_set validation (long_tp_ > long_ep_) &   # p2's self.price_entry & self.price_stop_loss cannot be vectorized
    # self.long_open_res2 *= close > long_out_    # reject hl_out exec_openution


    self.df_res['short_open1_{}'.format(self.config.selection_id)] = self.short_open_res1 * (not self.config.pos_set.short_ban)
    self.df_res['long_open1_{}'.format(self.config.selection_id)] = self.long_open_res1 * (not self.config.pos_set.long_ban)

    self.df_res['short_open2_{}'.format(self.config.selection_id)] = self.short_open_res2
    self.df_res['long_open2_{}'.format(self.config.selection_id)] = self.long_open_res2

    if self.config.show_detail:
        self.sys_log.info("point validation")
        self.sys_log.info("np.sum(self.short_open_res1 == 1) : {}".format(np.sum(self.short_open_res1 == 1)))
        self.sys_log.info("np.sum(self.long_open_res1 == 1) : {}".format(np.sum(self.long_open_res1 == 1)))
        # self.sys_log.info("np.sum(self.short_open_res2 == 1) : {}".format(np.sum(self.short_open_res2 == 1)))
        # self.sys_log.info("np.sum(self.long_open_res2 == 1) : {}".format(np.sum(self.long_open_res2 == 1)))




def get_reward_risk_ratio(self, unit_RRratio_adj_fee=np.arange(0, 2, 0.1)):

    """
    v2.0
        modify to RRratio.
            apply updated formula.
        no more position_tp / ep1 / out _{}

    last confirmed at, 20240610 1552.
    """

    fee_entry = self.config.trader_set.fee_market # can be replaced later.
    fee_exit = self.config.trader_set.fee_market
    
    self.df_res['RRratio'] = abs(self.price_take_profit - self.price_entry) / abs(self.price_entry - self.price_stop_loss)
    self.df_res['RRratio_adj_fee'] = (abs(self.price_take_profit - self.price_entry) - (self.price_entry * fee_entry + self.price_take_profit * fee_exit)) / (abs(self.price_entry - self.price_stop_loss) + (self.price_entry * fee_entry + self.price_stop_loss * fee_exit))
        
    # unit_RRratio_adj_fee = np.arange(0, 1, 0.1)
    self.df_res['RRratio_adj_fee_category'] = pd.cut(self.df_res['RRratio_adj_fee'], unit_RRratio_adj_fee, precision=0, duplicates='drop').astype(str)




def get_side_open(self, open_num=1):

    """
    v2.0    
        rm utils_list, config_list
    v2.1 
        self.side_open is init in loop_table_condition.

    last confirmed at, 20240528 1531.
    """
    
    # point
        # 여기 왜 latest_index 사용했는지 모르겠음.
            # --> bar_close point 를 고려해 mr_res 와의 index 분리
        # vecto. 하는 이상, point x ep_loc 의 index 는 sync. 되어야함 => complete_index 사용 (solved)
    
    if self.df_res['short_open{}_{}'.format(open_num, self.config.selection_id)].to_numpy()[self.config.trader_set.complete_index]:        
        self.sys_log.info("[ short ] true_point selection_id : {}".format(self.config.selection_id))

        # ban
        if self.config.pos_set.short_ban:
            self.sys_log.info("self.config.pos_set.short_ban : {}".format(self.config.pos_set.short_ban))
        else:
            # mr_res
            if open_num == 1:
                mr_res, zone_arr = self.entry_point_location(self, ep_loc_side=OrderSide.SELL)
            else:
                mr_res, zone_arr = self.entry_point2_location(self, ep_loc_side=OrderSide.SELL)

            # assign
            if mr_res[self.config.trader_set.complete_index]:
                self.side_open = OrderSide.SELL

    # point
    if self.df_res['long_open{}_{}'.format(open_num, self.config.selection_id)].to_numpy()[self.config.trader_set.complete_index]:    
    # if 1:          
        self.sys_log.info("[ long ] true_point selection_id : {}".format(self.config.selection_id))

        # ban
        if self.config.pos_set.long_ban:  # additional const. on trader
            self.sys_log.info("self.config.pos_set.long_ban : {}".format(self.config.pos_set.long_ban))
        else:
            # mr_res
            if open_num == 1:
                mr_res, zone_arr = self.entry_point_location(self, ep_loc_side=OrderSide.BUY)
            else:
                mr_res, zone_arr = self.entry_point2_location(self, ep_loc_side=OrderSide.BUY)

            # assign
            if mr_res[self.config.trader_set.complete_index]:
                self.side_open = OrderSide.BUY





def entry_point_location(self, ep_loc_side=OrderSide.SELL):
    
    """
    v3.0
        vectorized calc.
            multi-stem 에 따라 dynamic vars.가 입력되기 때문에 class 내부 vars. 로 종속시키지 않음
            min & max variables 사용
    v4.0
        Class mode.
    v4.1
        table_condition init with np.nan.
        
    last confirmed at, 20240529 0946.
    """

    # param init
    # self.len_df = len(self.df_res)
    mr_res = np.ones(self.len_df)
    zone_arr = np.full(self.len_df, 0)

    
    # WRR32
    if not pd.isnull(self.table_condition_row.wrr_32_min_short):
        if ep_loc_side == OrderSide.SELL:
            cu_wrr_32_ = self.df_res['cu_wrr_32_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
            mr_res *= cu_wrr_32_ >= self.table_condition_row.wrr_32_min_short
            if self.config.show_detail:
                self.sys_log.info(
                    "cu_wrr_32_ >= self.table_condition_row.wrr_32_min_short : {:.5f} {:.5f} ({})".format(cu_wrr_32_[self.config.trader_set.complete_index],
                                                                                                       self.table_condition_row.wrr_32_min_short,
                                                                                                       mr_res[self.config.trader_set.complete_index]))
    if not pd.isnull(self.table_condition_row.wrr_32_min_long):
        if ep_loc_side == OrderSide.BUY:
            co_wrr_32_ = self.df_res['co_wrr_32_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
            mr_res *= co_wrr_32_ >= self.table_condition_row.wrr_32_min_long
            if self.config.show_detail:
                self.sys_log.info(
                    "co_wrr_32_ >= self.table_condition_row.wrr_32_min_long : {:.5f} {:.5f} ({})".format(co_wrr_32_[self.config.trader_set.complete_index],
                                                                                                      self.table_condition_row.wrr_32_min_long,
                                                                                                      mr_res[self.config.trader_set.complete_index]))
    if not pd.isnull(self.table_condition_row.wrr_32_max_short):
        if ep_loc_side == OrderSide.SELL:
            cu_wrr_32_ = self.df_res['cu_wrr_32_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
            mr_res *= cu_wrr_32_ <= self.table_condition_row.wrr_32_max_short
            if self.config.show_detail:
                self.sys_log.info(
                    "cu_wrr_32_ <= self.table_condition_row.wrr_32_max_short : {:.5f} {:.5f} ({})".format(cu_wrr_32_[self.config.trader_set.complete_index],
                                                                                                       self.table_condition_row.wrr_32_max_short,
                                                                                                       mr_res[self.config.trader_set.complete_index]))
    if not pd.isnull(self.table_condition_row.wrr_32_max_long):
        if ep_loc_side == OrderSide.BUY:
            co_wrr_32_ = self.df_res['co_wrr_32_{}{}'.format(self.config.tr_set.wave_itv1, self.config.tr_set.wave_period1)].to_numpy()
            mr_res *= co_wrr_32_ <= self.table_condition_row.wrr_32_max_long
            if self.config.show_detail:
                self.sys_log.info(
                    "co_wrr_32_ <= self.table_condition_row.wrr_32_max_long : {:.5f} {:.5f} ({})".format(co_wrr_32_[self.config.trader_set.complete_index],
                                                                                                      self.table_condition_row.wrr_32_max_long,
                                                                                                      mr_res[self.config.trader_set.complete_index]))
                

    # spread
    if not pd.isnull(self.table_condition_row.spread_min_short):
        if ep_loc_side == OrderSide.SELL:
            short_spread_ = self.df_res['short_spread_{}'.format(self.config.selection_id)].to_numpy()
            mr_res *= short_spread_ >= self.table_condition_row.spread_min_short
            if self.config.show_detail:
                self.sys_log.info(
                    "short_spread_ >= self.table_condition_row.spread_min_short : {:.5f} {:.5f} ({})".format(short_spread_[self.config.trader_set.complete_index], 
                                                                                                             self.table_condition_row.spread_min_short,
                                                                                                             mr_res[self.config.trader_set.complete_index]))

    if not pd.isnull(self.table_condition_row.spread_min_long):
        if ep_loc_side == OrderSide.BUY:
            long_spread_ = self.df_res['long_spread_{}'.format(self.config.selection_id)].to_numpy()
            mr_res *= long_spread_ >= self.table_condition_row.spread_min_long
            if self.config.show_detail:
                self.sys_log.info(
                    "long_spread_ >= self.table_condition_row.spread_min_long : {:.5f} {:.5f} ({})".format(long_spread_[self.config.trader_set.complete_index], 
                                                                                                           self.table_condition_row.spread_min_long, 
                                                                                                           mr_res[self.config.trader_set.complete_index]))

    if not pd.isnull(self.table_condition_row.spread_max_short):
        if ep_loc_side == OrderSide.SELL:
            short_spread_ = self.df_res['short_spread_{}'.format(self.config.selection_id)].to_numpy()
            mr_res *= short_spread_ <= self.table_condition_row.spread_max_short
            if self.config.show_detail:
                self.sys_log.info(
                    "short_spread_ <= self.table_condition_row.spread_max_short : {:.5f} {:.5f} ({})".format(short_spread_[self.config.trader_set.complete_index], 
                                                                                                             self.table_condition_row.spread_max_short,
                                                                                                             mr_res[self.config.trader_set.complete_index]))

    if not pd.isnull(self.table_condition_row.spread_max_long):
        if ep_loc_side == OrderSide.BUY:
            long_spread_ = self.df_res['long_spread_{}'.format(self.config.selection_id)].to_numpy()
            mr_res *= long_spread_ <= self.table_condition_row.spread_max_long
            if self.config.show_detail:
                self.sys_log.info(
                    "long_spread_ <= self.table_condition_row.spread_max_long : {:.5f} {:.5f} ({})".format(long_spread_[self.config.trader_set.complete_index], 
                                                                                                           self.table_condition_row.spread_max_long,
                                                                                                           mr_res[self.config.trader_set.complete_index]))

                
    # zone
    if self.config.loc_set.zone1.use_zone:
        pass
        
    return mr_res, zone_arr  # mr_res 의 True idx 가 open signal
    
    
def entry_point2_location(self, ep_loc_side=OrderSide.SELL):
    
    """
    v3.0
        vectorized calc.
            multi-stem 에 따라 dynamic vars.가 입력되기 때문에 Class 내부 vars. 로 종속시키지 않음
            min & max variables 사용
    v4.0
        Class mode.
        
    last confirmed at, 20240412 2125
    """
    
    # 0. param init
    # self.len_df = len(self.df_res)
    mr_res = np.ones(self.len_df)
    zone_arr = np.full(self.len_df, 0)

    
    # 1. WRR32
    if self.config.loc_set.point2.wrr_32_min != "None":
        if ep_loc_side == OrderSide.SELL:
            cu_wrr_32_ = self.df_res['cu_wrr_32_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy()
            mr_res *= cu_wrr_32_ >= self.config.loc_set.point2.wrr_32_min
            if self.config.show_detail:
                self.sys_log.info(
                    "cu_wrr_32_ >= self.config.loc_set.point2.wrr_32_min : {:.5f} {:.5f} ({})".format(cu_wrr_32_[self.config.trader_set.complete_index],
                                                                                                 self.config.loc_set.point2.wrr_32_min,
                                                                                                 mr_res[self.config.trader_set.complete_index]))
        else:
            co_wrr_32_ = self.df_res['co_wrr_32_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy()
            mr_res *= co_wrr_32_ >= self.config.loc_set.point2.wrr_32_min
            if self.config.show_detail:
                self.sys_log.info(
                    "co_wrr_32_ >= self.config.loc_set.point2.wrr_32_min : {:.5f} {:.5f} ({})".format(co_wrr_32_[self.config.trader_set.complete_index],
                                                                                                 self.config.loc_set.point2.wrr_32_min,
                                                                                                 mr_res[self.config.trader_set.complete_index]))
    if self.config.loc_set.point2.wrr_32_max != "None":
        if ep_loc_side == OrderSide.SELL:
            cu_wrr_32_ = self.df_res['cu_wrr_32_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy()
            mr_res *= cu_wrr_32_ <= self.config.loc_set.point2.wrr_32_max
            if self.config.show_detail:
                self.sys_log.info(
                    "cu_wrr_32_ <= self.config.loc_set.point2.wrr_32_max : {:.5f} {:.5f} ({})".format(cu_wrr_32_[self.config.trader_set.complete_index],
                                                                                                 self.config.loc_set.point2.wrr_32_max,
                                                                                                 mr_res[self.config.trader_set.complete_index]))
        else:
            co_wrr_32_ = self.df_res['co_wrr_32_{}{}'.format(self.config.tr_set.wave_itv2, self.config.tr_set.wave_period2)].to_numpy()
            mr_res *= co_wrr_32_ <= self.config.loc_set.point2.wrr_32_max
            if self.config.show_detail:
                self.sys_log.info(
                    "co_wrr_32_ <= self.config.loc_set.point2.wrr_32_max : {:.5f} {:.5f} ({})".format(co_wrr_32_[self.config.trader_set.complete_index],
                                                                                                 self.config.loc_set.point2.wrr_32_max,
                                                                                                 mr_res[self.config.trader_set.complete_index]))

    # 2. zone.
    if self.config.loc_set.zone2.use_zone:
        
        cci_ = self.df_res['cci_30T20'].to_numpy()
        b1_cci_ = self.df_res['cci_30T20'].shift(30).to_numpy()
        # base_value = -100

        if ep_loc_side == OrderSide.SELL:
            # mr_res *= cci_ < 0
            mr_res *= cci_ < b1_cci_
            # mr_res *= (cci_ > -100) & (cci_ < -80)
            if self.config.show_detail:
                self.sys_log.info("cci_ < b1_cci_ : {:.5f} {:.5f} ({})".format(cci_[self.config.trader_set.complete_index], 
                                                                             b1_cci_[self.config.trader_set.complete_index], 
                                                                             mr_res[self.config.trader_set.complete_index]))
        else:
            # mr_res *= cci_ < 0
            mr_res *= cci_ > b1_cci_
            # mr_res *= (cci_ > 80) & (cci_ < 100)
            if self.config.show_detail:
                self.sys_log.info("cci_ > b1_cci_ : {:.5f} {:.5f} ({})".format(cci_[self.config.trader_set.complete_index], 
                                                                             b1_cci_[self.config.trader_set.complete_index], 
                                                                             mr_res[self.config.trader_set.complete_index]))

    return mr_res, zone_arr  # mr_res 의 True idx 가 open signal



def get_balance_available(self, asset_type='USDT'):
    
    try:      
        server_time = self.time()['serverTime']
        response = self.balance(recvWindow=6000, timestamp=server_time)
    except Exception as e:
        msg = "error in get_balance() : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
    else:
        available_asset = float([res['availableBalance'] for res in response if res['asset'] == asset_type][0])
        self.balance_available = self.calc_with_precision(available_asset, 2) # default for Binance
        


def get_balance_info(self, balance_min, mode="PROD"):

    """
    v2.0
        reject balance insufficient period. 

    last confirmed at, 20240524 2021.
    """

    self.balance_insufficient = 0

    # get balance_available
    if not self.config.trader_set.backtrade:
        
        self.get_balance_available(self)
        self.sys_log.info('balance_available : {:.2f}'.format(self.balance_available))

        # reject balance insufficient period.
            # balance_min
        if self.balance_available < balance_min:
            self.sys_log.warning('balance_available {:.2f} < balance_min {:.2f}\n'.format(self.balance_available, balance_min))
            self.balance_insufficient = 1

            # balance < asset 
        if self.balance_available < self.config.trader_set.initial_asset:
            self.sys_log.warning('balance_available {:.2f} < initial_asset {:.2f}\n'.format(self.balance_available, self.config.trader_set.initial_asset))
            self.balance_insufficient = 1

        else:
            self.balance_available = self.config.trader_set.initial_asset # initial_asset is requested asset for one trade.
            


def get_precision(self, 
                  symbol):

    """
    v2.0
        modify to vivid mode.

    last confirmed at, 20240614 1403.
    """
    
    try:
        response = self.exchange_info()
    except Exception as e:
        msg = "error in get_precision : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)

        # time.sleep(self.config.trader_set.api_term)
    else:
        precision_price, precision_quantity = [[data['pricePrecision'], data['quantityPrecision']] 
                                                         for data in response['symbols'] if data['symbol'] == symbol][0]

        return precision_price, precision_quantity
        


def get_price(self, 
              side_open, 
              df_res):

    """
    v1.0
        df_res_open replaced with df_res.
        modify OrderSide.BUY to 'BUY'
    v1.1
        modify to vivid mode.

    last confirmed at, 20240614 1345.
    """
    
    if side_open == 'BUY':
        price_entry = df_res['long_ep1_{}'.format(self.config.selection_id)].to_numpy()[self.config.trader_set.complete_index]
        price_stop_loss = df_res['long_out_{}'.format(self.config.selection_id)].to_numpy()[self.config.trader_set.complete_index]
        price_take_profit = df_res['long_tp_{}'.format(self.config.selection_id)].to_numpy()[self.config.trader_set.complete_index]
        
    else:
        price_entry = df_res['short_ep1_{}'.format(self.config.selection_id)].to_numpy()[self.config.trader_set.complete_index]
        price_stop_loss = df_res['short_out_{}'.format(self.config.selection_id)].to_numpy()[self.config.trader_set.complete_index]
        price_take_profit = df_res['short_tp_{}'.format(self.config.selection_id)].to_numpy()[self.config.trader_set.complete_index]        

    return price_entry, price_stop_loss, price_take_profit

  


def get_leverage_limit(UMFutures, symbol, price_entry, price_stop_loss, fee_entry, fee_exit):

    """
    v2.0
        divide into server & user
            compare which one is minimal.
    v3.0
        modify to vivid input & output.

    last confirmed at, 20240613 1329.
    """    

    # leverage_limit (server)
    server_time = UMFutures.time()['serverTime']
    response = UMFutures.leverage_brackets(symbol=symbol, recvWindow=6000, timestamp=server_time)
    leverage_limit_server = response[0]['brackets'][0]['initialLeverage']    
    
    loss = abs(price_entry - price_stop_loss) + (price_entry * fee_entry + price_stop_loss * fee_exit)
    leverage_limit_user = max(int(price_entry / loss), 1)

    leverage_limit = min(leverage_limit_user, leverage_limit_server)

    return loss, leverage_limit_user, leverage_limit_server, leverage_limit
    


def get_leverage(self, mode_round='floor'):

    """
    v2.0 --> v2_1
        math.ceil self.leverage.
    v2_1 --> v2_2
        target_pct pointed at self.price_take_profit. (origin --> self.price_stop_loss)
        adj. get_pr_v8
    v3.0
        Class mode
            add self,
                temporarily not removing origin parameter.
                
            replace balance_available_ with self.price_stop_loss.
    v3.1
        calibrate target_pct.
        add checking pd.isnull(self.leverage) for messenger mode.

    last confirmed at, 20240528 1539.
    """
    
    # init.
    price_fluc = None
    fee_sum = self.config.trader_set.fee_market + self.config.trader_set.fee_market # conservative.
    if pd.isnull(self.leverage):
        self.leverage = self.config.lvrg_set.leverage
    
    # calculate leverage.
        # target_pct is based on price_take_profit.
    if not pd.isnull(self.price_take_profit):
        if not self.config.lvrg_set.static_lvrg_short:
            if self.side_open == 'SELL':
                price_fluc = self.price_entry / (self.price_take_profit - (self.price_take_profit + self.price_entry) * fee_sum)

        if not self.config.lvrg_set.static_lvrg_long:
            if self.side_open == 'BUY':
                price_fluc = (self.price_take_profit - (self.price_take_profit + self.price_entry) * fee_sum) / self.price_entry

        if price_fluc is not None:
            self.leverage = self.config.lvrg_set.target_pct / abs(price_fluc - 1)

    # make as an integer.
    if not self.config.lvrg_set.allow_float:
        if mode_round == 'floor':            
            self.leverage = math.floor(self.leverage)
        else:
            self.leverage = math.ceil(self.leverage)
    
    # leverage rejection.
        # if calculated leverage is smaller than 1, consider rejection.
    if self.config.lvrg_set.lvrg_rejection:
        if self.leverage < 1:
            self.leverage = None

    # limit leverage
    self.leverage = min(self.leverage_limit, max(self.leverage, 1))
        


def set_leverage(self,):

    try:
        server_time = self.time()['serverTime']
        self.change_leverage(symbol=self.symbol, leverage=self.leverage, recvWindow=6000, timestamp=server_time)
    except Exception as e:
        msg = "error in change_initial_leverage : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
    else:
        self.sys_log.info('self.leverage changed to {}'.format(self.leverage))




def set_position_mode(self, ):

    """
    v1.0
        pass error -4059 -4046

    last confirmed at, 20240504 1948.
    """

    try:
        server_time = self.time()['serverTime']
        self.change_position_mode(dualSidePosition="true", recvWindow=2000, timestamp=server_time)
    except Exception as e:
        if '-4059' in str(e): # 'No need to change position side.'
            return
        msg = "error in set_position_mode : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
    else:
        self.sys_log.info("dualSidePosition is true.")


def set_margin_type(self, margin_type='CROSSED'):

    """
    v1.0
        pass error -4046

    last confirmed at, 20240504 1948.
    """

    # margin type => "cross or isolated"
    try:
        server_time = self.time()['serverTime']
        if margin_type == 'CROSSED':
            self.change_margin_type(symbol=self.symbol, marginType=FuturesMarginType.CROSSED, recvWindow=6000, timestamp=server_time)
        else:
            self.change_margin_type(symbol=self.symbol, marginType=FuturesMarginType.ISOLATED, recvWindow=6000, timestamp=server_time)
            
    except Exception as e:
        if '-4046' in str(e): # 'No need to change margin type.'
            return
        msg = "error in set_margin_type : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
    else:
        self.sys_log.info("margin type is {} now.".format(margin_type))



def get_order_info(self, ):

    """
    v1.0 
        order_res format.
            {'orderId': 12877344699,
              'symbol': 'THETAUSDT',
              'status': 'NEW',
              'clientOrderId': 't1eOIqWG2m72oxaMKLZHKE',
              'price': '1.9500',
              'avgPrice': '0.00',
              'origQty': '10.0',
              'executedQty': '0.0',
              'cumQty': '0.0',
              'cumQuote': '0.00000',
              'timeInForce': 'GTC',
              'type': 'LIMIT',
              'reduceOnly': False,
              'closePosition': False,
              'side': 'BUY',
              'positionSide': 'LONG',
              'stopPrice': '0.0000',
              'workingType': 'CONTRACT_PRICE',
              'priceProtect': False,
              'origType': 'LIMIT',
              'priceMatch': 'NONE',
              'selfTradePreventionMode': 'NONE',
              'goodTillDate': 0,
              'updateTime': 1713354764631},

    last confirmed at, 20240417 2100.
    """
    
    try:
        server_time = self.time()['serverTime']
        self.order_info = self.query_order(symbol=self.symbol, orderId=self.orderId, recvWindow=2000, timestamp=server_time)
        # return self.order_info
    except Exception as e:
        msg = "error in get_order_info : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
    else:
        self.sys_log.info("order_info : {}".format(self.order_info))
        

def get_price_realtime(self, ):
    
    """
    v3.0
        get price_market by self.price_market
        add non-error solution.
    v3.1
        remove loop & agg_trade
    v3.2
        restore agg_trade
    
    last confirmed at, 20240517 1351.
    """
    
    try:
        self.price_realtime =  self.price_market[self.symbol]
    except Exception as e:
        self.price_realtime = np.nan   
        
        if self.symbol not in self.price_market.keys():
            self.websocket_client.agg_trade(symbol=self.symbol, id=1, callback=self.agg_trade_message_handler)
            msg = "error in get_price_realtime : {} added to websocket_client.agg_trade".format(self.symbol)
        else:                
            msg = "error in get_price_realtime : {}".format(e)
        self.sys_log.error(msg)


def get_price_expiration(self, ):

    """
    v2.0 --> v3.0
        self.expired variable added.
    v4
        Class mode.
            rename function to get_price_expiration.

    last confirmed at, 20240428 1904.
    """

    # if self.config.tr_set.expire_tick != "None":
    #     if time.time() - datetime.timestamp(self.df_res.index[self.config.trader_set.latest_index]) >= self.config.tr_set.expire_tick * 60:
    #         self.expired = 1    
        
    if self.side_open == OrderSide.SELL:
        short_tp_ = self.df_res['short_tp_{}'.format(self.config.selection_id)].to_numpy()
        short_tp_gap_ = self.df_res['short_tp_gap_{}'.format(self.config.selection_id)].to_numpy()        
        self.price_expiration = short_tp_[self.config.trader_set.complete_index] + short_tp_gap_[self.config.trader_set.complete_index] * self.config.tr_set.expire_k1    
    else:
        long_tp_ = self.df_res['long_tp_{}'.format(self.config.selection_id)].to_numpy()  # iloc 5.34 ms, to_numpy() 3.94 ms
        long_tp_gap_ = self.df_res['long_tp_gap_{}'.format(self.config.selection_id)].to_numpy()        
        self.price_expiration = long_tp_[self.config.trader_set.complete_index] - long_tp_gap_[self.config.trader_set.complete_index] * self.config.tr_set.expire_k1


def get_price_liquidation(side_open, price_entry, fee_limit, fee_market, leverage):

    """
    v2.0
        consider vivide input & output.

    last confirmed at, 20240613 1223. 
    """
    
    if side_open == 'SELL':
        price_liquidation = price_entry / (1 + fee_limit + fee_market - 1 / leverage)
    else:
        price_liquidation = price_entry * (1 + fee_limit + fee_market - 1 / leverage)

    return price_liquidation



def check_order_expiration_onbarclose(self,):  # for point2

    """
    v4.0
        Class mode
    
    last confirmed at, 20240418 2459.
    """

    selection_id = self.config.selection_id

    if self.config.tr_set.expire_tick != "None":
        if datetime.timestamp(self.df_res.index[-1]) - datetime.timestamp(self.df_res.index[self.config.trader_set.latest_index]) \
                >= self.config.tr_set.expire_tick * 60:
            self.expired = 1

    if self.config.tr_set.expire_k1 != "None":  # Todo - onbarclose 에서는, self.df_res 으로 open_index 의 self.price_take_profit 정보를 사용
        if self.side_open == OrderSide.SELL:
            low = self.df_res['low'].to_numpy()
            short_tp_ = self.df_res['short_tp_{}'.format(selection_id)].to_numpy()  # id 에 따라 dynamic 변수라 이곳에서 numpy 화 진행
            short_tp_gap_ = self.df_res['short_tp_gap_{}'.format(selection_id)].to_numpy()
            if low[self.config.trader_set.complete_index] <= short_tp_[self.config.trader_set.complete_index] + short_tp_gap_[self.config.trader_set.complete_index] * self.config.tr_set.expire_k1:
                self.expired = 1
        else:
            high = self.df_res['high'].to_numpy()
            long_tp_ = self.df_res['long_tp_{}'.format(selection_id)].to_numpy()  # iloc 이 빠를까, to_numpy() 가 빠를까  # 3.94 ms --> 5.34 ms (iloc)
            long_tp_gap_ = self.df_res['long_tp_gap_{}'.format(selection_id)].to_numpy()
            if high[self.config.trader_set.complete_index] >= long_tp_[self.config.trader_set.complete_index] - long_tp_gap_[self.config.trader_set.complete_index] * self.config.tr_set.expire_k1:
                self.expired = 1
                



def check_stop_loss(self, ):

    """
    v2 --> v3.0
        1. get self.price_realtime from outer_scope
        2. inversion 고려 아직임. (inversion 사용하게 되면 고려할 것.) (Todo)
        3. add self.log_out
        4. add non_out.
    v4.0
        turn off self.order_market_on = None
        
    last confirmed at, 20240428 2135.
    """

    # init.
    # self.order_market_on = None
    self.log_out = None

    # check liquidation
    if self.side_open == OrderSide.SELL:
        const_str = "self.price_realtime >= self.price_liquidation"
        if eval(const_str):
            self.order_market_on = True
            self.log_out = self.price_liquidation
            self.sys_log.info("{} : {} {}".format(const_str, self.price_realtime, self.price_liquidation))
    else:
        const_str = "self.price_realtime <= self.price_liquidation"
        if eval(const_str):
            self.order_market_on = True
            self.log_out = self.price_liquidation
            self.sys_log.info("{} : {} {}".format(const_str, self.price_realtime, self.price_liquidation))

    if not self.config.out_set.non_out:

        # hl_out
        if self.config.out_set.hl_out:
            if self.side_open == OrderSide.SELL:
                const_str = "self.price_realtime >= self.price_stop_loss"
                if eval(const_str):
                    self.order_market_on = True
                    self.log_out = self.price_stop_loss
                    self.sys_log.info("{} : {} {}".format(const_str, self.price_realtime, self.price_stop_loss))
            else:
                const_str = "self.price_realtime <= self.price_stop_loss"
                if eval(const_str):
                    self.order_market_on = True
                    self.log_out = self.price_stop_loss
                    self.sys_log.info("{} : {} {}".format(const_str, self.price_realtime, self.price_stop_loss))

        # close_out
        else:
            close = self.df_res['close'].to_numpy()

            if self.side_open == OrderSide.SELL:
                if close[self.config.trader_set.complete_index] >= self.price_stop_loss:
                    self.order_market_on = True
                    self.log_out = close[self.config.trader_set.complete_index]
                    self.sys_log.info("{} : {} {}".format("close >= self.price_stop_loss", self.log_out, self.price_stop_loss))
            else:
                if close[self.config.trader_set.complete_index] <= self.price_stop_loss:
                    self.order_market_on = True
                    self.log_out = close[self.config.trader_set.complete_index]
                    self.sys_log.info("{} : {} {}".format("close <= self.price_stop_loss", self.log_out, self.price_stop_loss))
                    



def check_stop_loss_onbarclose(self,):

    """
    v1 --> v2.0
        1. add liquidation platform.
        2. add self.log_out
        3. add non_out
    """

    self.log_out = None

    high = self.df_res['high'].to_numpy()[self.config.trader_set.complete_index]
    low = self.df_res['low'].to_numpy()[self.config.trader_set.complete_index]
    close = self.df_res['close'].to_numpy()[self.config.trader_set.complete_index]

    # ------ 1. liquidation default check ------ #
    if self.side_open == OrderSide.SELL:
        const_str = "high >= self.price_liquidation"
        if eval(const_str):
            self.order_market_on = True
            self.log_out = self.price_liquidation
            self.sys_log.info("{} : {} {}".format(const_str, high, self.price_liquidation))
    else:
        const_str = "low <= self.price_liquidation"
        if eval(const_str):
            self.order_market_on = True
            self.log_out = self.price_liquidation
            self.sys_log.info("{} : {} {}".format(const_str, low, self.price_liquidation))

    if not self.config.out_set.non_out:

        # ------ 2. hl_out ------ #
        if self.config.out_set.hl_out:
            if self.side_open == OrderSide.SELL:
                const_str = "high >= self.price_stop_loss"
                if eval(const_str):
                    self.order_market_on = True
                    self.log_out = self.price_stop_loss
                    self.sys_log.info("{} : {} {}".format(const_str, high, self.price_stop_loss))
            else:
                const_str = "low <= self.price_stop_loss"
                if eval(const_str):
                    self.order_market_on = True
                    self.log_out = self.price_stop_loss
                    self.sys_log.info("{} : {} {}".format(const_str, low, self.price_stop_loss))

        # ------ 3. close_out ------ #
        else:
            if self.side_open == OrderSide.SELL:
                if close >= self.price_stop_loss:
                    self.order_market_on = True
                    self.log_out = close
                    self.sys_log.info("{} : {} {}".format("close >= self.price_stop_loss", self.log_out, self.price_stop_loss))
            else:
                if close <= self.price_stop_loss:
                    self.order_market_on = True
                    self.log_out = close
                    self.sys_log.info("{} : {} {}".format("close <= self.price_stop_loss", self.log_out, self.price_stop_loss))



def get_price_replacement(self, order_info_old, idx):

    """
    v2.0
        modify to vivid in-out

    last confirmed at, 20240617 1507.
    """

    self.price_replacement = 0

    # price_TP / SL replacement version.
    # self.df_res = pd.read_feather(order_info_old.path_df_res)
    self.df_res = pd.read_feather("{}\\{}.ftr".format(self.path_dir_df_res, self.symbol)) # temporary.
    
    # temporarily, modify side_open for getting updated stop_loss.
    self.side_open = order_info_old.side_close
    # self.get_price(self)
    
    self.price_entry, \
    self.price_stop_loss, \
    self.price_take_profit = self.get_price(self, 
                                          self.side_open, 
                                          self.df_res)
    self.table_trade.at[idx, 'price_stop_loss'] = self.price_take_profit
    
    if self.side_open == 'BUY':        
        if self.price_stop_loss < order_info_old.price_stop_loss:
            self.price_replacement = 1
    else:
        if self.price_stop_loss > order_info_old.price_stop_loss: 
            self.price_replacement = 1

    # return it.
    self.side_open = order_info_old.side_open
    # self.get_price(self)
    
    self.price_entry, \
    self.price_stop_loss, \
    self.price_take_profit = self.get_price(self, 
                                          self.side_open, 
                                          self.df_res)
    self.table_trade.at[idx, 'price_take_profit'] = self.price_take_profit




def get_quantity_unexecuted(self, ):

    """
    v2.0
        apply updated get_precision

    last confirmed at, 20240617 0728.
    """

    # init.
    quantity_unexecuted = np.nan
    
    self.order_cancel(self,) # input : symbol, orderId
    
    self.get_order_info(self,) # input : symbol, orderId
    quantity_unexecuted = abs(float(self.order_info['origQty'])) - abs(float(self.order_info['executedQty']))
    
        # get price, volume updated precision
    self.precision_price, \
    self.precision_quantity = self.get_precision(self, 
                                                self.symbol)
    self.sys_log.info('precision_quantity : {}'.format(self.precision_quantity))
    
    self.quantity_unexecuted = self.calc_with_precision(quantity_unexecuted, self.precision_quantity)
    self.sys_log.info('quantity_unexecuted : {}'.format(quantity_unexecuted))
    self.sys_log.info('quantity_unexecuted (adj. precision) : {}'.format(self.quantity_unexecuted))
    

def check_stop_loss_by_signal(self, ):

    """
    v3 --> v4.0
        1. add fisher_exit.
        2. add self.np_timeidx
    """

    self.log_out = None

    close = self.df_res['close'].to_numpy()[self.config.trader_set.complete_index]

    # 1. timestamp
    # if self.config.out_set.tf_exit != "None":
    #     if self.np_timeidx[i] % self.config.out_set.tf_exit == self.config.out_set.tf_exit - 1 and i != open_i:
    #         self.order_market_on = True

    # 2. fisher
    # if self.config.out_set.fisher_exit:

    #     itv_num = itv_to_number(self.table_condition_row.tf_entry)

    #     if self.np_timeidx[self.config.trader_set.complete_index] % itv_num == itv_num - 1:

    #         fisher_ = self.df_res['fisher_{}30'.format(self.table_condition_row.tf_entry)].to_numpy()
    #         fisher_band = self.config.out_set.fisher_band
    #         fisher_band2 = self.config.out_set.fisher_band2

    #         if self.side_open == OrderSide.SELL:
    #             if (fisher_[self.config.trader_set.complete_index - itv_num] > -fisher_band) & (fisher_[self.config.trader_set.complete_index] <= -fisher_band):
    #                 self.order_market_on = True
    #             elif (fisher_[self.config.trader_set.complete_index - itv_num] < fisher_band2) & (fisher_[self.config.trader_set.complete_index] >= fisher_band2):
    #                 self.order_market_on = True
    #         else:
    #             if (fisher_[self.config.trader_set.complete_index - itv_num] < fisher_band) & (fisher_[self.config.trader_set.complete_index] >= fisher_band):
    #                 self.order_market_on = True
    #             elif (fisher_[self.config.trader_set.complete_index - itv_num] > fisher_band2) & (fisher_[self.config.trader_set.complete_index] <= fisher_band2):
    #                 self.order_market_on = True

    # 3. rsi_exit
    if self.config.out_set.rsi_exit:
        rsi_ = self.df_res['rsi_%s' % self.config.loc_set.point.exp_itv].to_numpy()
        osc_band = self.config.loc_set.point.osc_band

        if self.side_open == OrderSide.SELL:
            if (rsi_[self.config.trader_set.complete_index - 1] >= 50 - osc_band) & (rsi_[self.config.trader_set.complete_index] < 50 - osc_band):
                self.order_market_on = True
        else:
            if (rsi_[self.config.trader_set.complete_index - 1] <= 50 + osc_band) & (rsi_[self.config.trader_set.complete_index] > 50 + osc_band):
                self.order_market_on = True

    # 4. cci_exit
    #           a. deprecated.
    # if self.config.out_set.cci_exit:
    #     wave_itv1 = self.config.tr_set.wave_itv1
    #     wave_period1 = self.config.tr_set.wave_period1
    #
    #     if self.side_open == OrderSide.SELL:
    #         wave_co_ = self.df_res['wave_co_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[self.config.trader_set.complete_index]
    #         if wave_co_:
    #             self.order_market_on = True
    #     else:
    #         wave_cu_ = self.df_res['wave_cu_{}{}'.format(wave_itv1, wave_period1)].to_numpy()[self.config.trader_set.complete_index]
    #         if wave_cu_:
    #             self.order_market_on = True

    # if self.order_market_on:
    #     self.log_out = close
    #     self.sys_log.info("signal self.price_stop_loss : {}".format(self.log_out))

    # return self.order_market_on, self.log_out





def order_limit(self, order_type, side_order, side_position, price, quantity):

    """
    v2.0
        1. self.symbol 도입.
        2. rename to retry_count & -1111 error update.
    v3.0
        Class mode
            remove order_data
            
    last confirmed at, 20240428 2042.
    """
    
    # init.
    retry_count = 0    
    self.error_code = 0
    self.order_res = None

    quantity_orig = quantity
    

    while 1:
        try:
            # 0. new_order (limit & market)
            # Todo, quantity 에 float 대신 str 을 입력하는게 정확한 것으로 앎.
            #   1. precision 보호 차원인가.
            server_time = self.time()['serverTime']
            if order_type == OrderType.MARKET:
                # a. market order
                self.order_res = self.new_order(symbol=self.symbol,
                                                side=side_order,
                                                positionSide=side_position,
                                                type=OrderType.MARKET,
                                                quantity=str(quantity),
                                                timestamp=server_time)
            else:
                # b. limit order
                self.order_res = self.new_order(timeInForce=TimeInForce.GTC,
                                                symbol=self.symbol,
                                                side=side_order,
                                                positionSide=side_position,
                                                type=OrderType.LIMIT,
                                                quantity=str(quantity),
                                                price=str(price),
                                                timestamp=server_time)
        except Exception as e:
            msg = "error in order_limit : {}".format(e)
            self.sys_log.error(msg)
            # self.push_msg(self, msg)

            # 1. order_limit() 에서 해결할 수 없는 error 일 경우, return
            #       a. -4003 : quantity less than zero
            # if "-4003" in str(e):
            #     self.error_code = -4003
            self.error_code = -4003

            return

            # retry_count += 1
            # # 2. order_limit() 에서 해결할 error 일 경우, continue
            # #       y. -1111 : Precision is over the maximum defined for this asset = quantity precision error
            # if '-1111' in str(e):

            #     try:
            #         #   _, precision_quantity = self.get_precision()
            #         precision_quantity = retry_count

            #         # close_remain_quantity = get_remaining_quantity(self.symbol)
            #         #   error 발생한 경우 정상 체결된 quantity 는 존재하지 않는다고 가정 - close_remain_quantity 에는 변함이 없을거라 봄
            #         quantity = self.calc_with_precision(quantity_orig, precision_quantity, def_type='floor')
            #         self.sys_log.info("modified quantity & precision : {} {}".format(quantity, precision_quantity))

            #     except Exception as e:
            #         msg = "error in get modified precision_quantity : {}".format(e)
            #         self.sys_log.error(msg)
            #         self.push_msg(self, msg)

            #     if retry_count >= 10:
            #         retry_count = 0

            #     # self.error_code = -1111
            #     # return None, self.balance_over, self.error_code
            #     time.sleep(self.config.trader_set.order_term)
            #     continue

            # #       a. -4014 : Price not increased by tick size
            # if '-4014' in str(e):
            #     try:
            #         realtime_price = self.get_price_realtime(self)
            #         self.precision_price = self.get_precision_by_price(realtime_price)
            #         realtime_price = self.calc_with_precision(realtime_price, self.precision_price)
            #         self.sys_log.info("modified realtime_price & precision : {}, {}".format(realtime_price, self.precision_price))

            #     except Exception as e:
            #         msg = "error in get_price_realtime (order_open phase): {}".format(e)
            #         self.sys_log.error(msg)
            #         self.push_msg(self, msg)

            #     if retry_count >= 10:
            #         self.error_code = 'unknown'

            #     time.sleep(self.config.trader_set.order_term)
            #     continue

            # #       b. -2019 : Margin is insufficient
            # # Todo - for tp_exec_qty miscalc., self.overbalance 수정되었으면, return 으로 받아야할 것 => 무슨 소리야..?
            # if '-2019' in str(e):
            #     # if order_data is not None:  # None 인 경우, self.leverage 가 정의되지 않음.
            #     # i. 예기치 못한 오류로 인해 over balance 상태가 되었을때의 조치
            #     try:
            #         # 1. get available quantity
            #         balance_available = self.get_balance_available() * 0.9

            #         _, precision_quantity = self.get_precision(self)
            #         quantity = balance_available / price * self.leverage
            #         quantity = self.calc_with_precision(quantity, precision_quantity, def_type='floor')
            #         self.sys_log.info('balance_available (temp) : {}'.format(balance_available))
            #         self.sys_log.info('quantity : {}'.format(quantity))

            #     except Exception as e:
            #         msg = "error in get_availableBalance (-2019 phase) : {}".format(e)
            #         self.sys_log.error(msg)
            #         self.push_msg(self, msg)

            #     # ii. balance 가 근본적으로 부족한 경우, 주문을 허용하지 않는다.
            #     retry_count += 1
            #     if retry_count > 5:
            #         msg = "retry_count over. : {}".format(retry_count)
            #         self.sys_log.error(msg)
            #         self.push_msg(self, msg)
            #         self.error_code = -2019
            #     else:
            #         time.sleep(self.config.trader_set.order_term)
            #         continue
            # else:
            #     msg = "unknown error : {}".format(e)
            #     self.sys_log.error(msg)
            #     self.push_msg(self, msg)
            #     self.error_code = 'unknown'

        else:
            # 3. 정상 order 의 self.error_code = 0.
            # self.order_res = order_res
            self.sys_log.info("order_limit succeed. order_res : {}".format(self.order_res))
            return







def order_cancel(self, ):

    """
    v1.0
        add table_order logic.

    last confirmed at, 20240423 2413.
    """
    
    try:
        server_time = self.time()['serverTime']
        _ = self.cancel_order(symbol=self.symbol, orderId=self.orderId, timestamp=server_time)
    except Exception as e:        
        msg = "error in order_cancel : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(self, msg)
    else:
        self.sys_log.info("{} {} canceled.".format(self.symbol, self.orderId))


def order_market(self, ):

    """
    v2.0
        update retry_count        
    v3.0
        remove order_res_list.
        remove error solution phase.   
    v3.1
        replace get_quantity_unexecuted()
        
    last confirmed at, 20240518 2212.
    """

    while 1:

        self.get_quantity_unexecuted(self,)

        # order_market
        retry_count = 0
        self.error_code = 0
        quantity_unexecuted_org = self.quantity_unexecuted
        while 1:
            try:
                server_time = self.time()['serverTime']
                self.order_res = self.new_order(symbol=self.symbol,
                                                side=self.side_close,
                                                positionSide=self.side_position,
                                                type=OrderType.MARKET,
                                                quantity=str(self.quantity_unexecuted),
                                                timestamp=server_time)
            except Exception as e:

                msg = "error in order_market : {}".format(e)
                self.sys_log.error(msg)
                self.push_msg(self, msg)
                continue

                # # -2022 ReduceOnly Order is rejected
                # if '-2022' in str(e):
                #     self.error_code = '-2022'
                #     break

                # # -4003 quantity less than zero
                # if '-4003' in str(e):
                #     self.error_code = '-4003'
                #     break
                    
                # # -1111 : Precision is over the maximum defined for this asset
                #     # = quantity precision error
                #     # 상당히 오래전에 만든 대응법인데, 이게 맞는 걸까.. (Todo)
                # if '-1111' in str(e):
                #     try:
                #         # _, precision_quantity = self.get_precision()
                #         precision_quantity = retry_count

                #         # self.quantity_unexecuted = get_remaining_quantity(self.symbol)
                #         #   error 발생한 경우 정상 체결된 quantity 는 존재하지 않는다고 가정 - self.quantity_unexecuted 에는 변함이 없을거라 봄
                #         self.quantity_unexecuted = self.calc_with_precision(quantity_unexecuted_org, precision_quantity, def_type='floor')
                #         self.sys_log.info("modified quantity & precision : {} {}".format(self.quantity_unexecuted, precision_quantity))

                #     except Exception as e:
                #         msg = "error in get modified precision_quantity : {}".format(e)
                #         self.sys_log.error(msg)
                #         self.push_msg(self, msg)

                #     retry_count += 1
                #     if retry_count >= 10:
                #         retry_count = 0

                # # self.config.out_set.out_type = OrderType.MARKET  # just 재시도
                # time.sleep(self.config.trader_set.order_term)

            else:
                self.sys_log.info("order_market succeed. : {}".format(self.order_res))
                break

        # 4. term for quantity consumption.
        time.sleep(1)

        # 5.  order_res 가 정의되는 경우 / 그렇지 않은 경우.
        # if self.error_code in ['-2022', '-4003']:
        #     self.sys_log.info("order_market executed.")
        #     return
        # else:
            # price_close_executed_list, quantity_close_executed_list, _ = self.order_cancel_multiple([self.order_res], order_type=OrderType.MARKET) # this line uses OrderType.MARKET!
            # self.get_order_info(self) # order_cancel_multiple contains this func.

        self.symbol = self.order_res['symbol'] # we have symbol column in table already.
        self.orderId = self.order_res['orderId']
        self.get_order_info(self,)
        
        if self.order_info['status'] == 'FILLED':
            self.sys_log.info("order_market filled.")
            return 
        else:
            self.sys_log.info("order_market failed.")
            continue
            

def get_income_info(self, 
                    table_log,
                    code,
                    side_position,
                    leverage,
                    income_accumulated,
                    profit_accumulated,
                    mode="PROD", 
                    currency="USDT"):  # 복수 pos 를 허용하기 위한 api 미사용

    """
    v2.0
        Class mode
            get income from table_log.
        add reject manual trade intervention.
    v3.0
        select cumQuote by order_way.
        modify using last row : PARTIALLY_FILLED & FILLED exist both, use FILLED only.
    v4.0
        vivid input / output

    last confimred at, 20240617 0725.    
    """

    table_log = table_log.astype({'cumQuote' : 'float'})
    table_log_valid = table_log[(table_log.code == code) & (table_log.cumQuote != 0)]
    table_log_valid['fee_ratio'] = np.where(table_log_valid.type == 'LIMIT', self.config.trader_set.fee_limit, self.config.trader_set.fee_market)
    table_log_valid['fee'] = table_log_valid.cumQuote * table_log_valid.fee_ratio
    
    # if PARTIALLY_FILLED & FILLED exist both, use FILLED only.
    table_log_valid_open = table_log_valid[table_log_valid.order_way == 'OPEN']
    table_log_valid_close = table_log_valid[table_log_valid.order_way == 'CLOSE']
    
    if len(table_log_valid_open) > 0 and len(table_log_valid_close) > 0:
        cumQuote_open = table_log_valid_open.iloc[-1].cumQuote
        cumQuote_close = table_log_valid_close.iloc[-1].cumQuote
        fee_open = table_log_valid_open.iloc[-1].fee
        fee_close = table_log_valid_close.iloc[-1].fee
    else:
        self.sys_log.info("table_log_valid length insufficient : {}".format(table_log_valid))
        return 0, income_accumulated, 0, profit_accumulated
        
    self.sys_log.info("cumQuote_open : {}".format(cumQuote_open))
    self.sys_log.info("cumQuote_close : {}".format(cumQuote_close))
    self.sys_log.info("fee_open : {}".format(fee_open))
    self.sys_log.info("fee_close : {}".format(fee_close))

    
    if side_position == 'LONG':
        income = cumQuote_close - cumQuote_open
    else:
        income = cumQuote_open - cumQuote_close
    income -= (fee_open + fee_close)
        
    income_accumulated += income
    self.sys_log.info("income : {:.4f} {}".format(income, currency))
    self.sys_log.info("income_accumulated : {:.4f} {}".format(income_accumulated, currency))

    
    # reject manual trade intervention.
    if cumQuote_open == 0:
        profit = 0.0
    else:
        profit = income / cumQuote_open * leverage
    profit_accumulated += profit
    self.sys_log.info("profit : {:.4f}".format(profit))


    # set to outer scope using output vars. later.
    # if self.config.trader_set.profit_mode == "PROD":
    #     # add your income.
    #     balance_available += income
        
    #     # update config.
    #     self.config.trader_set.initial_asset = balance_available

    #     with open(self.path_config, 'w') as cfg:
    #         json.dump(self.config, cfg, indent=2)   
    
            
    msg = ("cumQuote_open : {:.4f}\n" + 
            "cumQuote_close : {:.4f}\n" + 
            "fee_open : {:.4f}\n" + 
            "fee_close : {:.4f}\n" + 
            "income : {:.4f}\n" + 
            "income_accumulated : {:.4f}\n" + 
            "profit : {:.4f}\n" + 
            "profit_accumulated : {:.4f}\n")\
            .format(cumQuote_open,
                    cumQuote_close,
                    fee_open,
                    fee_close,
                    income,
                    income_accumulated,
                    profit,
                    profit_accumulated)

    self.push_msg(self, msg)

    return income, income_accumulated, profit, profit_accumulated



    








        