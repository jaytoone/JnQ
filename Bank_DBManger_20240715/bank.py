
# Library for Futures.
from funcs.binance_f.model import *
from funcs.binance.um_futures import UMFutures
from funcs.binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
# from funcs.binance.error import ClientError


# Library for Bank.
# from funcs.binance.shield_module_v4_1 import ShieldModule
from funcs.public.constant import *
from funcs.public.indicator import *
from funcs.public.indicator_ import *
from funcs.public.broker import *



import pandas as pd
import numpy as np
import math
import threading

import time
from datetime import datetime


import hashlib
from psycopg2 import sql
from psycopg2.pool import SimpleConnectionPool

import pickle
import logging

import logging.config
from easydict import EasyDict
import json
from ast import literal_eval

import telegram
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters

from IPython.display import clear_output

pd.set_option('mode.chained_assignment',  None)
pd.set_option('display.max_columns', 100)


class DatabaseManager:
    """
    v0.1
        init.
    
    last confirmed at, 20240715 2306.
    """
    def __init__(self, dbname, user, password, host, port):
        self.pool = SimpleConnectionPool(1,   # min thread
                                         10,  # max thread
                                         dbname=dbname, 
                                         user=user, 
                                         password=password,
                                         host=host, 
                                         port=port)
        self.file_hashes = {}

    
    def calculate_file_hash(self, file_path):
        """Calculate the hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def fetch_table(self, table_name, limit=None):
    
        """
        v1.1
            using engine, much faster.
        v1.2
            replace engine with psycopg2.
    
        last confirmed at, 20240713 2343.
        """
        
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                query = f"SELECT * FROM {table_name}"
                if limit:
                    query += f" LIMIT {limit}"
                cur.execute(query)
                result = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return pd.DataFrame(result, columns=columns)
        except Exception as e:
            self.sys_log.error(f"error fetching data from {table_name}: {e}")
            return pd.DataFrame()
        finally:
            self.pool.putconn(conn)

        
    def replace_table(self, df, table_name, send=False):
        
        """
        v1.1 using engine, much faster.
        v1.2 using temp_table, stay as consistent state.
        v1.2.1 modify to psycopg2, considering latency.
        v1.2.2 apply pool.
        v1.2.3 replace to truncate instead deletion.

        last confirmed at, 20240715 2010.
        """
        temp_csv = f'{table_name}.csv'
        df.to_csv(temp_csv, index=False, header=False)

        if send:
            # Check if the file has changed
            new_hash = self.calculate_file_hash(temp_csv)
            old_hash = self.file_hashes.get(temp_csv)

            if new_hash == old_hash:
                print("File has not changed. Skipping replace_table.")
                return

            # Update the hash
            self.file_hashes[temp_csv] = new_hash

            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    # Fetch column names and data types from the existing table
                    cur.execute(sql.SQL("""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position
                    """), [table_name])

                    # Retrieve the results
                    columns = cur.fetchall()
                    column_names = [col[0] for col in columns]

                    # Truncate the table
                    cur.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name)))

                    # Use COPY command to load data
                    with open(temp_csv, 'r') as f:
                        cur.copy_expert(sql.SQL("COPY {} ({}) FROM STDIN WITH CSV").format(
                            sql.Identifier(table_name),
                            sql.SQL(', ').join(map(sql.Identifier, column_names))
                        ), f)

                    # Commit changes
                    conn.commit()
            except Exception as e:
                if conn:
                    conn.rollback()
                print(f"Error occurred in replace_table: {e}")
            finally:
                # Release the connection back to the pool
                self.pool.putconn(conn)

    
    def insert_dataframe_to_schema_table(self, df, schema, table_name):
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                for i, row in df.iterrows():
                    try:
                        cur.execute(sql.SQL("""
                            INSERT INTO {}.{} (open, high, low, close, volume, symbol, datetime, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (symbol, datetime) DO NOTHING
                        """).format(sql.Identifier(schema), sql.Identifier(table_name)),
                        (row['open'], row['high'], row['low'], row['close'], row['volume'], row['symbol'], row['datetime'], row['timestamp']))
                    except Exception as e:
                        print(f"Error occurred while inserting row: {e}")
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error occurred: {e}")
        finally:
            self.pool.putconn(conn)


class TokenBucket:    
    
    """
    v0.1
        follow up server's api used-weight (tokens_used).

    last confirmed at, 20240714 2343.    
    """
    
    def __init__(self, sys_log, capacity):
        self.capacity = capacity  # Maximum number of tokens the bucket can hold
        self.tokens = capacity
        self.lock = threading.Lock()
        self.sys_log = sys_log
        # self.get_used_weight = get_used_weight # this func. require another api_wegiht...
        self.last_refill_time = time.time()
        
    def refill(self):
        now = time.time()
        # Calculate the number of minutes that have passed since the last refill
        minutes_passed = (now - self.last_refill_time) // 60
        if minutes_passed >= 1:
            self.tokens = self.capacity
            self.tokens_used = 0
            self.last_refill_time = now - (now % 60)  # Reset to the start of the current minute


    def consume(self, tokens_used, tokens):
        with self.lock: 
            self.tokens_used = tokens_used            
            self.refill()
            
            self.tokens = self.capacity - self.tokens_used
            self.sys_log.debug(f"tokens : {self.tokens}")
            # self.sys_log.debug(f"tokens_used : {self.tokens_used}")
            
            if self.tokens >= tokens: # enough to use.
                return True
            return False

    def wait_for_token_consume(self, tokens_used, tokens=10):
        
        wait_time = 1
        while not self.consume(tokens_used, tokens): # set maximium weight that can be consumed.
            time.sleep(wait_time)
            # wait_time = min(wait_time * 2, 60)  # Max wait time of 60 seconds # convert to static.


class Bank(UMFutures):

    """
    v0.1
        follow up server's api used-weight (tokens_used).
    v0.2
        using pool for sql.
    v0.3
        use DatabaseManager.

    last confirmed at, 20240714 2344.    
    """
    
    def __init__(self, **kwargs):
        
        api_key, secret_key = self.load_key(kwargs['path_api'])        
        UMFutures.__init__(self, key=api_key, secret=secret_key, show_header=True)

        self.websocket_client = UMFuturesWebsocketClient()
        self.websocket_client.start()
        self.price_market = {}
        
        
        self.path_save_log = kwargs['path_save_log']
        self.set_logger()
        
        self.path_config = kwargs['path_config']
        with open(self.path_config, 'r') as f:
            self.config = EasyDict(json.load(f))            
            

        api_rate_limit = kwargs['api_rate_limit']
        self.token_bucket = TokenBucket(sys_log=self.sys_log, capacity=api_rate_limit)

      
        # load_table 
        db_user = self.config.database.user
        db_password = self.config.database.password
        db_host = self.config.database.host
        db_port = self.config.database.port
        db_name = self.config.database.name     
        
        self.db_manager = DatabaseManager(dbname=db_name, 
                                         user=db_user, 
                                         password=db_password,
                                         host=db_host, 
                                         port=db_port)        

        self.table_account_name = kwargs['table_account_name']
        self.table_condition_name = kwargs['table_condition_name']
        self.table_trade_name = kwargs['table_trade_name']
        self.table_log_name = kwargs['table_log_name']
        
        self.table_account = self.db_manager.fetch_table(self.table_account_name)
        self.table_condition = self.db_manager.fetch_table(self.table_condition_name)
        self.table_trade = self.db_manager.fetch_table(self.table_trade_name)
        self.table_log = self.db_manager.fetch_table(self.table_log_name)

        
        self.path_dir_df_res = kwargs['path_dir_df_res']

        
        # add messegner
        self.chat_id = kwargs['chat_id']
        self.token = kwargs['token'] # "7156517117:AAF_Qsz3pPTHSHWY-ZKb8LSKP6RyHSMTupo"        
        self.msg_bot = None # default.
        self.get_messenger_bot()

        
        # income
        self.income = 0.0
        self.income_accumulated = self.config.trader_set.income_accumulated
        
        # profit
        self.profit = 0.0
        self.profit_accumulated = self.config.trader_set.profit_accumulated
        
        
        
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
        websocket streaming method 를 이용한 get_price_realtime method.
            try --> received data = None 일 경우를 대비한다.
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
    
        # file_handler = logging.FileHandler(self.path_save_log)
        file_handler = logging.handlers.RotatingFileHandler(self.path_save_log, maxBytes=10 * 1000 * 1000, backupCount=10)
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
        
        """
        v1.0
            this function has some time dely.
    
        last confirmed at, 20240517 1413.
        """        
        
        try:
            self.msg_bot.sendMessage(chat_id=self.chat_id, text=msg)
        except Exception as e:
            self.sys_log.error(e)

            
    def set_leverage(self,
                 symbol,
                 leverage):
        
        """
        v2.0
            vivid mode.
            
            v2.1
                add token.
            v2.2
                show_header=True.

        last confirmed at, 20240712 1041.
        """

        try:        
            server_time = self.time()['data']['serverTime']
            response = self.change_leverage(symbol=symbol, 
                                leverage=leverage, 
                                recvWindow=6000, 
                                timestamp=server_time)        
            
            tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
            self.token_bucket.wait_for_token_consume(tokens_used) 
            
        except Exception as e:
            msg = "error in change_initial_leverage : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)
        else:
            self.sys_log.info('leverage changed to {}'.format(leverage))

    def set_position_mode(self, 
                        dualSidePosition='true'):    
        
        """
        v1.0
            pass error -4059 -4046
        v2.0
            show_header=True. (token)

        last confirmed at, 20240712 1038.
        """

        try:
            server_time = self.time()['data']['serverTime']
            response = self.change_position_mode(dualSidePosition=dualSidePosition,
                                    recvWindow=2000,
                                    timestamp=server_time)
            
            tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
            self.token_bucket.wait_for_token_consume(tokens_used) 
            
        except Exception as e:
            if '-4059' in str(e): # 'No need to change position side.'
                return
            msg = "error in set_position_mode : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)
        else:
            self.sys_log.info("dualSidePosition is true.")
            
    def set_margin_type(self, 
                        symbol,
                        marginType='CROSSED'): # CROSSED / ISOLATED
        
        """
        v1.0
            pass error -4046
        v2.0
            vivid mode.
            
            v2.1
                show_header=True.

        last confirmed at, 20240712 1036.
        """

        # margin type => "cross or isolated"
        try:            
            server_time = self.time()['data']['serverTime']
            response = self.change_margin_type(symbol=symbol, 
                                    marginType=marginType, 
                                    recvWindow=6000, 
                                    timestamp=server_time)
            
            tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
            self.token_bucket.wait_for_token_consume(tokens_used) 
                
        except Exception as e:
            if '-4046' in str(e): # 'No need to change margin type.'
                return
            msg = "error in set_margin_type : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)
        else:
            self.sys_log.info("margin type is {} now.".format(marginType))
         
            
       
            
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




def get_df_new_by_streamer(self, ):
    
    try:
        self.df_res = next(self.streamer)  # Todo, 무결성 검증 미진행
    except Exception as e:
        msg = "error in get_df_new_by_streamer : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(msg)
        # self.kill_proc()  # error 처리할 필요없이, backtrader 프로그램 종료
        # return None, None   # output len 유지
    # else:
    #     return df_res


def get_df_new(self, interval='1m', days=2, end_date=None, limit=1500, timesleep=None):
        
    """
    v2.0
        add self.config.trader_set.get_df_new_timeout
        add df_res = None to all return phase.
    v3.0
        replace concat_candlestick to v2.0
    v3.1
        replace to bank.concat_candlestick
    v4.0
        integrate concat_candlestick logic.
        modify return if df_list.
        
        v4.1
            add token info.

    last confirmed at, 20240712 1024.
    """

    limit_kline = 1500
    assert limit <= limit_kline, f"assert limit < limit_kline ({limit_kline})"

    while True:
        if end_date is None:
            end_date = str(datetime.now()).split(' ')[0]

        timestamp_end = int(datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59).timestamp() * 1000)
        timestamp_start = timestamp_end - days * 24 * 60 * 60 * 1000
        timestamp_unit = itv_to_number(interval) * 60 * 1000 * limit

        time_arr_start = np.arange(timestamp_start, timestamp_end, timestamp_unit)
        time_arr_end = time_arr_start + timestamp_unit

        df_list = []

        for time_start, time_end in zip(time_arr_start, time_arr_end):
            try:                
                response = self.klines(
                    symbol=self.symbol,
                    interval=interval,
                    startTime=int(time_start),
                    endTime=int(time_end),
                    limit=limit
                )
                
                tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
                self.token_bucket.wait_for_token_consume(tokens_used) 

                response = response['data'] # replace response.

                if not response:
                    if df_list: # if df_list lose middle df, occur error.
                        return None, ''
                    time.sleep(timesleep)
                    continue

                df = pd.DataFrame(np.array(response)).set_index(0).iloc[:, :5]
                df.index = list(map(lambda x: datetime.fromtimestamp(int(x) / 1000), df.index)) # modify to datetime format.
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)

                if True:  # Always show process for debugging
                    self.sys_log.debug(f"{self.symbol} {df.index[0]} --> {df.index[-1]}")

                df_list.append(df)
                
            except Exception as e:
                msg = f"error in klines : {str(e)}"
                self.sys_log.error(msg)
                if 'sum_df' not in msg:
                    self.push_msg(msg)
                time.sleep(self.config.trader_set.api_term)
                
            else:
                if timesleep:
                    time.sleep(timesleep)

        if df_list:
            sum_df = pd.concat(df_list)

            return sum_df[~sum_df.index.duplicated(keep='last')]


def set_price_and_open_signal(self, mode='OPEN', env='BANK'):
    
    """
    v2.0
        Class mode.
    		use self, as parameter.
            modulize self.short_open_res1 *= phase.
            include np.timeidx
    v3.0
        turn off adj_wave_point. (messenger version)
        modify 'IDEP' to 'IDEP'
        
        v3.1
            modify to TableCondition_v0.3
        v3.2
            vivid mode.
    
    last confirmed at, 20240702 1342.
    """
        
    self.len_df = len(self.df_res)
    self.np_timeidx = np.array([intmin_np(date_) for date_ in self.df_res.index.to_numpy()])     
    close = self.df_res['close'].to_numpy()
    
    set_price_box(self, )
    set_price(self, close)       
    get_reward_risk_ratio(self, ) # default
    
    if env == 'IDEP':
        adj_price_unit(self,)


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


def get_side_info(self, side_open):
    
    if side_open == 'BUY':
        side_close = 'SELL'
        side_position = 'LONG'
        # if self.config.pos_set.long_fake:
        #     self.order_motion = 1
    else:
        side_close = 'BUY'
        side_position = 'SHORT'
        # if self.config.pos_set.short_fake:
        #     self.order_motion = 1

    return side_close, side_position


def get_price_entry(self,
                    df_res,
                    side_open,
                    price_entry):
    
    """
    v1.0
        validate if price_entry has been replaced with price_open.

    last confirmed at, 20240710 0850.
    """
    
    price_open = df_res['open'].to_numpy()[-1]  # price_open uses latest_index.

    if side_open == 'BUY':
        price_entry = min(price_open, price_entry)
    else:
        price_entry = max(price_open, price_entry)

    return price_entry


def get_balance_available(self, 
                          asset_type='USDT'):
    
    """
    v1.2
        Bank show_header=True.

    last confirmed at, 240712 1025.
    """
    
    try:      
        server_time = self.time()['data']['serverTime']
        response = self.balance(recvWindow=6000, timestamp=server_time)

        tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
        self.token_bucket.wait_for_token_consume(tokens_used) 
    except Exception as e:
        msg = "error in get_balance() : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(msg)
    else:
        available_asset = float([res['availableBalance'] for res in response['data'] if res['asset'] == asset_type][0])
        balance_available = self.calc_with_precision(available_asset, 2) # default for Binance

        return balance_available
        
        
        

def get_margin_consistency(self, 
                         margin,
                         account,
                         # balance_account,
                         # balance_min, 
                         mode="PROD"):
    
    """
    v1.0
        derived after get_balance_info v2.0
        modify to vivid mode.
            compare margin & TableAccount.
            Class mode remain, cause we are using core (public) object.
                like sys_log, tables, etc...

    last confirmed at, 20240701 0829.
    """

    # init.
    consistency = True

    # update self.table_account.balance_insufficient
    self.table_account.balance_insufficient = self.table_account.balance < self.table_account.balance_min  
    self.sys_log.debug("self.table_account : \n{}".format(self.table_account))  

    # get balance_available
    # check balance_available.
    balance_account_total = self.table_account.balance.sum()
    balance_available = get_balance_available(self)
    self.sys_log.info('balance_account_total : {:.2f}'.format(balance_account_total))
    self.sys_log.info('balance_available : {:.2f}'.format(balance_available))
    
    # reject balance_account_total over.
    if balance_available <= balance_account_total:
        self.sys_log.warning("over balance : balance_available {:.2f} <= balance_account_total {:.2f}".format(balance_available, balance_account_total))
        consistency = False

    
    # get a row by account.
    table_account_row = self.table_account[self.table_account['account'] == account]
    self.sys_log.debug("table_account_row : \n{}".format(table_account_row))

    if not table_account_row.empty:    
        # check min balance.
        if table_account_row.balance_insufficient.values[0]:
            self.sys_log.warning("table_account_row.balance_insufficient = True.")
            consistency = False
        
        # check balance < asset. 
        if table_account_row.balance.values[0] < margin:
            self.sys_log.warning("table_account_row.balance < margin {:.2f}".format(margin))
            consistency = False
    else:
        self.sys_log.warning("table_account_row is empty.")
        consistency = False

    if consistency:        
        self.table_account.loc[self.table_account['account'] == account, 'balance'] -= margin   

    return consistency
            
                   



def get_precision(self, 
                  symbol):
    
    """
    v2.0
        modify to vivid mode.
        
        v2.1
            add token.
        v2.2
            show_header=True.

    last confirmed at, 2024712 1026.
    """
    
    try:        
        response = self.exchange_info()
        
        tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
        self.token_bucket.wait_for_token_consume(tokens_used) 
        
    except Exception as e:
        msg = "error in get_precision : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(msg)

        # time.sleep(self.config.trader_set.api_term)
    else:
        precision_price, precision_quantity = [[data['pricePrecision'], data['quantityPrecision']] 
                                                         for data in response['data']['symbols'] if data['symbol'] == symbol][0]

        return precision_price, precision_quantity
        
        

def get_leverage_limit(self, 
                       symbol, 
                       price_entry, 
                       price_stop_loss,
                       fee_entry, 
                       fee_exit):
    
    """
    v2.0
        divide into server & user
            compare which one is minimal.
        leverage_limit is calculated by amount / target_loss	
        	quantity = target_loss / loss
        	amount = quantity * price_entry
        	
        	leverage_limit  = amount / target_loss
                ex) amount_required = 150 USDT, target_loss = 15 USDT, leverage_limit = 10.
        	leverage_limit  = ((target_loss / loss) * price_entry) / target_loss
        	leverage_limit  = ((1 / loss) * price_entry)
        	
        	leverage_limit  = price_entry / loss
        	loss = abs(price_entry - price_stop_loss) + (price_entry * fee_entry + price_stop_loss * fee_exit)
    v3.0
        modify to vivid input & output.
        
            v3.1
                modify leverage_limit_user logic.
                    more comprehensive.
            v3.2
                add token.
            v3.3
                show_header=True.

    last confirmed at, 20240712 1027.
    """
    
    # leverage_limit (server)
    server_time = self.time()['data']['serverTime']
    response = self.leverage_brackets(symbol=symbol, recvWindow=6000, timestamp=server_time)
    
    tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
    self.token_bucket.wait_for_token_consume(tokens_used) 

    
    leverage_limit_server = response['data'][0]['brackets'][0]['initialLeverage']    
    
    loss = abs(price_entry - price_stop_loss) + (price_entry * fee_entry + price_stop_loss * fee_exit)
    loss_pct = loss / price_entry * 100
    leverage_limit_user = np.maximum(1, np.floor(100 / loss_pct).astype(int))

    leverage_limit = min(leverage_limit_user, leverage_limit_server)

    return loss, leverage_limit_user, leverage_limit_server, leverage_limit



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


def get_order_info(self, 
                   symbol,
                   orderId):
    
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
    v2.0 
        vivid mode.
        
        v2.1
            apply token.
        v2.2
            show_header=True.
            recvWindow has been changed to 6000. (preventing 'Timestamp for this request is outside of the recvWindow' error)

    last confirmed at, 20240714 1939.
    """
 
    try:                
        server_time = self.time()['data']['serverTime']
        response = self.query_order(symbol=symbol, 
                                      orderId=orderId, 
                                      recvWindow=6000, 
                                      timestamp=server_time)
        
        tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
        self.token_bucket.wait_for_token_consume(tokens_used) 
        
        return response['data']
        
    except Exception as e:
        msg = f"error in get_order_info : {e}, tokens : {self.token_bucket.tokens}"
        self.sys_log.error(msg)
        self.push_msg(msg)
        

def get_price_realtime(self, symbol):
    
    """
    v3.0
        get price_market by self.price_market
        add non-error solution.
    v3.1
        remove loop & agg_trade
    v3.2
        restore agg_trade
    v3.3
        vivid mode.        
        access to self.price_market by symbol.
            we don't need token for it.        
    
    last confirmed at, 20240713 2328.
    """
        
    try:
        price_realtime =  self.price_market[symbol]
    except Exception as e:
        price_realtime = np.nan   
        
        if symbol not in self.price_market.keys():
            self.websocket_client.agg_trade(symbol=symbol, id=1, callback=self.agg_trade_message_handler)
            msg = "error in get_price_realtime : {} added to websocket_client.agg_trade".format(symbol)
        else:                
            msg = "error in get_price_realtime : {}".format(e)
        self.sys_log.error(msg)
    
    return price_realtime


def check_expiration(side_position,
                    price_realtime, 
                    price_expiration):
    
    """
    Checks if the position has expired based on the real-time price and the expiration price.

    Parameters:
    - price_realtime: float, the current real-time price.
    - price_expiration: float, the price at which the position expires.
    - side_position: str, the position side, either 'SHORT' or 'LONG'.

    Returns:
    - expired: int, 1 if the position has expired, 0 otherwise.
    """
    
    expired = 0
    if side_position == 'SHORT':
        if price_realtime <= price_expiration:
            expired = 1
    else:  # side_position == 'LONG'
        if price_realtime >= price_expiration:
            expired = 1
    return expired


def check_stop_loss(self,
                    side_open,
                    price_realtime,
                    price_liquidation,
                    price_stop_loss):

    """
    v3.0
        1. get price_realtime from outer_scope
        2. inversion not considered yet. (Todo)
        3. add self.log_out
        4. add non_out.
    v4.0
        init order_market_on = None from outer scope.
    v4.1
        vivid mode.
            remove eval, considering security & debug future problem.
        
    last confirmed at, 20240701 2300.
    """
    
    order_market_on = False
            
    if side_open == 'SELL':
        if price_realtime >= price_stop_loss:
            order_market_on = True
            self.sys_log.info("price_realtime {} >= price_stop_loss {}".format(price_realtime, price_stop_loss))
    else:
        if price_realtime <= price_stop_loss:
            order_market_on = True
            self.sys_log.info("price_realtime {} <= price_stop_loss {}".format(price_realtime, price_stop_loss))

    self.sys_log.info("order_market_on : {}".format(order_market_on))

    return order_market_on


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


def order_cancel(self, 
                symbol,
                orderId):    
    
    """
    v1.0
        add table_order logic.
    v2.1
        show_header=True.

    last confirmed at, 20240712 1028.
    """
    
    try:     
        server_time = self.time()['data']['serverTime']
        response = self.cancel_order(symbol=symbol, 
                              orderId=orderId, 
                              timestamp=server_time)
        
        tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
        self.token_bucket.wait_for_token_consume(tokens_used) 
        
    except Exception as e:        
        msg = "error in order_cancel : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(msg)
    else:
        self.sys_log.info("{} {} canceled.".format(symbol, orderId))


def get_quantity_unexecuted(self, 
                            symbol,
                            orderId):

    """
    v2.0
        apply updated get_precision
    v3.0
        vivid mode.
        
        v3.1
            allow Nonetype return for order_info.

    last confirmed at, 20240715 1954.
    """

    order_cancel(self, 
                symbol,
                orderId)
    
    self.order_info = get_order_info(self, 
                                    symbol,
                                    orderId)

    if self.order_info is None:
        return 0
        
    quantity_unexecuted = abs(float(self.order_info['origQty'])) - abs(float(self.order_info['executedQty']))
    
        # get price, volume updated precision
    precision_price, \
    precision_quantity = get_precision(self, 
                                            symbol)
    self.sys_log.info('precision_quantity : {}'.format(precision_quantity))
    
    quantity_unexecuted = self.calc_with_precision(quantity_unexecuted, precision_quantity)
    self.sys_log.info('quantity_unexecuted : {}'.format(quantity_unexecuted))
    self.sys_log.info('quantity_unexecuted (adj. precision) : {}'.format(quantity_unexecuted))

    return quantity_unexecuted


def order_limit(self, 
                symbol,
                side_order, 
                side_position, 
                price, 
                quantity):
    
    """
    v2.0
        1. symbol included.
        2. rename to retry_count & -1111 error update.
    v3.0
        Class mode
            remove order_data
    v4.0
        vivid mode.
            remove orderType... we don't need this in order_'limit'.        
        v4.1
            show_header=True.
            
    last confirmed at, 20240712 1029.
    """
        
    # init.  
    order_result = None
    error_code = 0
    
    try:        
        server_time = self.time()['data']['serverTime']
        response = self.new_order(timeInForce=TimeInForce.GTC,
                                        symbol=symbol,
                                        side=side_order,
                                        positionSide=side_position,
                                        type=OrderType.LIMIT,
                                        quantity=str(quantity),
                                        price=str(price),
                                        timestamp=server_time)
        
        tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
        self.token_bucket.wait_for_token_consume(tokens_used) 

        order_result = response['data']        
        self.sys_log.info("order_limit succeed. order_result : {}".format(order_result))
        
    except Exception as e:
        msg = "error in order_limit : {}".format(e)
        self.sys_log.error(msg)
        # self.push_msg(msg)

        # error casing. (later)

        # 1. order_limit() 에서 해결할 수 없는 error 일 경우, return
        #       a. -4003 : quantity less than zero
        # if "-4003" in str(e):
        #     error_code = -4003
        error_code = -4003

    return order_result, error_code



def order_market(self, 
                 symbol,
                 side_order,
                 side_position,
                 quantity):

    """
    v2.0
        update retry_count        
    v3.0
        remove order_res_list.
        remove error solution phase.
        
        v3.1
            replace get_quantity_unexecuted()
    v4.0
        vivid mode.

        v4.1
            show_header=True.
        
    last confirmed at, 20240712 1030.
    """
    
    while 1:

        # quantity_unexecuted = get_quantity_unexecuted(self, 
        #                                             symbol,
        #                                             orderId)

        # order_market
        order_result = None
        error_code = 0
        
        # while 1:
        try:
            server_time = self.time()['data']['serverTime']
            response = self.new_order(symbol=symbol,
                                        side=side_order,
                                        positionSide=side_position,
                                        type=OrderType.MARKET,
                                        quantity=str(quantity),
                                        timestamp=server_time)
            
            tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M'))
            self.token_bucket.wait_for_token_consume(tokens_used) 

            order_result = response['data']            
            self.sys_log.info("order_market succeed. : {}".format(order_result))
            
        except Exception as e:
            msg = "error in order_market : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)                
            time.sleep(self.config.trader_set.order_term)
            continue

            # # -2022 ReduceOnly Order is rejected
            # if '-2022' in str(e):
            #     error_code = '-2022'
            #     break

            # # -4003 quantity less than zero
            # if '-4003' in str(e):
            #     error_code = '-4003'
            #     break
                
            # # -1111 : Precision is over the maximum defined for this asset
            #     # = quantity precision error


        # 4. term for quantity consumption.
        time.sleep(1)

        if order_result:

            # order_result doesn't update executedQty. (remain at zero)
            
            # symbol = order_result['symbol'] # we have symbol column in table already.
            # orderId = order_result['orderId']        
            
            self.order_info = get_order_info(self, 
                                           order_result['symbol'],
                                           order_result['orderId'])
        
            if self.order_info['status'] == 'FILLED':
                self.sys_log.info("order_market filled.")
                return order_result, error_code
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
                    currency="USDT"):    

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
        
        v4.1
            push_msg included to Bank.

    last confimred at, 20240702 1353.    
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

    self.push_msg(msg)

    return income, income_accumulated, profit, profit_accumulated



    








        