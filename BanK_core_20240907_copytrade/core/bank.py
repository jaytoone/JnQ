
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
import re


import hashlib
from psycopg2 import sql
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_values


import pickle
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import gzip
import os
# import logging.config

from easydict import EasyDict
import json
from ast import literal_eval

import telegram
from telegram.ext import Updater
from telegram.ext import MessageHandler, Filters

from IPython.display import clear_output

pd.set_option('mode.chained_assignment',  None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


class DatabaseManager:
    """
    v0.1 - Init.
    v0.1.1 - update replace_table.      
    v0.2
        - add
            IDEP monitor methodd. 
            insert_table
            fetch_trade_result
         - update
            replace_table v1.3.1
            fetch_table_result v0.2
    v0.3
        - update
            fetch_table_result v0.4
            fetch_df_res v0.3
    v0.3.1
        - modify
            use path_dir_table
        - update
            fetch_table_trade v0.5
    
    Last confirmed, 20240929 1439.
    """
    
    def __init__(self, dbname, user, password, host, port, path_dir_table):
        self.pool = SimpleConnectionPool(1,   # min thread
                                         10,  # max thread
                                         dbname=dbname, 
                                         user=user, 
                                         password=password,
                                         host=host, 
                                         port=port)
        self.path_dir_table = path_dir_table
        self.file_hashes = {}
        
    def insert_table(self, schema, table_name, df, conflict_columns):
        """
        Generalized function: Insert DataFrame into table
            use for table_result, df_res.
        """
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Convert each row of the DataFrame to a tuple
                values = [tuple(row) for row in df.itertuples(index=False, name=None)]
                
                # Safely format column names
                columns = [sql.Identifier(col) for col in df.columns]
                
                # Construct query
                query = sql.SQL("""
                    INSERT INTO {}.{} ({})
                    VALUES %s
                    ON CONFLICT ({}) DO NOTHING
                """).format(
                    sql.Identifier(schema),
                    sql.Identifier(table_name),
                    sql.SQL(', ').join(columns),
                    sql.SQL(', ').join(map(sql.Identifier, conflict_columns))
                )
                
                # Use execute_values to insert multiple rows
                execute_values(cur, query, values)
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error occurred: {e}")
        finally:
            self.pool.putconn(conn)      

    def fetch_trade_result(self, schema_name, config, datetime_before=None):
        """
        v0.3
            - update
                Added datetime_before parameter as a string to filter rows based on timestamp.
                datetime_before is converted to Unix timestamp to match the int type of the timestamp column.
        v0.4
            - update
                Added schema_name parameter to allow specifying a schema.
                Added datetime_before parameter as a string to filter rows based on timestamp.
                datetime_before is converted to Unix timestamp to match the int type of the timestamp column.
        v0.5
            - update
                table name.

        Last confirmed: 20240928 1750.
        """

        conn = self.pool.getconn()
        try:
            cursor = conn.cursor()

            # Construct the dynamic table name based on the partitioned columns
            table_name = f"{config['priceBox_indicator']}_{config['point_mode']}_{config['point_indicator']}_{config['zone_indicator']}"

            # Start constructing the query
            query = sql.SQL("""
                SELECT * FROM {schema_name}.{table_name}
                WHERE "priceBox_value" = {priceBox_value}
                    AND "point_value" = {point_value}
                    AND "zone_value" = {zone_value}
                    AND "interval" = {interval}
            """)

            # Finalize query formatting
            query = query.format(
                schema_name=sql.Identifier(schema_name),
                table_name=sql.Identifier(table_name),
                priceBox_value=sql.Literal(config['priceBox_value']),
                point_value=sql.Literal(config['point_value']),
                zone_value=sql.Literal(config['zone_value']),
                interval=sql.Literal(config['interval'])
            )

            # Convert datetime_before string to Unix timestamp if provided
            if datetime_before is not None:
                datetime_obj = datetime.strptime(datetime_before, '%Y-%m-%d %H:%M:%S')
                datetime_before_ts = int(time.mktime(datetime_obj.timetuple()))
            else:
                datetime_before_ts = None

            # Add datetime_before condition if provided
            if datetime_before_ts is not None:
                query = query + sql.SQL(" AND timestamp_entry <= {datetime_before_ts}").format(
                    datetime_before_ts=sql.Literal(datetime_before_ts)
                )

            cursor.execute(query)
            rows = cursor.fetchall()

            if rows:
                columns = [desc[0] for desc in cursor.description]
                result_df = pd.DataFrame(rows, columns=columns)
            else:
                result_df = pd.DataFrame()

            cursor.close()
            return result_df
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
        finally:
            self.pool.putconn(conn)

    def distinct_symbol_from_table(self, schema, table_name):
        
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                
                cur.execute(sql.SQL("""
                    SELECT DISTINCT symbol FROM {}.{}
                """).format(
                    sql.Identifier(schema), 
                    sql.Identifier(table_name)
                ))
                
                result = cur.fetchall()
                unique_symbols = [row[0] for row in result]
                return unique_symbols
        except Exception as e:
            print(f"Error get_symbol_from_table {table_name}: {e}")
            return []
        finally:
            self.pool.putconn(conn)        

    def fetch_df_res(self, schema, table_name, symbol, limit=None, datetime_after=None, datetime_before=None):
        """
        v0.1
            315 ms ± 4.23 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            - add
                order by.
                limit parameter to fetch limited rows.
            - modify
                fetch the last 'limit' rows from the sorted table.
        v0.2
            - update
                Added datetime_after parameter to filter rows based on timestamp.
        v0.3
            - update
                Added datetime_before parameter to filter rows based on timestamp.

        Last confirmed: 20240825 2127.
        """
        
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Build the WHERE clause
                where_clause = sql.SQL("WHERE symbol = {}").format(sql.Literal(symbol))
                
                # Add datetime_after condition if provided
                if datetime_after is not None:
                    where_clause = where_clause + sql.SQL(" AND datetime >= {}").format(sql.Literal(datetime_after))
                
                # Add datetime_before condition if provided
                if datetime_before is not None:
                    where_clause = where_clause + sql.SQL(" AND datetime <= {}").format(sql.Literal(datetime_before))
                
                # Construct the full query
                query = sql.SQL("""
                    SELECT * FROM (
                        SELECT * FROM {}.{} {} ORDER BY timestamp DESC
                        {}
                    ) subquery ORDER BY timestamp
                """).format(
                    sql.Identifier(schema), 
                    sql.Identifier(table_name), 
                    where_clause,
                    sql.SQL('LIMIT {}').format(sql.Literal(limit)) if limit is not None else sql.SQL('')
                )
                
                cur.execute(query)
                result = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return pd.DataFrame(result, columns=columns).set_index('datetime')
        except Exception as e:
            print(f"Error fetching data from {table_name}: {e}")
            return pd.DataFrame()
        finally:
            self.pool.putconn(conn)

    
    def calculate_file_hash(self, file_path):
        """Calculate the hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def fetch_table(self, table_name, limit=None, primary_key='id'):
        """
        v1.1 - using engine, much faster.
        v1.2 - replace engine with psycopg2.
        v1.2.1 - update using pkey for ordering.

        Last confirmed: 2024-07-24 10:02
        """
        
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                query = f"SELECT * FROM {table_name}"
                if primary_key:
                    query += f" ORDER BY {primary_key}"
                if limit:
                    query += f" LIMIT {limit}"
                cur.execute(query)
                result = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return pd.DataFrame(result, columns=columns)
        except Exception as e:
            print(f"error fetching data from {table_name}: {e}")
            return pd.DataFrame()
        finally:
            self.pool.putconn(conn)
            
    def fill_missing_ids(self, df, id_column='id'):
        # Find the maximum id value
        max_id = df[id_column].max()
        if pd.isna(max_id):
            max_id = 0
        else:
            max_id = int(max_id)

        # Identify missing id values and fill them
        missing_idx = df[df[id_column].isna() | (df[id_column] == 0)].index
        df.loc[missing_idx, id_column] = range(max_id + 1, max_id + len(missing_idx) + 1)

        return df     
          
    def replace_table(self, df, table_name, send=False, mode=['UPDATE'], primary_keys=['id'], batch_size=1000):
        """
        v1.1 using engine, much faster.
        v1.2 using temp_table, stay as consistent state.
        v1.2.1 modify to psycopg2, considering latency.
        v1.2.2 apply pool.
        v1.2.3 replace to truncate instead deletion.
        v1.2.4 add scheduler.
        v1.3 - modify to use update, not upload.
            csv header = True.
        v1.3.1 
            - modify
                to use execute_values
            - fix 
                issues with execute_values usage and delete handling
        v1.4
            - update
                save csv into self.path_dir_table dir.

        Last confirmed:, 20240908 0836.
        """
        
        df = self.fill_missing_ids(df)
            
        temp_csv = f'{table_name}.csv'
        path_csv = os.path.join(self.path_dir_table, temp_csv)
        df.to_csv(path_csv, index=False, header=True)

        if send:        
            new_hash = self.calculate_file_hash(path_csv)
            if self.file_hashes.get(temp_csv) == new_hash:
                print("File has not changed. Skipping replace_table.")
                return

            self.file_hashes[temp_csv] = new_hash

            existing_data = self.fetch_table(table_name)

            to_delete = pd.DataFrame()
            changes = pd.DataFrame()
            
            if 'DELETE' in mode:
                to_delete = existing_data[~existing_data.apply(tuple, 1).isin(df.apply(tuple, 1))]

            if 'UPDATE' in mode:
                changes = df[~df.apply(tuple, 1).isin(existing_data.apply(tuple, 1))]
            
            if not to_delete.empty or not changes.empty:
                conn = self.pool.getconn()
                try:
                    with conn.cursor() as cur:
                        if not to_delete.empty:
                            # Handle deletions
                            primary_keys_str = ' AND '.join([f'"{col}" = %s' for col in primary_keys])
                            delete_stmt = f"DELETE FROM {table_name} WHERE {primary_keys_str}"
                            cur.executemany(delete_stmt, [tuple(row[pk] for pk in primary_keys) for _, row in to_delete.iterrows()])
                        
                        if not changes.empty:
                            # Handle insertions/updates
                            columns = [sql.Identifier(col) for col in df.columns]
                            values = [tuple(row) for row in changes.itertuples(index=False, name=None)]
                            primary_keys_str = sql.SQL(', ').join(map(sql.Identifier, primary_keys))
                            on_conflict = sql.SQL(', ').join([sql.SQL(f'"{col}" = EXCLUDED."{col}"') for col in df.columns if col not in primary_keys])
                            insert_stmt = sql.SQL("""
                                INSERT INTO {} ({})
                                VALUES %s
                                ON CONFLICT ({}) DO UPDATE SET {}
                            """).format(
                                sql.Identifier(table_name),
                                sql.SQL(', ').join(columns),
                                primary_keys_str,
                                on_conflict
                            )
                            for start in range(0, len(values), batch_size):
                                execute_values(cur, insert_stmt, values[start:start + batch_size])
                        conn.commit()
                except Exception as e:
                    conn.rollback()
                    print(f"Error occurred: {e}")
                finally:
                    self.pool.putconn(conn)


class TokenBucket:
    """
    v0.1
        follow up server's api used-weight (tokens_used).
    v0.2
        - update
            simplify.
    v0.3
        - update
            refill based on tokens_used reset.
        - deprecate
            we cannot know reset 'tokens_used' info.
    v0.3.1
        - update
            removed unnecessary tokens_used parameter from wait_for_token_consume.
    v0.3.2
        - update
            divide try / exception on refill().
    v0.3.3
        - update
            set retry-after
    v0.4
        - optimization
            simplified error handling, removed redundant loops, improved clarity.
   
    Last confirmed at, 20240823 0743
    """

    def __init__(self, sys_log, server_timer, capacity):
        self.server_timer = server_timer
        self.server_time = int(time.time() * 1000)  # Initialize with local time
        self.capacity = capacity  # Maximum number of tokens the bucket can handle per minute
        self.tokens_used = 0
        self.lock = threading.Lock()
        self.sys_log = sys_log
        self.last_tokens_used = 0
        
    def refill(self):
        try:
            response = self.server_timer()
            self.server_time = response['data']['serverTime']
            self.tokens_used = int(response['header'].get('X-MBX-USED-WEIGHT-1M', 0))  # Default to 0 if not found
            
            # Reset tokens_used if it's less than last_tokens_used, indicating a reset
            if self.tokens_used < self.last_tokens_used:
                self.tokens_used = 0
                
        except Exception as e:
            self.sys_log.error(f"Error in TokenBucket.refill(): {e}")
            self.handle_refill_error(e)
            
        self.last_tokens_used = self.tokens_used  # Update last_tokens_used for the next cycle

    def handle_refill_error(self, error):
        """Handles errors during the refill process, including retry-after handling."""
        error_message = str(error)
        retry_after = 1  # Default to 1 second if 'retry-after' is not found

        # Extract the 'retry-after' value from the error message, if available
        retry_match = re.search(r"'retry-after':\s*'(\d+)'", error_message)
        if retry_match:
            retry_after = int(retry_match.group(1))
        else:
            self.sys_log.warning("Could not find the 'retry-after' value.")
        
        # Extract the 'x-mbx-used-weight-1m' value from the error message, if available
        tokens_match = re.search(r"'x-mbx-used-weight-1m':\s*'(\d+)'", error_message)
        if tokens_match:
            self.tokens_used = int(tokens_match.group(1))
        else:
            self.sys_log.warning("Could not find the 'x-mbx-used-weight-1m' value.")
        
        # Wait for the 'retry-after' duration before retrying
        time.sleep(retry_after)
        self.refill()  # Retry the refill after waiting

    def check(self, tokens_needed):
        with self.lock:
            self.refill()  # Fetch the latest tokens_used from the server
            
            tokens_available = self.capacity - self.tokens_used
            self.sys_log.debug(f"tokens_available: {tokens_available}")
            return tokens_available >= tokens_needed

    def wait_for_tokens(self, tokens_needed=20):  # Minimum 20 tokens
        while not self.check(tokens_needed):
            time.sleep(1)  # Sleep briefly before rechecking


class Bank(UMFutures):

    """
    v0.1
        follow up server's api used-weight (tokens_used).
    v0.2
        using pool for sql.
    v0.3
        use DatabaseManager.
    v0.3.1
        - update
            use while loop.
            get_messenger_bot v3.1
    v0.3.2
        - update
            use log_name.   
    v0.4 
        - update
            use TokenBucket v0.3
    v0.4.1
        - modify
            to use config values.

    Last confirmed, 20240907 2001.
    """
    
    def __init__(self, **kwargs):
        
        self.path_config = kwargs['path_config']
        with open(self.path_config, 'r') as f:
            self.config = EasyDict(json.load(f))   
            
        # make dirs.
        for path in ['path_dir_log', 'path_dir_table', 'path_dir_df_res']:
            os.makedirs(kwargs[path], exist_ok=True)
            
        self.path_log = os.path.join(kwargs['path_dir_log'], "{}.log".format(datetime.now().strftime('%Y%m%d%H%M%S')))
        self.path_dir_df_res = kwargs['path_dir_df_res']  
        
            
        api_key, secret_key = self.load_key(self.config.broker.path_api)        
        UMFutures.__init__(self, key=api_key, secret=secret_key, show_header=True)

        self.websocket_client = UMFuturesWebsocketClient()
        self.websocket_client.start()
        self.price_market = {}
        
        
        # logger.
        self.set_logger(self.config.bank.log_name)            
        
        # token.    
        self.token_bucket = TokenBucket(sys_log=self.sys_log,
                                        server_timer=self.time,
                                        capacity=kwargs['api_rate_limit'])

      
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
                                         port=db_port,
                                         path_dir_table=kwargs['path_dir_table'])        

        self.table_account_name = self.config.database.table_account_name
        self.table_condition_name = self.config.database.table_condition_name
        self.table_trade_name = self.config.database.table_trade_name
        self.table_log_name = self.config.database.table_log_name
        
        self.table_account = self.db_manager.fetch_table(self.table_account_name)
        self.table_condition = self.db_manager.fetch_table(self.table_condition_name)
        self.table_trade = self.db_manager.fetch_table(self.table_trade_name)
        self.table_log = self.db_manager.fetch_table(self.table_log_name)
        
        # update_balance with margin data info table_trade.
        self.table_account = self.update_balance()

        
        # messegner
        self.chat_id = self.config.messenger.chat_id
        self.token = self.config.messenger.token
        
        self.get_messenger_bot()

        
        # income
        self.income = 0.0
        self.income_accumulated = self.config.bank.income_accumulated
        
        # profit
        self.profit = 0.0
        self.profit_accumulated = self.config.bank.profit_accumulated
        
        
        
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

    def namer(self, name):
        return name + ".gz"

    def rotator(self, source, dest):
        with open(source, "rb") as sf:
            with gzip.open(dest, "wb") as df:
                df.writelines(sf)
        os.remove(source)

    def set_logger(self, name='Bank'):
        
        """
        v1.0
            add RotatingFileHandler
        v1.1
            - modify
                sequence of code.
                add name param.
        v1.2
            add file compressure.
        v1.2.1
            - modify
                now use path_log.

        Last confirmed, 20240908 0908.
        """      
        
        self.sys_log = logging.getLogger(name)  
        self.sys_log.setLevel(logging.DEBUG)   
        
        simple_formatter = logging.Formatter("[%(name)s] %(message)s")
        complex_formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] - %(message)s")
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.DEBUG)
        
        # Use TimedRotatingFileHandler with a daily rotation
        timed_handler = TimedRotatingFileHandler(self.path_log, when='midnight', interval=1, backupCount=2)
        timed_handler.setFormatter(complex_formatter)
        timed_handler.setLevel(logging.DEBUG)

        # Add RotatingFileHandler to limit file size
        rotating_handler = RotatingFileHandler(self.path_log, maxBytes=100 * 1024 * 1024, backupCount=24)
        rotating_handler.setFormatter(complex_formatter)
        rotating_handler.setLevel(logging.DEBUG)
        # rotating_handler.rotator = self.rotator
        # rotating_handler.namer = self.namer

        # Adding handlers
        self.sys_log.addHandler(console_handler)
        # self.sys_log.addHandler(timed_handler)
        self.sys_log.addHandler(rotating_handler)
        
        # Prevent propagation to the root logger
        self.sys_log.propagate = False
        
    def echo(self, update, context):    
        self.user_text = update.message.text

    def get_messenger_bot(self):

        """
        v2.0
            use token from self directly.
        v3.0
            remove self.msg_bot exist condition.
        v3.1
            add check to prevent re-initialization if self.msg_bot is already assigned.

        Last confirmed: 2024-07-24 14:00
        """
        
        # Check if msg_bot is already assigned
        if hasattr(self, 'msg_bot') and self.msg_bot is not None:
            self.sys_log.debug("msg_bot is already assigned.")
            return
        
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
        
        msg = f"msg_bot {self.token} assigned."
        self.sys_log.debug(msg)
        self.push_msg(msg)
        
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
            
    def update_balance(self):
        """
        Updates the 'balance' column in the 'table_account' DataFrame by subtracting the total margin 
        from the 'balance_origin' for each account.
        
        Parameters:
        self (object): An object that contains the 'table_trade' and 'table_account' DataFrames.

        Returns:
        pd.DataFrame: The updated 'table_account' DataFrame with the new balance values.
        """
        # Calculate the total margin for each account
        total_margin_df = self.table_trade.groupby('account')['margin'].sum().reset_index()

        # Merge the total margin with the account table
        df_account = self.table_account.merge(total_margin_df, on='account', how='left')
        
        # Fill NaN values in the margin column with 0
        df_account['margin'] = df_account['margin'].fillna(0)

        # Update the balance by subtracting the total margin from balance_origin
        df_account['balance'] = df_account['balance_origin'] - df_account['margin']

        # Optionally, drop the 'margin' column
        df_account.drop(columns=['margin'], inplace=True)

        # Return the updated account DataFrame
        return df_account

            
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
        v2.3
            while loop completion.

        Last confirmed: 2024-07-21 22:49
        """
        
        while True:
            try:       
                self.token_bucket.wait_for_tokens()
                                
                server_time = self.token_bucket.server_time
                response = self.change_leverage(symbol=symbol, 
                                    leverage=leverage, 
                                    recvWindow=6000, 
                                    timestamp=server_time)  
                
                self.sys_log.info('leverage changed to {}'.format(leverage))
                return
                
            except Exception as e:
                msg = "error in change_initial_leverage : {}".format(e)
                self.sys_log.error(msg)
                self.push_msg(msg)
                time.sleep(self.config.term.symbolic)


    def set_position_mode(self, dualSidePosition='true'):
        """
        v1.0
            pass error -4059 -4046
        v2.0
            show_header=True. (token)
        v2.1
            get persistency
        v2.2
            while loop completion.

        Last confirmed: 2024-07-21 22:49
        """
        
        while True:
            try:
                self.token_bucket.wait_for_tokens() 
                
                server_time = self.token_bucket.server_time
                response = self.change_position_mode(dualSidePosition=dualSidePosition,
                                                    recvWindow=6000,
                                                    timestamp=server_time)                
                
                self.sys_log.info("dualSidePosition is true.")
                return
                
            except Exception as e:
                if '-4059' in str(e): # 'No need to change position side.'
                    return
                msg = "error in set_position_mode : {}".format(e)
                self.sys_log.error(msg)
                self.push_msg(msg)
                time.sleep(self.config.term.symbolic)
            
    def set_margin_type(self, symbol, marginType='CROSSED'): # CROSSED / ISOLATED
        """
        v1.0
            pass error -4046
        v2.0
            vivid mode.
        v2.1
            show_header=True.
        v2.2
            while loop completion.

        Last confirmed: 2024-07-21 22:49
        """
        
        while True:
            try:            
                self.token_bucket.wait_for_tokens() 
                
                server_time = self.token_bucket.server_time
                response = self.change_margin_type(symbol=symbol, 
                                                marginType=marginType, 
                                                recvWindow=6000, 
                                                timestamp=server_time)                
                
                self.sys_log.info("margin type is {} now.".format(marginType))
                return
                
            except Exception as e:
                if '-4046' in str(e): # 'No need to change margin type.'
                    return
                msg = "error in set_margin_type : {}".format(e)
                self.sys_log.error(msg)
                self.push_msg(msg)
                time.sleep(self.config.term.symbolic)
     
            
def get_tickers(self, ):  
    return [info['symbol'] for info in self.exchange_info()['data']['symbols']]  
  
    
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
        - Add 
            `self.config.trader_set.get_df_new_timeout`.
        - Set 
            `df_res = None` in all return phases.
    v3.0 
        - Replace 
            `concat_candlestick` with v2.0 logic.
    v3.1 
        - Replace 
            with `bank.concat_candlestick`.
    v4.0 
        - Integrate 
            `concat_candlestick` logic directly into the function.
        - Modify 
            return behavior based on the presence of `df_list`.
    v4.1 
        - Add 
            token information.
        - Replace 
            `time.sleep` with `term.df_new`.
    v4.2 
        - Modify 
            fetch `df` in reverse chronological order and 
            sort `sum_df` in ascending chronological order.
        - Add 
            `wait_for_tokens` logic.
        - Update 
            to return `None` only, deprecating previous return behavior.
            Add handling for `-1122` error.
    v4.2.1 
        - Modify 
            behavior for handling consecutive errors.
            end_date assertion.
            return error_code.
        - update
            add server_time.
            add timegap error.
            add no symbol without error.
    v4.2.2
        - Modify
            focus on patternized error '01'.  (this pattern should be rejected.)   

    Last confirmed, 20240922 0840.
    """

    limit_kline = 1500
    assert limit <= limit_kline, f"assert limit <= limit_kline: ({limit_kline})"
    assert days > 0, f"assert days: {days} > 0"
    
        
    while True:
        
        if end_date is None:
            timestamp_end = time.time() * 1000
        else:
            assert end_date != str(datetime.now()).split(' ')[0], "end_date shouldd not be today."
            timestamp_end = int(datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59).timestamp() * 1000)
            
        timestamp_start = timestamp_end - days * 24 * 60 * 60 * 1000
        timestamp_unit = itv_to_number(interval) * 60 * 1000 * limit

        time_arr_start = np.arange(timestamp_start, timestamp_end, timestamp_unit)
        time_arr_end = time_arr_start + timestamp_unit


        df_list = []
        res_error_list = []

        for idx, (time_start, time_end) in enumerate(zip(reversed(time_arr_start), reversed(time_arr_end))):
            try:
                self.sys_log.debug(f"{datetime.fromtimestamp(time_start / 1000)} - {datetime.fromtimestamp(time_end / 1000)}")
                    
                self.token_bucket.wait_for_tokens()
                server_time = self.token_bucket.server_time
                
                response = self.klines(
                    symbol=self.symbol,
                    interval=interval,
                    startTime=int(time_start),
                    endTime=int(time_end),
                    limit=limit,
                    timestamp=server_time
                )

                response = response.get('data', [])  # Safely handle the response

                if not response:
                    res_error_list.append(0)     
                    msg = f"Error in get_df_new, res_error_list: {res_error_list}"                  
                    self.sys_log.error(msg)                    
                    self.push_msg(msg)
                    
                    # consequent error (= means data end, should return valid df_list.)  
                        # criteria : 100 (v) | 101 (sparse error)   
                    if df_list:                        
                        if res_error_list[-2:]  == [0, 0]:
                            sum_df = pd.concat(df_list).sort_index()
                            return sum_df[~sum_df.index.duplicated(keep='last')]
                    else:
                        # no symbol but no error...
                            # first idx response should not be normal.
                            # no more retries than 2.
                        if idx == 0 or len(res_error_list) > 2: 
                            return '-1122'
                        
                        time.sleep(timesleep)
                        continue
                else:
                    res_error_list.append(1)
                    
                    # sparsed data error.
                    if res_error_list[-2:]  == [0, 1]:
                        self.sys_log.error(f"Error in get_df_new, sparsed data error, res_error_list: {res_error_list}")
                        self.push_msg(msg)
                        return                
                    

                df = pd.DataFrame(np.array(response)).set_index(0).iloc[:, :5]
                df.index = list(map(lambda x: datetime.fromtimestamp(int(x) / 1000), df.index))  # Modify to datetime format.
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df = df.astype(float)

                df_list.append(df)
                self.sys_log.debug(f"{self.symbol} {df.index[0]} - {df.index[-1]}")
                

                # Check last row's time.
                if idx == 0:
                    time_gap = time.time() - datetime.timestamp(df.index[-1])
                    itv_number = itv_to_number(interval)
                    self.sys_log.debug(f"itv_number: {itv_number}")
                    
                    # we don't need data exceeded 2 min.
                    if time_gap > (itv_number + 1) * 60:
                        msg = f"Error in in get_df_new : time_gap {time_gap} exceeded ({itv_number} + 1) * 60"
                        self.sys_log.warning(msg)
                        return
                    
                    # give one more chance over 1 min.
                    elif time_gap > itv_number * 60:
                        df_list = []  # Reset until valid time comes.
                        msg = f"Warning in get_df_new : time_gap {time_gap}"
                        self.sys_log.warning(msg)
                        time.sleep(self.config.term.df_new)
                        break  # Start from the beginning.

            except Exception as e:
                msg = f"Error in klines : {e}"
                self.sys_log.error(msg)
                self.push_msg(msg)
                
                if '-1122' in msg: # no symbol error.
                    return '-1122'
                
                elif 'sum_df' not in msg:  
                    # What type of this message could be ?
                        # Length mismatch: Expected axis has 0 elements, new values have 5 elements
                    pass
                    
                time.sleep(self.config.term.df_new)

            else:
                if timesleep:
                    time.sleep(timesleep)

        if df_list:
            sum_df = pd.concat(df_list).sort_index()
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

    fee_entry = self.config.broker.fee_market # can be replaced later.
    fee_exit = self.config.broker.fee_market
    
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
    v1.3   
        get persistency.
    v1.3.1
        - modify
            to wait_for_tokens.
        
    Last confirmed: 2024-08-22.
    """
    
    while True:
        try:      
            self.token_bucket.wait_for_tokens() 
            
            server_time = self.token_bucket.server_time
            response = self.balance(recvWindow=6000, timestamp=server_time)
            
        except Exception as e:
            msg = "error in get_balance() : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)
        else:
            available_asset = float([res['availableBalance'] for res in response['data'] if res['asset'] == asset_type][0])
            balance_available = self.calc_with_precision(available_asset, 2) # default for Binance

            return balance_available
        
        time.sleep(self.config.term.balance)
        
        
def get_account_normality(self, 
                         account,
                         ):

    """
    v1.0 
        - modify
            derived after get_balance_info v2.0
            modify to vivid mode.
                compare margin & TableAccount.
                Class mode remain, cause we are using core (public) object.
                    like sys_log, tables, etc...
    v1.1
        - update
            func1 : check account balance sum normality
            func2 : account normality
            return balance.
    v1.1.1
        - modify
            return balance_origin.     
            not allow account = None
    v1.1.2
        - modify
            calc. balance_insufficient
            just warn, balance_account_total over case.
           

    Last confirmed: 20240926 0545.
    """

    # init.
    consistency = True
    balance = None
    balance_origin = None

    # # update self.table_account.balance_insufficient
    self.table_account.balance_insufficient = self.table_account.balance < self.table_account.balance_min  
    self.sys_log.debug("self.table_account : \n{}".format(self.table_account))  

    # get balance_available
    balance_account_total = self.table_account.balance.sum()
    balance_available = get_balance_available(self)
    self.sys_log.info('balance_account_total : {:.2f}'.format(balance_account_total))
    self.sys_log.info('balance_available : {:.2f}'.format(balance_available))
    
    # 1. reject balance_account_total over.
    if balance_available < balance_account_total:
        msg = "over balance : balance_available {:.2f} < balance_account_total {:.2f}".format(balance_available, balance_account_total)
        self.sys_log.warning(msg)
        self.push_msg(msg)
        # consistency = False
    
    # get a row by account.
    if account: # is not None.
        table_account_row = self.table_account[self.table_account['account'] == account]
        self.sys_log.debug("table_account_row : \n{}".format(table_account_row))

        # 2. account selection
        # 3. check margin available
        # 4. check min margin
        if not table_account_row.empty:
            if table_account_row.balance_insufficient.iloc[0]:
                msg = f"balance_insufficient: \n{table_account_row}"
                self.sys_log.warning(msg)
                self.push_msg(msg)
                consistency = False
            else:                
                balance = table_account_row.balance.iloc[0]     
                balance_origin = table_account_row.balance_origin.iloc[0]       
        else:
            msg = f"Empty row: \n{table_account_row}"
            self.sys_log.warning(msg)
            self.push_msg(msg)
            consistency = False

    return consistency, balance, balance_origin
            
            
                 
               
       

def get_precision(self, symbol):
    """
    v2.0
        modify to vivid mode.
    v2.1
        add token.
    v2.2
        show_header=True.
    v2.3
        while loop completion.

    Last confirmed: 2024-07-21 22:49
    """
    
    while True:
        try:       
            self.token_bucket.wait_for_tokens() 
             
            response = self.exchange_info()  
            precision_price, precision_quantity = [[data['pricePrecision'], data['quantityPrecision']] 
                                                         for data in response['data']['symbols'] if data['symbol'] == symbol][0]
            return precision_price, precision_quantity
            
        except Exception as e:
            msg = "error in get_precision : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)
            time.sleep(self.config.term.precision)      
        

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
    v3.4
        while loop completion.

    Last confirmed: 2024-07-21 22:49
    """
      
    while True:
        try:
            self.token_bucket.wait_for_tokens()            
            server_time = self.token_bucket.server_time
            
            response = self.leverage_brackets(symbol=symbol, recvWindow=6000, timestamp=server_time)                        
            leverage_limit_server = response['data'][0]['brackets'][0]['initialLeverage']    
            
            loss = abs(price_entry - price_stop_loss) + (price_entry * fee_entry + price_stop_loss * fee_exit)
            loss_pct = loss / price_entry * 100
            leverage_limit_user = np.maximum(1, np.floor(100 / loss_pct).astype(int))

            leverage_limit = min(leverage_limit_user, leverage_limit_server)

            return loss, leverage_limit_user, leverage_limit_server, leverage_limit
            
        except Exception as e:
            msg = "error in get_leverage_limit : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)
            time.sleep(self.config.term.leverage_limit)




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


def get_order_info(self, symbol, orderId):
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
    v2.3
        while loop completion.
    v2.3.1
        - update
            add exception for Token error.

    Last confirmed: 2024-08-20 18:49.
    """
 
    while True:
        try:         
            self.token_bucket.wait_for_tokens() 
                   
            server_time = self.token_bucket.server_time
            response = self.query_order(symbol=symbol, 
                                        orderId=orderId, 
                                        recvWindow=6000, 
                                        timestamp=server_time)            
            
            return response['data']
            
        except Exception as e:
            msg = f"error in get_order_info : {e}"
            self.sys_log.error(msg)
            self.push_msg(msg)
            time.sleep(self.config.term.order_info)
            

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
        - modify
            wrap with float.
    
    Last confirmed: 2024-07-22 10:54
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
    
    return float(price_realtime)


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
        self.token_bucket.wait_for_tokens() 
        
        server_time = self.token_bucket.server_time
        response = self.cancel_order(symbol=symbol, 
                              orderId=orderId, 
                              timestamp=server_time)
        
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
        - update
            show_header=True.    
    v4.2
        - update
            error case -4014.
            
    Last confirmed, 20241012 1248.
    """
    
    while 1:
        
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
            self.push_msg(msg)
            time.sleep(self.config.term.order_limit)      

            # error casing. (later)
            # -4014, 'Price not increased by tick size.',
            if "-4014" in str(e):
                price_precision_prev = self.get_precision_by_price(price)
                self.sys_log.debug(f"price_precision_prev: {price_precision_prev}")
                if price_precision_prev == 0:
                    error_code = -4014
                else:
                    price_precision_new = price_precision_prev - 1
                    price = self.calc_with_precision(price, price_precision_new)
                    self.sys_log.debug(f"price_precision_new: {price_precision_new}")
                    self.sys_log.debug(f"price: {price}")
                    continue
                            
            
            # -4003, 'quantity less than zero.'
                # if error cannot be solved in this loop, return
            elif "-4003" in str(e):
                error_code = -4003
            else:
                error_code = 'Unknown'

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
            self.token_bucket.wait_for_tokens() 
            
            server_time = self.token_bucket.server_time
            response = self.new_order(symbol=symbol,
                                        side=side_order,
                                        positionSide=side_position,
                                        type=OrderType.MARKET,
                                        quantity=str(quantity),
                                        timestamp=server_time)
            
            order_result = response['data']            
            self.sys_log.info("order_market succeed. : {}".format(order_result))
            
        except Exception as e:
            msg = "error in order_market : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)                
            time.sleep(self.config.term.order_market)        

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
            
        else:
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
        - modify
            push_msg integrated to Bank. 
    v4.1.1
        - update
            remove mode from this func.
        - modify 
            config info.
    v4.2
        - update
            replace OPEN X PARTIALLY_FILLED  rows.

    Last confimred: 20240907 1248.
    """

    # Convert cumQuote to float and filter rows
    table_log_valid = self.table_log.astype({'cumQuote': 'float'}).query('code == @code and cumQuote != 0')

    # Filter rows with 'OPEN' and 'PARTIALLY_FILLED' status
    open_partial_row = table_log_valid[
        (table_log_valid['order_way'] == 'OPEN') & (table_log_valid['status'] == 'PARTIALLY_FILLED')
    ]

    # Process the filtered rows
    for idx, row in open_partial_row.iterrows():
        self.order_info = get_order_info(self, row.symbol, row.orderId)
        
        if self.order_info:  # Proceed only if order information exists
            start_time = time.time()

            # Update the table_log with order information
            self.table_log.loc[idx, self.order_info.keys()] = self.order_info.values()

            # Log the elapsed time for updating table_log
            self.sys_log.debug(
                f"------------------------------------------------\n"
                f"LoopTableTrade : elapsed time, update table_log : {time.time() - start_time:.4f}s\n"
                f"------------------------------------------------"
            )

    # Copy and filter the table again (only if there are matching rows)
    if not open_partial_row.empty:    
        table_log_valid = self.table_log.astype({'cumQuote': 'float'}).query('code == @code and cumQuote != 0')

    
    fee_limit = self.config.broker.fee_limit
    fee_market = self.config.broker.fee_market
    table_log_valid['fee_ratio'] = np.where(table_log_valid['type'] == 'LIMIT', fee_limit, fee_market)
    
    table_log_valid['fee'] = table_log_valid['cumQuote'] * table_log_valid['fee_ratio']

    
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

    








        