
# Library for Futures.
from funcs.binance_f.model import *
from funcs.binance.um_futures import UMFutures
from funcs.binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
# from funcs.binance.error import ClientError


# Library for Bank.
# from funcs.binance.shield_module_v4_1 import ShieldModule
from funcs.public.constant import *
# from funcs.public.indicator import *
# from funcs.public.indicator_ import *
from funcs.public.broker import *


import os

import pandas as pd
import numpy as np
import math
import threading
import asyncio

import time
from datetime import datetime, timedelta
import re


import hashlib
import uuid
import io
from psycopg2 import sql
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_values


import pickle
import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

import gzip
from pathlib import Path
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
            add fetch_partition
            add insert_table
            add create_table
            add load_partition
            db_usage
    
    20241107 1658.
    """
    
    def __init__(self, dbname, user, password, host, port, path_dir_table):
        self.pool = SimpleConnectionPool(1,   # min thread
                                         20,  # max thread
                                         dbname=dbname, 
                                         user=user, 
                                         password=password,
                                         host=host, 
                                         port=port)
        self.path_dir_table = path_dir_table
        self.file_hashes = {}
        
        
    @staticmethod
    def get_table_name(config):
        # Extract values from the config dictionary
        priceBox_indicator = config.get("priceBox_indicator")
        point_mode = config.get("point_mode")
        point_indicator = config.get("point_indicator")
        zone_indicator = config.get("zone_indicator")

        # Concatenate the values to form the table name
        table_name = f"{priceBox_indicator}_{point_mode}_{point_indicator}_{zone_indicator}"
        
        return table_name
    
    
    @staticmethod
    def load_df_res(interval, start_year_month, end_year_month, base_directory=r'D:\Project\SystemTrading\Project\JnQ\anal\df_res'):
        """
        Load df_res data from Feather files for the specified symbol, interval, and date range.
        
        Args:
            symbol (str): The symbol for which to load data.
            interval (str): The interval of the data.
            start_year_month (str): The start date in 'YYYY-MM' format.
            end_year_month (str): The end date in 'YYYY-MM' format.
            base_directory (str): The base directory where the Feather files are stored.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded and filtered data.
        """
        # 시작 및 끝 연월을 datetime으로 변환
        start_date = pd.to_datetime(start_year_month, format='%Y-%m')
        end_date = pd.to_datetime(end_year_month, format='%Y-%m')
        
        # 저장된 Feather 파일이 있는 디렉토리 경로 설정
        data_dir = Path(base_directory) / interval
        
        # 통합할 데이터프레임 리스트
        dataframes = []
        
        # 디렉토리 내의 모든 Feather 파일 확인
        for file in data_dir.glob(f"{interval}_*.ftr"):
            # 파일 이름에서 연-월 정보 추출, 'YYYY-MM' 형식에만 매칭
            year_month_str = file.stem.split('_')[-1]        
            
            # 'YYYY-MM' 형식의 파일만 처리
            if '-' in year_month_str and len(year_month_str) == 7:
                try:
                    file_date = pd.to_datetime(year_month_str, format='%Y-%m')
                    
                    # 시작 연월과 끝 연월 사이에 있는 파일만 로드
                    if start_date <= file_date <= end_date:
                        df = pd.read_feather(file)
                        
                        if not df.empty:
                            dataframes.append(df)
                            print(f"Loaded and filtered {file.name}")
                except ValueError:
                    # 파일 이름 형식이 잘못된 경우 건너뜁니다.
                    continue
        
        # 모든 파일을 하나의 데이터프레임으로 통합
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print("All relevant files have been loaded, filtered, and concatenated.")
            return combined_df
        else:
            print("No files found within the specified date range.")
            return pd.DataFrame()  # 빈 데이터프레임 반환

               
    def load_partition(self, config, start_year_month, end_year_month, base_directory=r'D:\Project\SystemTrading\Project\JnQ\anal\table_trade_result'):
        # 필터링 조건 설정
        priceBox_value = config.get('priceBox_value')
        point_value = config.get('point_value')
        zone_value = config.get('zone_value')
        interval = config.get('interval')
        
        # 테이블 이름 설정
        table_name = self.get_table_name(config)
        
        # 저장된 Feather 파일이 있는 디렉토리 경로 설정
        data_dir = Path(base_directory) / table_name
        
        # 시작 및 끝 연월을 datetime으로 변환하여 비교할 수 있게 함
        start_date = pd.to_datetime(start_year_month, format='%Y-%m')
        end_date = pd.to_datetime(end_year_month, format='%Y-%m')
        
        # 통합할 데이터프레임 리스트
        dataframes = []
        
        # 디렉토리 내의 모든 Feather 파일을 확인
        for file in data_dir.glob(f"{table_name}_*.ftr"):
            # 파일 이름에서 연-월 정보 추출, 'YYYY-MM' 형식에만 매칭
            year_month_str = file.stem.split('_')[-1]
            
            # 'YYYY-MM' 형식의 파일만 처리
            if '-' in year_month_str and len(year_month_str) == 7:
                try:
                    file_date = pd.to_datetime(year_month_str, format='%Y-%m')
                    
                    # 시작 연월과 끝 연월 사이에 있는 파일만 로드
                    if start_date <= file_date <= end_date:
                        df = pd.read_feather(file)
                        
                        # 조건에 맞는 행만 필터링
                        filtered_df = df[
                            (df["priceBox_value"] == priceBox_value) &
                            (df["point_value"] == point_value) &
                            (df["zone_value"] == zone_value) &
                            (df["interval"] == interval)
                        ]
                        
                        # 필터링된 데이터프레임을 리스트에 추가
                        dataframes.append(filtered_df)
                        # print(f"Loaded and filtered {file.name}")
                except ValueError:
                    # 파일 이름 형식이 잘못된 경우 건너뜁니다.
                    continue
        
        # 모든 파일을 하나의 데이터프레임으로 통합
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print("All relevant files have been loaded, filtered, and concatenated.")
            return combined_df
        else:
            print("No files found within the specified date range.")
            return pd.DataFrame()  # 빈 데이터프레임 반환
        

    def create_table(self, schema, config):
            """
            v0.1
                - init
                
            20241101 1023.
            """
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    # Get the dynamic table name using the get_table_name function
                    table_name = self.get_table_name(config)

                    # Use psycopg2.sql to safely create identifiers for table and schema
                    create_table_sql = sql.SQL('''
                        CREATE TABLE IF NOT EXISTS {}.{} (
                            symbol varchar(50) NOT NULL,
                            "position" varchar(10) NOT NULL,
                            status varchar(10) NOT NULL,
                            idx_entry int4 NOT NULL,
                            idx_exit int4 NOT NULL,
                            price_take_profit float8 NOT NULL,
                            price_entry float8 NOT NULL,
                            price_stop_loss float8 NOT NULL,
                            price_exit float8 NOT NULL,
                            timestamp_entry int8 NOT NULL,
                            timestamp_exit int8 NOT NULL,
                            "priceBox_value" float4 NOT NULL,
                            point_value float4 NOT NULL,
                            zone_value float4 DEFAULT '-1'::integer NOT NULL,
                            "interval" varchar(10) NOT NULL,
                            CONSTRAINT {} PRIMARY KEY (symbol, timestamp_entry, "priceBox_value", point_value, zone_value, interval)
                        ) PARTITION BY RANGE (timestamp_entry);
                    ''').format(
                        sql.Identifier(schema),           # Schema name
                        sql.Identifier(table_name),       # Dynamic table name
                        sql.Identifier(f"{table_name}_pkey")  # Primary key constraint name
                    )
                    
                    # Execute the table creation query
                    cur.execute(create_table_sql.as_string(conn))
                    
                    # Partition creation DO block
                    partition_creation_sql = sql.SQL('''
                        DO $$
                        DECLARE
                            year INT := 2018;
                            month INT := 1;
                            start_ts BIGINT;
                            end_ts BIGINT;
                            partition_name TEXT;
                        BEGIN
                            WHILE year < 2026 LOOP
                                start_ts := EXTRACT(EPOCH FROM TO_TIMESTAMP(year || '-' || month || '-01', 'YYYY-MM-DD'));
                                end_ts := EXTRACT(EPOCH FROM (TO_TIMESTAMP(year || '-' || month || '-01', 'YYYY-MM-DD') + INTERVAL '1 month'));

                                partition_name := {} || '_'  || year || '_' || LPAD(month::TEXT, 2, '0');

                                EXECUTE format('
                                    CREATE TABLE IF NOT EXISTS {}.%I PARTITION OF {}.{}
                                    FOR VALUES FROM (%L) TO (%L);',
                                    partition_name, start_ts, end_ts);

                                month := month + 1;
                                IF month > 12 THEN
                                    month := 1;
                                    year := year + 1;
                                END IF;
                            END LOOP;
                        END $$;
                    ''').format(
                        sql.Literal(table_name),
                        sql.Identifier(schema),
                        sql.Identifier(schema),
                        sql.Identifier(table_name),
                    )

                    # Execute the partition creation query
                    cur.execute(partition_creation_sql.as_string(conn))
                    conn.commit()
                    print(f"Table {schema}.{table_name} created successfully with default partitions.")

            except Exception as e:
                if conn:
                    conn.rollback()
                print(f"Error occurred in create_table: {e}")
            finally:
                self.pool.putconn(conn)

        
    def insert_table(self, schema, table_name, df, conflict_columns):
        """
        v1.1    
            - update
                use COPY.
            - modify
                use temp table for conflicts.
        
        20241031 1400.    
        """
        conn = self.pool.getconn()
        try:
            # Create an in-memory buffer
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, header=False)
            buffer.seek(0)

            with conn.cursor() as cur:
                # Create a temporary table for data import
                temp_table_name = f"{table_name}_temp_{uuid.uuid4().hex}"
                
                cur.execute(sql.SQL("CREATE TEMP TABLE {} (LIKE {} INCLUDING ALL)")
                            .format(sql.Identifier(temp_table_name),
                                    sql.Identifier(schema, table_name)))

                # COPY data into the temporary table
                cur.copy_expert(
                    sql.SQL("COPY {} FROM STDIN WITH CSV").format(sql.Identifier(temp_table_name)),
                    buffer
                )

                # Insert from the temporary table into the target table with conflict handling
                insert_query = sql.SQL("""
                    INSERT INTO {}.{} 
                    SELECT * FROM {}
                    ON CONFLICT ({}) DO NOTHING
                """).format(
                    sql.Identifier(schema),
                    sql.Identifier(table_name),
                    sql.Identifier(temp_table_name),
                    sql.SQL(', ').join(map(sql.Identifier, conflict_columns))
                )

                cur.execute(insert_query)
                conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            print(f"Error occurred: {e}")

        finally:
            self.pool.putconn(conn)
            
            
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
                       

    def fetch_partition(bank, schema_name, config, start_partition, end_partition, limit=None):
        """
        v0.2.2
            - modify
                    use condition.
                    
        20241101 1007
        """
        conn = bank.pool.getconn()
        try:
            with conn.cursor() as cur:
                # Generate the range of partitions between start_partition and end_partition
                start_year, start_month = map(int, start_partition.split('_'))
                end_year, end_month = map(int, end_partition.split('_'))

                # Construct the dynamic table name based on the partitioned columns
                table_name = f"{config['priceBox_indicator']}_{config['point_mode']}_{config['point_indicator']}_{config['zone_indicator']}"
            
                # Create a list to store all partition names between start and end
                partition_names = []
                year, month = start_year, start_month
                while (year < end_year) or (year == end_year and month <= end_month):
                    partition_names.append(f"{table_name}_{year:04d}_{month:02d}")
                    # Move to the next month
                    if month == 12:
                        year += 1
                        month = 1
                    else:
                        month += 1                 

                # Construct a query to fetch data from all relevant partitions
                query_parts = []
                for partition in partition_names:
                    
                    query = sql.SQL("""
                        SELECT * FROM {schema_name}.{table_name}
                        WHERE "priceBox_value" = {priceBox_value}
                            AND "point_value" = {point_value}
                            AND "zone_value" = {zone_value}
                            AND "interval" = {interval}
                    """).format(
                        schema_name=sql.Identifier(schema_name),
                        table_name=sql.Identifier(partition),  # 수정된 부분
                        priceBox_value=sql.Literal(config['priceBox_value']),
                        point_value=sql.Literal(config['point_value']),
                        zone_value=sql.Literal(config['zone_value']),
                        interval=sql.Literal(config['interval'])
                    )
                    
                    query_parts.append(query)                
                    # print(f"partition appended: {partition}")   
                    
                full_query = sql.SQL(" UNION ALL ").join(query_parts)

                if limit:
                    full_query += sql.SQL(" LIMIT {}").format(sql.Literal(limit))
                
                # Execute the combined query
                cur.execute(full_query)
                result = cur.fetchall()
                columns = [desc[0] for desc in cur.description]

                # Return the result as a pandas DataFrame
                return pd.DataFrame(result, columns=columns)
        except Exception as e:
            print(f"Error fetching data from partitions: {e}")
            return pd.DataFrame()
        finally:
            bank.pool.putconn(conn)



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


    def fetch_table(self, table_name, usd_db=False, limit=None, primary_key=None):
        """
        v1.1 - using engine, much faster.
        v1.2 - replace engine with psycopg2.
        v1.2.1 - update using pkey for ordering.
        v1.2.2 
            - modify
                use read_csv.

        20241107 1642.
        """
        
        if usd_db:
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
                print(f"Error fetching data from {table_name}: {e}")
                return pd.DataFrame()
            finally:
                self.pool.putconn(conn)
                
        else:
            return pd.read_csv(os.path.join(self.path_dir_table, f'{table_name}.csv'))
                               
    
    def calculate_file_hash(self, file_path):
        """Calculate the hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()


            
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
            log file name

    Last confirmed, 20241017 1137.
    """
    
    def __init__(self, **kwargs):
        
        self.path_config = kwargs['path_config']
        with open(self.path_config, 'r') as f:
            self.config = EasyDict(json.load(f))   
            
        # make dirs.
        for path in ['path_dir_log', 'path_dir_table', 'path_dir_df_res']:
            os.makedirs(kwargs[path], exist_ok=True)
            
        self.path_log = os.path.join(kwargs['path_dir_log'], f"{self.config.bank.log_name}.log")
        
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
        
        self.table_account = self.db_manager.fetch_table(self.table_account_name, usd_db=self.config.database.usage)
        self.table_condition = self.db_manager.fetch_table(self.table_condition_name, usd_db=self.config.database.usage)
        self.table_trade = self.db_manager.fetch_table(self.table_trade_name, usd_db=self.config.database.usage)
        self.table_log = self.db_manager.fetch_table(self.table_log_name, usd_db=self.config.database.usage)
        
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
                msg = "Error in change_initial_leverage : {}".format(e)
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
                msg = "Error in set_position_mode : {}".format(e)
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
                msg = "Error in set_margin_type : {}".format(e)
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
        msg = "Error in get_df_new_by_streamer : {}".format(e)
        self.sys_log.error(msg)
        self.push_msg(msg)
        # self.kill_proc()  # error 처리할 필요없이, backtrader 프로그램 종료
        # return None, None   # output len 유지
    # else:
    #     return df_res


def get_df_new(self, symbol, interval, days, end_date=None, limit=1500, timesleep=None):
    
    """
    v4.2.2
        - focus on patternized error '01'.  (this pattern should be rejected.)
        - error 부연 설명 진행. 20250311 1916.
        - df phase modified  20250312 0830.
    v4.2.4
        - add 'symbol' param for asyncio. 20250318 2009.
    """
    
    # print(f"limit : {limit}, {type(limit)}")

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
                    symbol=symbol,
                    interval=interval,
                    startTime=int(time_start),
                    endTime=int(time_end),
                    limit=limit,
                    timestamp=server_time
                )

                response = response.get('data', [])  # Safely handle the response


                if not response:
                    res_error_list.append(0)     
                    msg = f"Error in {symbol} get_df_new, res_error_list: {res_error_list}"                  
                    self.sys_log.error(msg)                    
                    self.push_msg(msg)
                    
                    # consequent error (= means data end, should return valid df_list.)  
                        # criteria : 100 (v) | 101 (sparse error)   
                    if df_list:                        
                        if res_error_list[-2:]  == [0, 0]: # 일시적인 에러가 아니라고 판단하는 것. (아님을 판단하기 위해 0 을 두번까지 확인한다.)
                            sum_df = pd.concat(df_list).sort_index()
                            return sum_df[~sum_df.index.duplicated(keep='last')]
                        else:
                            pass # 일시적인 에러인지를 판단하기 위해 retry 기회 준다.
                    else:
                        # 저장된 유효 df 가 없는 경우
                            # idx = 0 는 error 가 없어야한다. (아닌 경우 invalid symbol 확률이 크다.)
                            # 재시도는 2번까지.
                        if idx == 0 or len(res_error_list) > 2: 
                            return '-1122'
                        
                        time.sleep(timesleep)
                        continue
                else:
                    res_error_list.append(1)
                    
                    # sparsed data error.
                    if res_error_list[-2:]  == [0, 1]: # 일시적인 에러, sparsed data 발생.
                        self.sys_log.error(f"Error in get_df_new, sparsed data error, res_error_list: {res_error_list}")
                        self.push_msg(msg)
                        return                
                    

                    df = pd.DataFrame(np.array(response)).set_index(0).iloc[:, :5]
                    df.index = list(map(lambda x: datetime.fromtimestamp(int(x) / 1000), df.index))  # Modify to datetime format.
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    df = df.astype(float)

                    df_list.append(df)
                    self.sys_log.debug(f"{symbol} {df.index[0]} - {df.index[-1]}")
                    

                    # Check Realtime df sync. by lastest row's timestamp.
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
                
                if '-1122' in msg: # no / invalid symbol error.
                    return '-1122'
                
                elif 'sum_df' not in msg:  
                    # Todo) What type of this message could be ?
                        # Length mismatch: Expected axis has 0 elements, new values have 5 elements
                    pass
                    
                time.sleep(self.config.term.df_new)

            else:
                if timesleep:
                    time.sleep(timesleep)

        if df_list:
            sum_df = pd.concat(df_list).sort_index()
            return sum_df[~sum_df.index.duplicated(keep='last')]     

        
def __________________0():
    pass


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
            msg = "Error in get_balance() : {}".format(e)
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
            msg = "Error in get_precision : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)
            time.sleep(self.config.term.precision)      
        



def get_leverage_brackets(self):
    """
    Creates a mapping of symbol to its bracket data from the leverage_brackets API response.
    
    Returns:
        data_by_symbol (dict): A dictionary mapping each symbol to its bracket information.
    """
    
    # Get the current server time
    server_time = self.time()['data']['serverTime']
    
    # Request leverage bracket information
    response = self.leverage_brackets(
        recvWindow=6000, 
        timestamp=server_time
    )

    # Extract data from the API response
    data = response['data']

    # Create a mapping of symbol to its bracket data
    data_by_symbol = {item['symbol']: item['brackets'] for item in data}
    
    return data_by_symbol



def get_leverage_limit(self, 
                       symbol, 
                       amount_entry,
                       loss,
                       price_entry, 
                       ):
    """
    v3.1
        modify leverage_limit_user logic.
            more comprehensive.
    v3.2
        add token.
    v3.3
        show_header=True.
    v3.4
        while loop completion.
    v3.5
        - update
            leverage_limit use amount_entry.

    Last confirmed, 20241012 1425.
    """
      
    while 1:
        try:
            leverage_brackets = get_leverage_brackets(self)
            
            # Find the appropriate bracket for the given symbol and amount_entry
            if symbol in leverage_brackets:
                brackets = leverage_brackets[symbol]
                
                # Use a list comprehension and min function to find the correct bracket quickly
                leverage_limit_server = min(
                    (bracket['initialLeverage'] for bracket in brackets 
                    if bracket['notionalFloor'] <= amount_entry <= bracket['notionalCap']),
                    default=np.inf
                )
            else:
                leverage_limit_server = np.inf  # If the symbol is not found, default to infinity or any other value

            
            loss_pct = loss / price_entry * 100
            leverage_limit_user = np.maximum(1, np.floor(100 / loss_pct).astype(int))

            leverage_limit = min(leverage_limit_user, leverage_limit_server)
                        
            self.sys_log.debug(f"leverage_limit_server: {leverage_limit_server}")
            self.sys_log.debug(f"leverage_limit_user: {leverage_limit_user}")

            return leverage_limit_user, leverage_limit_server, leverage_limit
            
        except Exception as e:
            msg = "Error in get_leverage_limit : {}".format(e)
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


        
def __________________1():
    pass


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
        - add orderId in error msg. 20250320 2158.
    """
 
    orderId = int(orderId)
    
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
            msg = f"Error in get_order_info, {symbol} {orderId}: {e}"
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
            msg = "Error in get_price_realtime : {} added to websocket_client.agg_trade".format(symbol)
        else:                
            msg = "Error in get_price_realtime : {}".format(e)
        self.sys_log.error(msg)
    
    return float(price_realtime)


def check_expiration(self, 
                    interval,
                    entry_timeIndex,
                    side_position,
                    price_realtime, 
                    price_expiration,
                    ncandle_game=None):
    """
    v0.2
        - Expired_x: 가격 조건에 따른 만료 (LONG: 현재가 >= 만료가, SHORT: 현재가 <= 만료가)
        - Expired_y: ncandle_game 값에 따른 만료. (예: ncandle_game == 2이면, 진입 시각 + (interval*2) 이후 만료)
    
    Returns:
        int: 만료 상태
             0 - 만료 아님,
             1 - 가격 조건(Expired_x)에 의한 만료,
             2 - ncandle_game 조건(Expired_y)에 의한 만료. 20250317 1355.        
    """
    expired = 0

    # 01 Expired_x: 가격 조건 검사
    if side_position == 'LONG':
        if price_realtime >= price_expiration:
            expired = 1
    else:  # SHORT인 경우
        if price_realtime <= price_expiration:
            expired = 1

    # 02 Expired_y: Candle Game 조건 검사
    if ncandle_game:
        itv_number = itv_to_number(interval)
        entry_timestamp = int(time.mktime(time.strptime(str(entry_timeIndex), "%Y-%m-%d %H:%M:%S")))
        
        # 예: ncandle_game==2이면, 진입 시각이 "04:15:00"일 경우 만료 시각은 "04:44:59"가 됨.
            # backtest 는 timeIndex 만으로 보기 때문에 헷갈릴 수 있는 부분. ("04:15:00", "04:30:00")
        if time.time() - entry_timestamp >= itv_number * 60 * ncandle_game:
            expired = 2

    # 필요한 입력 인자와 최종 expired 상태를 보기 좋게 로그로 남김.
    self.sys_log.info(
        "check_expiration: side=%s | price_realtime=%.2f | price_expiration=%.2f | expired=%d | interval=%s | entry_timeIndex=%s | ncandle_game=%s",
        side_position, price_realtime, price_expiration, expired, interval, entry_timeIndex, ncandle_game
    )
    
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
           


def check_closer(self,
                interval,
                entry_timeIndex,
                ncandle_game=None):    
    """
    v0.1
        - init. 20250317 1044.
    """
    
    order_market_on = False
    
    if ncandle_game: 
        itv_number = itv_to_number(interval)
        entry_timestamp = int(time.mktime(time.strptime(str(entry_timeIndex), "%Y-%m-%d %H:%M:%S")))
        
         # if ncandle_game == 2, 04:30:01 기준, 
            # 진입바의 timeIndex = 04:15:00, expiry 는 04:45:00 timeIndex 에 일어난다.         
        if time.time() - entry_timestamp > itv_number * 60 * ncandle_game:            
            order_market_on = True
            
    # 필요한 파라미터와 결과를 로그에 출력
    self.sys_log.info(
        "check_closer: order_market_on=%s | interval=%s | entry_timeIndex=%s | ncandle_game=%s",
        order_market_on, interval, entry_timeIndex, ncandle_game
    )    
    
    return order_market_on


def wait_barClosed(self, interval, entry_timeIndex):
    """
    v0.1
        - use interval, entry_timeIndex. 20250317 0926.
        - candle_close_time의 초와 마이크로초를 00으로 설정 (예: 09:30:00)
    """
    
    # entry_timeIndex를 datetime 객체로 변환 (예: "2025-03-17 04:15:00")
    try:
        entry_dt = datetime.strptime(entry_timeIndex, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        msg = "Error parsing entry_timeIndex {}: {}".format(entry_timeIndex, e)
        self.sys_log.error(msg)
        self.push_msg(msg)
        entry_dt = datetime.now()
    
    # itv_to_number()로 interval을 분 단위 숫자로 변환
    itv_number = itv_to_number(interval)
    
    # candle_close_time 계산 후 초와 마이크로초를 0으로 세팅
    candle_close_time = (entry_dt + timedelta(minutes=itv_number)).replace(second=0, microsecond=0)
    now = datetime.now()
    
    if now < candle_close_time:
        wait_seconds = (candle_close_time - now).total_seconds()
        self.sys_log.debug("Waiting {:.2f} seconds until candle close time: {}.".format(wait_seconds, candle_close_time))
        time.sleep(wait_seconds)
    else:
        self.sys_log.debug("Candle already closed at {}.".format(candle_close_time))


        
def __________________2():
    pass


def order_cancel(self, 
                symbol,
                orderId):    
    
    """
    v1.0
        add table_order logic.
    v2.1
        show_header=True.
    v2.2
        - -2011, 'Unknown order sent.' 다룬다. 20250321 1843.
    """
    
    canceled = True
    
    try:     
        self.token_bucket.wait_for_tokens() 
        
        server_time = self.token_bucket.server_time
        response = self.cancel_order(symbol=symbol, 
                              orderId=orderId, 
                              timestamp=server_time)
        
    except Exception as e:        
        msg = f"Error in order_cancel, {symbol} {orderId}: {e}"
        self.sys_log.error(msg)
        self.push_msg(msg)
        canceled = False 
    else:
        self.sys_log.info("{} {} canceled.".format(symbol, orderId))
        
    return canceled




def get_quantity_unexecuted(self, 
                            symbol,
                            orderId):

    """
    v2.0
        apply updated get_precision
    v3.0
        vivid mode.
    v3.1
        - allow Nonetype return for order_info.
    v3.2
        - adj. order_cancel v2.2 20250321 1843.
        - remove self.
    """

    # if canceled, return True.
    if order_cancel(self, 
                    symbol,
                    orderId):
    
        order_info = get_order_info(self, 
                                    symbol,
                                    orderId)

        if order_info: # = not None.
            
                # get price, volume updated precision
            _, \
            precision_quantity = get_precision(self, symbol)
            self.sys_log.info('precision_quantity : {}'.format(precision_quantity))
            
            quantity_unexecuted = abs(float(order_info['origQty'])) - abs(float(order_info['executedQty']))
            self.sys_log.info('quantity_unexecuted : {}'.format(quantity_unexecuted))            
            
            quantity_unexecuted = self.calc_with_precision(quantity_unexecuted, precision_quantity)
            self.sys_log.info('quantity_unexecuted (adj. precision) : {}'.format(quantity_unexecuted))
            
            return quantity_unexecuted
        else:      
            return 0  
    else:        
        return 0




def order_limit(self, 
                symbol,
                side_order, 
                side_position, 
                price, 
                quantity):
    
    """    
    v4.2
        - error case -4014.            
        - adj. dynamic loop_cnt (precision deviate by '-loop_cnt') 20250319 0642.
    v4.2.1
        - adj. loop_cnt_max. 20250319 0645.
        - add error msg 'symbol' 20250319 1906.
    """
    
    loop_cnt  = 0
    loop_cnt_max  = 3
    
    while 1:
        
        # init.  
        order_result = None
        error_code = 0
        
        try:        
            self.token_bucket.wait_for_tokens() 

            server_time = self.token_bucket.server_time
            response = self.new_order(timeInForce=TimeInForce.GTC,
                                            symbol=symbol,
                                            side=side_order,
                                            positionSide=side_position,
                                            type=OrderType.LIMIT,
                                            quantity=str(quantity),
                                            price=str(price),
                                            timestamp=server_time)
            
            order_result = response['data']        
            self.sys_log.info("order_limit succeed: {}".format(order_result))
            
        except Exception as e:
            msg = F"Error in order_limit : {symbol} {e}"
            self.sys_log.error(msg)
            self.push_msg(msg)
            time.sleep(self.config.term.order_limit)      


            ################################
            # Todo) Error Case
            ################################\
                
            # -4014 : 'Price not increased by tick size.',
                # get_precision doesn't work.
            if "-4014" in str(e):
                price_precision_prev = self.get_precision_by_price(price)
                self.sys_log.debug(f"price_precision_prev: {price_precision_prev}")
                
                if price_precision_prev == 0:
                    error_code = -4014
                else:
                     # 첫 번째 루프(0), 두 번째 루프(-1), 세 번째 루프(-2)
                    price_precision_new = price_precision_prev - loop_cnt
                    price = self.calc_with_precision(price, price_precision_new)
                    
                    self.sys_log.debug(f"price_precision_new: {price_precision_new}")
                    self.sys_log.debug(f"price: {price}")
                    
                    loop_cnt += 1  # loop_cnt 증가

                    # 최대 3회 재시도 후 실패하면 오류 반환
                    if loop_cnt <= loop_cnt_max:
                        continue
                    else:
                        error_code = -4014
                            
            
            # -4003 : 'quantity less than zero.'
                # if error cannot be solved in this loop, return
            elif "-4003" in str(e):
                error_code = -4003
            else:
                error_code = 'Unknown'

        return order_result, error_code
  
    
def order_stop_market(self, 
                symbol,
                side_order, 
                side_position, 
                price, 
                quantity):
    
    """
    v0.1
        - init
    v0.2
        - adj. loop_cnt_max 20250321 1943.
    """
    
    loop_cnt  = 0
    loop_cnt_max = 3
    
    while 1:
        
        # init.  
        order_result = None
        error_code = 0
        
        try:        
            self.token_bucket.wait_for_tokens() 
            server_time = self.token_bucket.server_time
            
            response = self.new_order(timeInForce=TimeInForce.GTC,
                                        symbol=symbol,
                                        side=side_order,
                                        positionSide=side_position,
                                        type=OrderType.STOP_MARKET,
                                        quantity=str(quantity),
                                        stopPrice=str(price), 
                                        timestamp=server_time
                                        )
            
            order_result = response['data']        
            self.sys_log.info("order_stop_market succeed: {}".format(order_result))
            
        except Exception as e:
            msg = "Error in order_limit : {}".format(e)
            self.sys_log.error(msg)
            self.push_msg(msg)
            time.sleep(self.config.term.order_stop_market)      

            # error casing. (later)
            # -4014, 'Price not increased by tick size.',
            if "-4014" in str(e):
                
                price_precision_prev = self.get_precision_by_price(price)
                self.sys_log.debug(f"price_precision_prev: {price_precision_prev}")
                
                if price_precision_prev == 0:
                    error_code = -4014
                else:
                     # 첫 번째 루프(0), 두 번째 루프(-1), 세 번째 루프(-2)
                    price_precision_new = price_precision_prev - loop_cnt
                    price = self.calc_with_precision(price, price_precision_new)
                    
                    self.sys_log.debug(f"price_precision_new: {price_precision_new}")
                    self.sys_log.debug(f"price: {price}")
                    
                    loop_cnt += 1  # loop_cnt 증가

                    # 최대 3회 재시도 후 실패하면 오류 반환
                    if loop_cnt <= loop_cnt_max:
                        continue
                    else:
                        error_code = -4014
            
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
            self.sys_log.info("order_market succeed: {}".format(order_result))
            
        except Exception as e:
            msg = "Error in order_market : {}".format(e)
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
    v4.3 - table_log 직접 사용하도록 수정 20250319 2139.
    """
    
    # 유효한 데이터 필터링
    table_log_valid = table_log.astype({'cumQuote': 'float'}).query('code == @code and cumQuote != 0')
    
    # 'OPEN' & 'PARTIALLY_FILLED' 상태의 주문 처리
    open_partial_row = table_log_valid.query("order_way == 'OPEN' and status == 'PARTIALLY_FILLED'")
    
    for idx, row in open_partial_row.iterrows():
        order_info = get_order_info(self, row.symbol, row.orderId)
        if order_info:
            start_time = time.time()
            table_log.loc[idx, order_info.keys()] = order_info.values()
            self.sys_log.debug(f"LoopTableTrade : update table_log elapsed time: {time.time() - start_time:.4f}s")
    
    # 다시 필터링 (업데이트된 데이터 반영)
    if not open_partial_row.empty:
        table_log_valid = table_log.astype({'cumQuote': 'float'}).query('code == @code and cumQuote != 0')
    
    # 수수료 계산
    fee_limit = self.config.broker.fee_limit
    fee_market = self.config.broker.fee_market
    table_log_valid['fee_ratio'] = np.where(table_log_valid['type'] == 'LIMIT', fee_limit, fee_market)
    table_log_valid['fee'] = table_log_valid['cumQuote'] * table_log_valid['fee_ratio']
    
    # OPEN과 CLOSE 주문 분리
    table_log_valid_open = table_log_valid.query("order_way == 'OPEN'")
    table_log_valid_close = table_log_valid.query("order_way == 'CLOSE'")
    
    if table_log_valid_open.empty or table_log_valid_close.empty:
        self.sys_log.info(f"table_log_valid length insufficient: {table_log_valid}")
        return 0, income_accumulated, 0, profit_accumulated
    
    # 가장 최근 OPEN과 CLOSE 주문의 금액 및 수수료
    last_open = table_log_valid_open.iloc[-1]
    last_close = table_log_valid_close.iloc[-1]
    
    cumQuote_open = last_open.cumQuote
    cumQuote_close = last_close.cumQuote
    fee_open = last_open.fee
    fee_close = last_close.fee
    
    self.sys_log.info(f"cumQuote_open: {cumQuote_open}")
    self.sys_log.info(f"cumQuote_close: {cumQuote_close}")
    self.sys_log.info(f"fee_open: {fee_open}")
    self.sys_log.info(f"fee_close: {fee_close}")
    
    # 손익 계산
    income = (cumQuote_close - cumQuote_open if side_position == 'LONG' else cumQuote_open - cumQuote_close) - (fee_open + fee_close)
    income_accumulated += income
    
    self.sys_log.info(f"income: {income:.4f} {currency}")
    self.sys_log.info(f"income_accumulated: {income_accumulated:.4f} {currency}")
    
    # 수익률 계산 (수동 거래 개입 방지)
    profit = (income / cumQuote_open * leverage) if cumQuote_open != 0 else 0.0
    profit_accumulated += profit
    
    self.sys_log.info(f"profit: {profit:.4f}")
    
    # 메시지 푸시
    msg = (f"cumQuote_open : {cumQuote_open:.4f}\n"
           f"cumQuote_close : {cumQuote_close:.4f}\n"
           f"fee_open : {fee_open:.4f}\n"
           f"fee_close : {fee_close:.4f}\n"
           f"income : {income:.4f}\n"
           f"income_accumulated : {income_accumulated:.4f}\n"
           f"profit : {profit:.4f}\n"
           f"profit_accumulated : {profit_accumulated:.4f}\n")
    self.push_msg(msg)
    
    return income, income_accumulated, profit, profit_accumulated

    








        