from bank import *
from idep import *
from params import * 


from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


# executor = ThreadPoolExecutor(max_workers=10)
executor = ThreadPoolExecutor(max_workers=30)

@dataclass
class TradeInfo:
    symbol: str
    side_open: str
    
    position: str
    interval: str
    
    divider: float
    account: str
    
    df_res: object
    code: str
    
    price_expiration: float
    price_take_profit: float
    price_entry: float
    price_stop_loss: float




def loop_messenger(self,):

    """
    v2.0
        - simple messenger for check program alive. 20240925 0950.
    """
    
    if self.user_text is not None:

        user_text_upper_case = self.user_text.upper()        
        
        # default msg = invalid payload.
        msg = "error in self.user_text : invalid payload."
                                
        if 'WATCH' in user_text_upper_case:
            # msg = "Program is running..."
            msg = json.dumps(self.config.bank, indent=4)

        self.push_msg(msg)
        self.user_text = None     
        
        
async def loop_table_condition_async(self, drop=False, debugging=False):
    """
    v1.6.1
        - sleep for empty table.
    v1.7
        - adj, Anchor Concept. 20250314 1914.
        - add miliesec. to code (for prevent code duplication --> get_income_info affection) 20250318 1843.
        - adj. get_df_new v4.2.4 20250318 2020.
        - position 추적, balance > balance_orig error 수정 완료. 20250321 0540.
    v1.8.1
        - self. manual 수정. 20250321 0540.
    v2.1 (Async Hybrid)
        - get_df_new ~ df_res 분석까지 비동기화 처리
        - 조건 만족 시 init_table_trade는 동기화 처리 (lock)
    v2.1.1 
        - (Async Hybrid + Semaphore) 20250321 2300
        - 요청 타이밍 분산 적용 (random sleep) 20250322 2312.
        - while loop for complete 병렬. 20250322 0635.
        - save before 완전 병렬화 20250326 0715.
    """
    
    while True:
        
        loop_start = time.time()
        self.sys_log.debug(f"[loop_table_condition] START at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        if self.table_condition.empty:
            await asyncio.sleep(1)
            continue        

        tasks = []
        trade_candidates = []  # 조건 만족 결과 임시 저장 리스트
        semaphore = asyncio.Semaphore(10)   # stable for 2400 API limits
        # semaphore = asyncio.Semaphore(20)
        # semaphore = asyncio.Semaphore(30)

        for idx, row in self.table_condition.iterrows():

            # init.
            symbol = row.symbol
            side_open = row.side_open
            position = row.position
            interval = row.interval
            divider = row.divider
            account = row.account

            # do not repeat it in interval.
            interval_number = itv_to_number(interval)
            timestamp_current = int(time.time())
            if timestamp_current - int(row.timestampLastUsed) < interval_number * 60:
                continue
            else:
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()

                timestamp_anchor = 1718236800  # 2024-06-13 9:00:00
                timestamp_remain = (timestamp_current - timestamp_anchor) % (interval_number * 60)
                timestamp_current -= timestamp_remain
                self.table_condition.at[idx, 'timestampLastUsed'] = timestamp_current

                self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} check timestampLastUsed : %.4fs" % (time.time() - start_time))
                self.sys_log.debug("------------------------------------------------")

            # days 계산 및 비동기 fetch 작업 추가
            assert interval_number < 1440, "assert, inteval < 1d"
            days = math.ceil((1200 / (60 * 24 / interval_number)))  # 1200 : min len for indicators to be synced.

            # ✅ 인자로 현재 loop 변수들을 넘겨서 바로 고정시킴
            async def sem_task(idx_, symbol_, position_, interval_, divider_, account_, side_open_, days_):
                async with semaphore:
                    await asyncio.sleep(random.uniform(2, 2.5))
                    # await asyncio.sleep(random.uniform(1, 1.5))
                    # await asyncio.sleep(random.uniform(0.5, 1.5))
                    # await asyncio.sleep(random.uniform(0.1, 0.5))
                    # await asyncio.sleep(random.uniform(0.01, 0.05))
                    return await get_trade_info_async(
                        self, idx_, symbol_, position_, interval_, divider_, account_, side_open_, days_, debugging
                    )

            tasks.append(sem_task(idx, symbol, position, interval, divider, account, side_open, days))

        # 병렬 fetch 실행
        results = await asyncio.gather(*tasks)

        for trade_info in results:
            if trade_info is not None:
                trade_candidates.append(trade_info)

        # 동기화된 init_table_trade 실행
        for trade_info in trade_candidates:
            # init_table_trade_async 내부에서 lock 사용 중.
            await init_table_trade_async(self, trade_info)
            # async with self.lock_trade:
            #     init_table_trade(self, trade_info)

            if drop:
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()

                self.table_condition.drop([trade_info.df_res.symbol[0]], inplace=True)

                self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} drop rows  : %.4fs" % (time.time() - start_time))
                self.sys_log.debug("------------------------------------------------")
                
        
        # DB 저장
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        async with self.lock_trade:
            self.db_manager.replace_table(self.table_condition, self.table_condition_name, send=self.config.database.usage)

        self.sys_log.debug(f"LoopTableCondition : elasped time, replace_table condition (send={self.config.database.usage}) : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")

        self.sys_log.debug(f"[loop_table_condition] END   at {datetime.now().strftime('%H:%M:%S.%f')} (elapsed: {time.time() - loop_start:.2f}s)")
                
                
        # Set enough term for interval.
        # await asyncio.sleep(self.config.term.loop_table_condition)
        # await asyncio.sleep(max(0, self.config.term.loop_table_condition - (time.time() - loop_start)))
        await asyncio.sleep(1) 
     
    
async def get_trade_info_async_wrapper(self, symbol, base_interval, divider, account, days, debugging):
    """
    v0.1
        - adj. get_trade_info_async v0.2 20250403 2000.
    """
    
    trade_info_list = []

    try:
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        loop = asyncio.get_running_loop()
        df_res_15m = await loop.run_in_executor(executor, get_df_new, self, symbol, base_interval, days, None, 1500, 0.0)

        self.sys_log.debug(f"LoopTableCondition : elapsed time, {symbol} get_df_new : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")

        # TF 정의 및 리샘플
        tf_map = {
            '15m': ('15T', '9H'),
            '30m': ('30T', '9H'),
            '1h':  ('1H',  '9H'),
            '2h':  ('2H',  '9H'),
            '4h':  ('4H',  '9H'),
        }

        df_dict = {'15m': df_res_15m.tail(1200)}
        for tf, (rule, offset) in tf_map.items():
            if tf == '15m':
                continue
            df_dict[tf] = to_htf(df_res_15m, rule, offset).tail(1200)

        # 비동기 분석 함수
        async def analyze(interval, df):
             return await get_trade_info_async(self, df, symbol, interval, divider, account, debugging)

        # 분석 병렬 실행
        tasks = [analyze(interval, df) for interval, df in df_dict.items()]
        trade_info_list = await asyncio.gather(*tasks)

    except Exception as e:
        self.sys_log.error(f"Error in get_trade_info_wrapper for {symbol}: {e}")

    return trade_info_list

        
async def get_trade_info_async(self, idx, symbol, position, interval, divider, account, side_open, days, debugging):
    
    """
    v0.1
        - extract get_df_new ~ df_res 분석까지 비동기화 처리 20250321
        - add side_open & expiration 20250325 2137.
    v0.1.1
        - remove expiration. 20250328 2317.
    """
    
    # Init.
    position = None
    # expired_x = False
    
    try:
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        loop = asyncio.get_running_loop()
        # df_res = await loop.run_in_executor(None, get_df_new, self, symbol, interval, days, None, 1500, 0.0)
        df_res = await loop.run_in_executor(executor, get_df_new, self, symbol, interval, days, None, 1500, 0.0)        

        self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} get_df_new : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")

        # df_res 무결성 검증
        if df_res is None or type(df_res) is str or len(df_res) < abs(self.config.bank.complete_index):
            msg = f"Warning in df_res : {df_res}"
            self.sys_log.warning(msg)
            self.push_msg(msg)
        else:

            # Indicator 처리
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()

            df_res = get_DC(df_res, params.get('DC_period', 5))
            df_res = get_DCconsecutive(df_res)
            df_res = get_CCI(df_res, params.get('CCI_period', 21), params.get('CCIMA_period', 14))
            df_res = get_BB(df_res, period=params.get('BB_period', 20), multiple=params.get('BB_multiple', 2), level=params.get('BB_level', 2))
            df_res = get_CandleFeatures(df_res)
            df_res = get_SAR(df_res, start=params.get('SAR_start', 0.02), increment=params.get('SAR_increment', 0.02), maximum=params.get('SAR_maximum', 0.2))

            self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} get indicators : %.4fs" % (time.time() - start_time))
            self.sys_log.debug("------------------------------------------------")
            

            # Price Channel & Opener
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            df_res = get_PriceChannel(df_res, **params)
            df_res = get_Opener(df_res, **params)

            c_index = self.config.bank.complete_index
            opener_long = df_res['OpenerLong'].iloc[c_index]
            opener_short = df_res['OpenerShort'].iloc[c_index]

            self.sys_log.debug(f"LoopTableCondition : Opener status: OpenerLong={opener_long}, OpenerShort={opener_short}")            
            self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} get Price Channel & Opener : %.4fs" % (time.time() - start_time))
            self.sys_log.debug("------------------------------------------------")


            if opener_long or opener_short or debugging:

                # Signal 분석
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()

                df_res = get_Starter(df_res, **params)
                df_res = get_Anchor(df_res, **params)
                df_res = get_SR(df_res, **params)
                df_res = get_Momentum(df_res, **params)
                df_res = get_Direction(df_res, **params)
                df_res = get_Expiry(df_res, **params)

                df_res['long'] = df_res['OpenerLong'] & df_res['StarterLong'] & df_res['AnchorLong'] & df_res['SRLong'] & df_res['MomentumLong'] & df_res['DirectionLong']
                df_res['short'] = df_res['OpenerShort'] & df_res['StarterShort'] & df_res['AnchorShort'] & df_res['SRShort'] & df_res['MomentumShort'] & df_res['DirectionShort']

                self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} get signal : %.4fs" % (time.time() - start_time))
                self.sys_log.debug("------------------------------------------------")
                

                # 조건 만족 시 코드 생성 및 가격 박스 추출
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()

                timestamp_millis = datetime.now().strftime('%f')
                code = f"{symbol}_{interval}_{str(df_res.index[c_index])}_{timestamp_millis}"
                
                
                # 방향 결정                
                if df_res['long'].iloc[c_index]:
                    position = 'LONG'
                    side_col = 'long'
                elif df_res['short'].iloc[c_index]:
                    position = 'SHORT'
                    side_col = 'short'
                    
                    
                if position:
                    side_open = 'BUY' if position == 'LONG' else 'SELL'

                    price_expiration = df_res[f'price_expiry_{side_col}'].iloc[c_index]
                    price_take_profit = df_res[f'price_take_profit_{side_col}'].iloc[c_index]
                    price_entry = df_res[f'price_entry_{side_col}'].iloc[c_index]
                    price_stop_loss = df_res[f'price_stop_loss_{side_col}'].iloc[c_index]     
                    
                    self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} get priceBox : %.4fs" % (time.time() - start_time))
                    self.sys_log.debug("------------------------------------------------")
                    
                    
                    # # check historical expiry
                    #     # add to table_log.
                    # self.sys_log.debug("------------------------------------------------")
                    # start_time = time.time()
                    
                    # if side_open == 'BUY':
                    #     df_res_high = df_res['high'].iloc[self.config.bank.latest_index]
                    #     self.sys_log.debug(f"df_res_high : {df_res_high}, price_expiration : {price_expiration}")
                    #     if df_res_high >= price_expiration:
                    #         expired_x = True
                    # else:
                    #     df_res_low = df_res['low'].iloc[self.config.bank.latest_index]
                    #     self.sys_log.debug(f"df_res_low : {df_res_low}, price_expiration : {price_expiration}")
                    #     if df_res_low <= price_expiration:
                    #         expired_x = True 
                        
                    # self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} check historical expiry : %.4fs" % (time.time() - start_time))
                    # self.sys_log.debug("------------------------------------------------")
                    
                
            # save df_res. (df_rew 무결성 검증 이후)
                # interval 별로 저장
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()

            path_dir_df_res = "{}\\{}".format(self.path_dir_df_res, str(df_res.index[c_index]).split(' ')[-1].replace(":", ""))
            os.makedirs(path_dir_df_res, exist_ok=True)
            
            path_df_res = "{}\\{}_{}.csv".format(path_dir_df_res, symbol, interval)
            df_res.to_csv(path_df_res, index=True)  # leave datetime index

            self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} {interval} save df_res : %.4fs" % (time.time() - start_time))
            self.sys_log.debug("------------------------------------------------")
            
            
            if position:
                trade_info = TradeInfo(
                    symbol, side_open, position, interval, divider, account,
                    df_res, code,
                    price_expiration, price_take_profit, price_entry, price_stop_loss
                )

                return trade_info
            else:
                return None
            

    except Exception as e:
        self.sys_log.error(f"Error in get_trade_info_async for {symbol}: {e}")
        return None

        
def loop_table_condition(self, drop=False, debugging=False):
   
    """   
    v1.6.1
        - sleep for empty table.
    v1.7
        - adj, Anchor Concept. 20250314 1914.
        - add miliesec. to code (for prevent code duplication --> get_income_info affection) 20250318 1843.
        - adj. get_df_new v4.2.4 20250318 2020.
        - position 추적, balance > balance_orig error 수정 완료. 20250321 0540.
    v1.8.1
        - self. manual 수정. 20250321 0540.
        - add symbol to time log phase. 20250323 2105.
    """

    if self.table_condition.empty:
        time.sleep(1)
    else:    
        for idx, row in self.table_condition.iterrows(): # idx can be used, cause table_condition's index are not reset.

            # init.                    
            symbol = row.symbol # symbol extraction.
            side_open = row.side_open
            
            position = row.position
            interval = row.interval
            
            divider = row.divider
            account = row.account
            

            # do not repeat it in interval.
            interval_number = itv_to_number(interval)
            timestamp_current = int(time.time())
            if timestamp_current - int(row.timestampLastUsed) < interval_number * 60:
                continue
            else:
                self.sys_log.debug("------------------------------------------------")            
                start_time = time.time()
                
                timestamp_anchor = 1718236800 # 2024-06-13 9:00:00
                timestamp_remain = (timestamp_current - timestamp_anchor) % (interval_number * 60)
                timestamp_current -= timestamp_remain
                self.table_condition.at[idx, 'timestampLastUsed'] =  timestamp_current
                    
                self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} check timestampLastUsed : %.2fs".format(time.time() - start_time))
                self.sys_log.debug("------------------------------------------------")
            


            # 01 get df_res
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            self.sys_log.debug(f"idx: {idx}")        
            
            assert interval_number < 1440, "assert, inteval < 1d"
            days = math.ceil((1200 / (60 * 24 / interval_number))) # 1200 : min len for indicators to be synced.
            self.sys_log.debug(f"days: {days}")
    
            df_res = get_df_new(self, 
                                    symbol=symbol,
                                    interval=interval, 
                                    days=days, 
                                    limit=1500, 
                                    timesleep=0.0) # changed to 0.0
                    
            self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} get_df_new : %.4fs" % (time.time() - start_time))
            self.sys_log.debug("------------------------------------------------")
    
    
    
            # df_res 무결성 검증.
                # 폐지된 symbol 은 drop.
            if df_res is None or type(df_res) is str or len(df_res) < abs(self.config.bank.complete_index):
                msg = f"Warning in df_res : {df_res}"
                self.sys_log.warning(msg)
                self.push_msg(msg)
                
                if type(df_res) == str:
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()   
                    
                    self.table_condition.drop([idx], inplace=True)    
                    
                    self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} drop invalid idx {idx}: %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
                continue
            else:
                self.sys_log.debug("LoopTableCondition : df_res validation (df_res.index[-1] = {})".format(df_res.index[-1]))                           
            
            
          
            # 02 Get indicators.
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            # df_res = get_EMAs(df_res)
            df_res = get_DC(df_res, params.get('DC_period', 5))
            df_res = get_DCconsecutive(df_res)
            # df_res = get_DC2(df_res, params.get('DC2_period', 60))
            df_res = get_CCI(df_res, params.get('CCI_period', 20), params.get('CCIMA_period', 14))
            df_res = get_BB(df_res, period=params.get('BB_period', 20), multiple=params.get('BB_multiple', 2), level=params.get('BB_level', 2))
            df_res = get_CandleFeatures(df_res)
            df_res = get_SAR(df_res, start=params.get('SAR_start', 0.02), increment=params.get('SAR_increment', 0.02), maximum=params.get('SAR_maximum', 0.2))
            
            self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} get indicators : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
          
          
            # 03 Get Price Channel & Openers.
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            df_res = get_PriceChannel(df_res, **params)
            df_res = get_Opener(df_res, **params)
            
            self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} get Opener : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
                      
            # 04 Get Signal.      
                # opener 가 있을 경우만 진행                 
            c_index = self.config.bank.complete_index
            opener_long = df_res['OpenerLong'].iloc[c_index]
            opener_short = df_res['OpenerShort'].iloc[c_index]
            
            self.sys_log.debug(f"LoopTableCondition : Opener status: OpenerLong={opener_long}, OpenerShort={opener_short}")


            if any([opener_long, opener_short]): # df_res saving (rechecking) 때문에 phase 들여쓰기 사용한다.
            
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
                df_res = get_Starter(df_res, **params)
                df_res = get_Anchor(df_res, **params)
                df_res = get_SR(df_res, **params)
                df_res = get_Momentum(df_res, **params)
                df_res = get_Direction(df_res, **params)
                df_res = get_Expiry(df_res, **params)

                # 롱/숏 포지션 결정
                df_res['long'] = df_res['OpenerLong'] & df_res['StarterLong'] & df_res['AnchorLong'] & df_res['SRLong'] & df_res['MomentumLong'] & df_res['DirectionLong']
                df_res['short'] = df_res['OpenerShort'] & df_res['StarterShort'] & df_res['AnchorShort'] & df_res['SRShort'] & df_res['MomentumShort'] & df_res['DirectionShort']
                            
                self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} get Long / Short signal : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")            
                    
                    
                                
                # Set code & side_open
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()            
                
                timestamp_millis = datetime.now().strftime('%f') # 밀리초 (0~999,999)
                code = f"{symbol}_{interval}_{str(df_res.index[self.config.bank.complete_index])}_{timestamp_millis}"
                side_col = 'long' if position == 'LONG' else 'short'

                if df_res[side_col].iloc[self.config.bank.complete_index] or debugging:
                    side_open = 'BUY' if position == 'LONG' else 'SELL'
                    
                    msg = f"{code} {side_open}"
                    self.push_msg(msg)

                self.sys_log.debug(f"LoopTableCondition : elapsed time, set code & side_open : {time.time() - start_time:.4f}s")
                self.sys_log.debug("------------------------------------------------")

                
                
                if not pd.isnull(side_open): # default = np.nan
                    self.sys_log.debug(f"side_open : {side_open}")                                
                                    
                    
                    # 01 Set Price Channel                
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()

                        # 'BUY' 또는 'SELL'에 따라 suffix 결정 ('long' 또는 'short')
                    suffix = "long" if side_open == "BUY" else "short"

                        # 동적 컬럼 접근
                    price_expiration = df_res[f'price_expiry_{suffix}'].to_numpy()[self.config.bank.complete_index]
                    price_take_profit = df_res[f'price_take_profit_{suffix}'].to_numpy()[self.config.bank.complete_index]
                    price_entry = df_res[f'price_entry_{suffix}'].to_numpy()[self.config.bank.complete_index]
                    price_stop_loss = df_res[f'price_stop_loss_{suffix}'].to_numpy()[self.config.bank.complete_index]

                    self.sys_log.debug(f"LoopTableCondition : elapsed time, get_priceBox, get_price_arr : {time.time() - start_time:.4f}s")
                    self.sys_log.debug("------------------------------------------------")
                                        
                    
                    # 02 init_table_trade  
                    trade_info = TradeInfo(
                        symbol, side_open, position, interval, divider, account,
                        df_res, code, 
                        price_expiration, price_take_profit, price_entry, price_stop_loss
                    )                          
                    init_table_trade(self, trade_info)
                    
                    if drop: # drop : condition check 을 거친 symbol 을 table_condition loop 에서 제외시킨다.
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()   
                        
                        self.table_condition.drop([idx], inplace=True)    
                        
                        self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} drop rows  : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("-----------------------------------------------")  

                    
                    
            # save df_res.
                # to use not-updated df_res in LoopTableTrade's REPLACEMENT phase.
                # check df_res integrity.
                # 무슨말인지 하나도 모르겠네.
                    # 전처리된 df_res 내용을 사용자가 recheck 하기 만든 phase 로 추정한다.
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()            
                
            path_df_res = "{}\\{}.csv".format(self.path_dir_df_res, symbol)
            df_res.to_csv(path_df_res, index=True) # leave datetime index.
            
            self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} save df_res : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")

            
            
            # 저장시간이 다소 소요될 수 있어, 더 앞의 인덴트로 옮기지 못한다.
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()   
            
            self.db_manager.replace_table(self.table_condition, self.table_condition_name)
            
            self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} replace_table condition : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
            
            # Prevent API flood.
            time.sleep(self.config.term.loop_table_condition)



        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()   
        
        self.db_manager.replace_table(self.table_condition, self.table_condition_name, send=self.config.database.usage)
        
        self.sys_log.debug(f"LoopTableCondition : elasped time, {symbol} replace_table condition (send={self.config.database.usage}) : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        

async def init_table_trade_async(self, trade_info: TradeInfo):
    
    """    
    v2.4
        - adj, Anchor Concept. 20250314 2146.
            price_expiration excluded.
        - position 추적, balance > balance_orig error 수정 완료. 20250321 0540.
    v2.5.1
        - self. manual 수정. 20250321 0540.
        - add code to time log phase. 20250323 2101.
        - add def_type to calc_with_precision 20250325 2136.
        - modify side_open --> position in get precision. 20250326 2028.
    v2.5.2
        - add historical high / low. 20250328 2304.
        - modify path_df_res .ftr --> csv. 20250330 0939.
    v2.5.3
        - reconstruct code phase.
            init table_trade_new_row above to nan. 20250401 2150.
                init all cases.
    """
    
    
    loop_start = time.time()
    self.sys_log.debug(f"[init_table_trade] START at {datetime.now().strftime('%H:%M:%S.%f')}")
           
    ##################################################
    # INIT
    ##################################################
    
    symbol = trade_info.symbol
    side_open = trade_info.side_open
    position = trade_info.position
    interval = trade_info.interval
    divider = trade_info.divider
    account = trade_info.account
    
    df_res = trade_info.df_res
    code = trade_info.code
    
    price_expiration = trade_info.price_expiration
    price_take_profit = trade_info.price_take_profit
    price_entry = trade_info.price_entry
    price_stop_loss = trade_info.price_stop_loss
 
    order_motion = 0  # for consuming time without trading.
    amount_min = 5
    
    
    ##################################################
    # Get side.
    ##################################################
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
        
    side_close, \
    side_position = get_side_info(self, 
                                    side_open)
        
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_side_info : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
     
    
    ##################################################
    # Pricing
    ##################################################
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
    
    price_entry = get_price_entry(self, 
                                df_res,
                                side_open,
                                price_entry)
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_price_entry : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
        

    # get price_liquidation
        # get_price_liquidation requires leverage.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   

    # we are using target_loss_pct now, so liquidation has no more meaning.
        # target_loss_pct max is 100.
    price_liquidation = price_stop_loss
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_price_liquidation : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    # # get price_expiration
    # self.sys_log.debug("------------------------------------------------")
    # start_time = time.time()  
    
    # price_expiration = price_take_profit # temporarily fix to price_take_profit.
    
    # self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_price_expiration : %.4fs" % (time.time() - start_time)) 
    # self.sys_log.debug("------------------------------------------------")

    
    # adj. precision_price & quantity
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()    
        
    precision_price, \
    precision_quantity = get_precision(self, 
                                        symbol)
    
    def_type = 'floor' if position == 'LONG' else 'ceil'
    
    price_take_profit, \
    price_entry, \
    price_stop_loss, \
    price_liquidation, \
    price_expiration = [self.calc_with_precision(price_, precision_price, def_type) for price_ in [price_take_profit, 
                                                                                        price_entry, 
                                                                                        price_stop_loss,
                                                                                        price_liquidation,
                                                                                        price_expiration]]  # add price_expiration.
            
    self.sys_log.info("price_take_profit : {}".format(price_take_profit))
    self.sys_log.info("price_entry : {}".format(price_entry))
    self.sys_log.info("price_stop_loss : {}".format(price_stop_loss))
    self.sys_log.info("price_liquidation : {}".format(price_liquidation))
    self.sys_log.info("price_expiration : {}".format(price_expiration))
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} adj. precision_price : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    ##################################################
    # balance.
    ##################################################
    
    # get balance
        # Stock 특성상, balance_min = price_entry 로 설정함.
        # open_data_dict.pop 은 watch_dict 의 일회성 / 영구성과 무관하다.    
            # balance : 주문을 넣으면 감소하는 금액
            # balance_origin : 주문을 넣어도 감소하지 않는 금액
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    account_normality, \
    balance_, \
    balance_origin = get_account_normality(self, 
                                        account)
    
    self.sys_log.info("account_normality : {}".format(account_normality))
    self.sys_log.info("balance_ : {}".format(balance_))
    self.sys_log.info("balance_origin : {}".format(balance_origin))
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_account_normality : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
    
    
    
    ##################################################
    # add to TableTrade
    ##################################################
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()  
    
    table_trade_new_row = pd.DataFrame(
                        data = np.array([np.nan] * len(self.table_trade.columns)).reshape(-1, 1).T,
                        columns=self.table_trade.columns,
                        dtype=object)

    # # save df_res.
    path_df_res_open = "{}/{}.csv".format(self.path_dir_df_res, code)
    # df_res.reset_index(drop=True).to_feather(self.path_df_res , compression='lz4')

    # init table_trade.
    table_trade_new_row.symbol = symbol 
    table_trade_new_row.code = code 
    table_trade_new_row.path_df_res_open = path_df_res_open 
    table_trade_new_row.order_motion = order_motion 
    
    table_trade_new_row.side_open = side_open 
    table_trade_new_row.side_close = side_close 
    table_trade_new_row.side_position = side_position 
    
    table_trade_new_row.precision_price = precision_price 
    table_trade_new_row.precision_quantity = precision_quantity 
    table_trade_new_row.price_take_profit = price_take_profit 
    table_trade_new_row.price_entry = price_entry 
    table_trade_new_row.price_stop_loss = price_stop_loss 
    table_trade_new_row.price_liquidation = price_liquidation 
    table_trade_new_row.price_expiration = price_expiration 
    
    table_trade_new_row.historical_high =  df_res['high'].iloc[self.config.bank.latest_index]   # added.  
    table_trade_new_row.historical_low =  df_res['low'].iloc[self.config.bank.latest_index]     # added.  
    
    table_trade_new_row.account = account
    table_trade_new_row.balance = balance_                 # added.  
    table_trade_new_row.balance_origin = balance_origin    # added.  
    
    
    if account_normality:  
                
    
        ##################################################
        # quantity.
        ##################################################
        
        # get quantity_open   
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()  
        
        target_loss = balance_origin / divider # divider's target is balance_origin.
            
        price_diff = abs(price_entry - price_stop_loss)
        # fee_entry = price_entry * self.config.broker.fee_market
        fee_entry = price_entry * self.config.broker.fee_limit # limit_entry
        fee_stop_loss = price_stop_loss * self.config.broker.fee_market        
        loss = price_diff + fee_entry + fee_stop_loss
        
        quantity_open = target_loss / loss
        quantity_open = self.calc_with_precision(quantity_open, precision_quantity)
        
        # Calculate entry and exit amounts
            # amount_entry cannot be larger than the target_loss.
        amount_entry = price_entry * quantity_open
        
        self.sys_log.info("target_loss : {}".format(target_loss))
        self.sys_log.info("loss : {}".format(loss))
        self.sys_log.info("quantity_open : {}".format(quantity_open))
        self.sys_log.info("amount_entry : {}".format(amount_entry))
            
        table_trade_new_row.target_loss = target_loss          # added.  
        table_trade_new_row.quantity_open = quantity_open 
        table_trade_new_row.amount_entry = amount_entry
        
        self.sys_log.debug(f"InitTableTrade : elasped time, {code} get margin : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        
        
        # func3 : check margin normality (> min balance)
        if amount_entry > amount_min:     
            
            # get leverage
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
                
            leverage_limit_user, \
            leverage_limit_server, \
            leverage = get_leverage_limit(self, 
                                            symbol, 
                                            amount_entry, 
                                            loss, 
                                            price_entry,
                                            )
            
            margin = amount_entry / leverage
            
            self.sys_log.info("leverage : {}".format(leverage))
            self.sys_log.info("margin : {}".format(margin))
            
            table_trade_new_row.leverage = leverage
            table_trade_new_row.margin = margin
            
            self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_leverage : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
            # set leverage
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
                                
            self.set_leverage(symbol, leverage)
                
            self.sys_log.debug(f"InitTableTrade : elasped time, {code} set_leverage : %.4fs" % (time.time() - start_time))
            self.sys_log.debug("------------------------------------------------")
            
            
            # func4 : subtract margin if normal on 'balance'.
            self.table_account.loc[self.table_account['account'] == account, 'balance'] -= margin
            
        else:
            table_trade_new_row.status = 'Error : amount_entry <= amount_min'    # added.
            table_trade_new_row.remove_row = 1
                    
    else:
        table_trade_new_row.status = 'Error : account_normality'    # added.
        table_trade_new_row.remove_row = 1
        
        
    # append row.
    async with self.lock_trade:
        self.table_trade = pd.concat([self.table_trade, table_trade_new_row]).reset_index(drop=True)            
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} append row to TableTrade. : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
    
    self.push_msg("{} {} row insert into TableTrade.".format(code, account))
        
        
    # remove. table_trade_async 에서 진행.
    # self.sys_log.debug("------------------------------------------------")
    # start_time = time.time()
    
    # self.db_manager.replace_table(self.table_account, self.table_account_name, send=self.config.database.usage)
    # self.db_manager.replace_table(self.table_trade, self.table_trade_name, send=self.config.database.usage, mode=['UPDATE', 'DELETE'])
    
    # self.sys_log.debug(f"InitTableTrade : elasped time, {code} replace_table (send={self.config.database.usage})  : %.4fs" % (time.time() - start_time)) 
    # self.sys_log.debug("------------------------------------------------")
    
    self.sys_log.debug(f"[init_table_trade] END   at {datetime.now().strftime('%H:%M:%S.%f')} (elapsed: {time.time() - loop_start:.2f}s)")
  

def init_table_trade(self, trade_info: TradeInfo):

    """    
    v2.4
        - adj, Anchor Concept. 20250314 2146.
            price_expiration excluded.
        - position 추적, balance > balance_orig error 수정 완료. 20250321 0540.
    v2.5.1
        - self. manual 수정. 20250321 0540.
        - add code to time log phase. 20250323 2101.
        - add def_type to calc_with_precision 20250325 2136.
    """
           
    ##################################################
    # INIT
    ##################################################
    
    symbol = trade_info.symbol
    side_open = trade_info.side_open
    position = trade_info.position
    interval = trade_info.interval
    divider = trade_info.divider
    account = trade_info.account
    
    df_res = trade_info.df_res
    code = trade_info.code
    
    price_expiration = trade_info.price_expiration
    price_take_profit = trade_info.price_take_profit
    price_entry = trade_info.price_entry
    price_stop_loss = trade_info.price_stop_loss
 
    order_motion = 0  # for consuming time without trading.
    amount_min = 5
    
    
    ##################################################
    # Get side.
    ##################################################
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
        
    side_close, \
    side_position = get_side_info(self, 
                                    side_open)
        
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_side_info : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
     
    
    ##################################################
    # Pricing
    ##################################################
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
    
    price_entry = get_price_entry(self, 
                                df_res,
                                side_open,
                                price_entry)
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_price_entry : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
        

    # get price_liquidation
        # get_price_liquidation requires leverage.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   

    # we are using target_loss_pct now, so liquidation has no more meaning.
        # target_loss_pct max is 100.
    price_liquidation = price_stop_loss
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_price_liquidation : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    # # get price_expiration
    # self.sys_log.debug("------------------------------------------------")
    # start_time = time.time()  
    
    # price_expiration = price_take_profit # temporarily fix to price_take_profit.
    
    # self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_price_expiration : %.4fs" % (time.time() - start_time)) 
    # self.sys_log.debug("------------------------------------------------")

    
    # adj. precision_price & quantity
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()    
        
    precision_price, \
    precision_quantity = get_precision(self, 
                                        symbol)
    
    def_type = 'floor' if side_open == 'BUY' else 'ceil'
    
    price_take_profit, \
    price_entry, \
    price_stop_loss, \
    price_liquidation, \
    price_expiration = [self.calc_with_precision(price_, precision_price, def_type) for price_ in [price_take_profit, 
                                                                                        price_entry, 
                                                                                        price_stop_loss,
                                                                                        price_liquidation,
                                                                                        price_expiration]]  # add price_expiration.
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} adj. precision_price : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    ##################################################
    # balance.
    ##################################################
    
    # get balance
        # Stock 특성상, balance_min = price_entry 로 설정함.
        # open_data_dict.pop 은 watch_dict 의 일회성 / 영구성과 무관하다.    
            # balance : 주문을 넣으면 감소하는 금액
            # balance_origin : 주문을 넣어도 감소하지 않는 금액
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    account_normality, \
    balance_, \
    balance_origin = get_account_normality(self, 
                                        account)
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_account_normality : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
    
    
    if account_normality:  
                
    
        ##################################################
        # quantity.
        ##################################################
        
        # get quantity_open   
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()  
        
        target_loss = balance_origin / divider # divider's target is balance_origin.
            
        price_diff = abs(price_entry - price_stop_loss)
        # fee_entry = price_entry * self.config.broker.fee_market
        fee_entry = price_entry * self.config.broker.fee_limit # limit_entry
        fee_stop_loss = price_stop_loss * self.config.broker.fee_market        
        loss = price_diff + fee_entry + fee_stop_loss
        
        quantity_open = target_loss / loss
        quantity_open = self.calc_with_precision(quantity_open, precision_quantity)
        
        # Calculate entry and exit amounts
            # amount_entry cannot be larger than the target_loss.
        amount_entry = price_entry * quantity_open
        
        self.sys_log.debug(f"InitTableTrade : elasped time, {code} get margin : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        
        
        # func3 : check margin normality (> min balance)
        if amount_entry > amount_min:     
            
            # get leverage
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
                
            leverage_limit_user, \
            leverage_limit_server, \
            leverage = get_leverage_limit(self, 
                                            symbol, 
                                            amount_entry, 
                                            loss, 
                                            price_entry,
                                            )
            
            margin = amount_entry / leverage
            
            self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_leverage : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
            # set leverage
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
                                
            self.set_leverage(symbol, leverage)
                
            self.sys_log.debug(f"InitTableTrade : elasped time, {code} set_leverage : %.4fs" % (time.time() - start_time))
            self.sys_log.debug("------------------------------------------------")
            
            
            # func4 : subtract margin if normal on 'balance'.
            self.table_account.loc[self.table_account['account'] == account, 'balance'] -= margin
            
        
            ##################################################
            # output bill.
            ##################################################
            
            self.sys_log.info("price_take_profit : {}".format(price_take_profit))
            self.sys_log.info("price_entry : {}".format(price_entry))
            self.sys_log.info("price_stop_loss : {}".format(price_stop_loss))
            self.sys_log.info("price_liquidation : {}".format(price_liquidation))
            self.sys_log.info("price_expiration : {}".format(price_expiration))
            
            self.sys_log.info("balance : {}".format(balance_))
            self.sys_log.info("balance_origin : {}".format(balance_origin))
            
            self.sys_log.info("target_loss : {}".format(target_loss))
            self.sys_log.info("loss : {}".format(loss))
            self.sys_log.info("quantity_open : {}".format(quantity_open))
            self.sys_log.info("amount_entry : {}".format(amount_entry))
            
            self.sys_log.info("leverage : {}".format(leverage))
            self.sys_log.info("margin : {}".format(margin))
            
        
        
            ##################################################
            # add to TableTrade
            ##################################################
            
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()  
            
            table_trade_new_row = pd.DataFrame(data = np.array([np.nan] * len(self.table_trade.columns)).reshape(-1, 1).T, columns=self.table_trade.columns, dtype=object)
        
            # # save df_res.
            path_df_res_open = "{}/{}.ftr".format(self.path_dir_df_res, code)
            # df_res.reset_index(drop=True).to_feather(self.path_df_res , compression='lz4')
        
            # init table_trade.
            table_trade_new_row.symbol = symbol 
            table_trade_new_row.code = code 
            table_trade_new_row.path_df_res_open = path_df_res_open 
            table_trade_new_row.order_motion = order_motion 
            
            table_trade_new_row.side_open = side_open 
            table_trade_new_row.side_close = side_close 
            table_trade_new_row.side_position = side_position 
            
            table_trade_new_row.precision_price = precision_price 
            table_trade_new_row.precision_quantity = precision_quantity 
            table_trade_new_row.price_take_profit = price_take_profit 
            table_trade_new_row.price_entry = price_entry 
            table_trade_new_row.price_stop_loss = price_stop_loss 
            table_trade_new_row.price_liquidation = price_liquidation 
            table_trade_new_row.price_expiration = price_expiration 
            
            table_trade_new_row.account = account
            table_trade_new_row.balance = balance_                 # added.  
            table_trade_new_row.balance_origin = balance_origin    # added.  
            
            table_trade_new_row.target_loss = target_loss          # added.  
            table_trade_new_row.quantity_open = quantity_open 
            table_trade_new_row.amount_entry = amount_entry
            
            table_trade_new_row.leverage = leverage
            table_trade_new_row.margin = margin
        
            # self.sys_log.info("table_trade_new_row : \n{}".format(table_trade_new_row.iloc[0]))
        
            # append row.
            self.table_trade = pd.concat([self.table_trade, table_trade_new_row]).reset_index(drop=True)            
            
            self.sys_log.debug(f"InitTableTrade : elasped time, {code} append row to TableTrade. : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            self.push_msg("{} {} row insert into TableTrade.".format(code, account))
        
        
    # preventing losing table_account info.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    self.db_manager.replace_table(self.table_account, self.table_account_name, send=self.config.database.usage)
    self.db_manager.replace_table(self.table_trade, self.table_trade_name, send=self.config.database.usage, mode=['UPDATE', 'DELETE'])
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} replace_table (send={self.config.database.usage})  : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
     

async def loop_table_trade_async(self):
    
    """
    v2.1
        - v1.3.1's asyncio version. 20250323 2125.
    v2.1.1
        - add historical expiry. 20250328 2307.
    v2.1.2  
        - replace .at indexing, use .loc with code unique, preventing async data interruption. 20250330 2216.
        - add table_loggigng to remove_row = 1 phase.
    v2.1.3
        - reconstruct code phase for status 'Error'. 20250401 2215.
    """
    
    
    while True:
        
        loop_start = time.time()
        self.sys_log.debug(f"[loop_table_trade]     START at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        if self.table_trade.empty:
            await asyncio.sleep(1)
            continue
        
        new_rows = [] # exist for stop_market (following the orderInfo)
        
        for idx, row in self.table_trade.iterrows(): # for save iteration, use copy().
            
            # Init.      
            order_info = None
            
            # below vars. should not be (re) declared in bank method.
            expired = 0 # 0 / 1 / 2 : No / x / y
            stop_loss_on = False
            closer_on = False

            # update & clean table.
            # some data need to be transferred to Bank's instance, some don't.
            # 'row' means un-updated information. (by get_order_info)
            code = row.code
            symbol = row.symbol  # we have symbol column in table already.
            orderId = row.orderId  # type assertion.
            status_prev = row.status  # save before updating status.
            order_motion = row.order_motion

            side_open = row.side_open
            price_expiration = row.price_expiration
            price_stop_loss = row.price_stop_loss
            price_liquidation = row.price_liquidation
            
            historical_high =  row.historical_high
            historical_low =  row.historical_low

            side_close = row.side_close
            side_position = row.side_position
            
            leverage = row.leverage
            margin = row.margin
            account = row.account
                            
            status_error = 1 if 'ERROR' in str(status_prev) else 0
            
            self.table_trade.loc[self.table_trade['code'] == code, 'order'] = 0 # init as 0.

            self.sys_log.debug("row : \n{}".format(row))
            self.sys_log.debug("row.dtypes : {}".format(row.dtypes))
            
            
            try:
                # 코드 형식이 {symbol}_{interval}_{entry_timeIndex}라고 가정합니다.
                _, interval, entry_timeIndex, _ = code.split('_')
            except Exception as e:
                msg = f"Error parsing code {code}: {e}"
                self.sys_log.error(msg)
                self.push_msg(msg)                
                continue # hard assertion.
            
            
            #############################################
            # Set Broker-Ticker option.
                # orderId 가 있더라고, price_market 가 없는 경우 가능 (프로그램 재시작 등.)
            #############################################

            if symbol not in self.price_market.keys() or pd.isnull(orderId): # initial (do only once.)
                if not status_error:
                
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    self.websocket_client.agg_trade(symbol=symbol, id=1, callback=self.agg_trade_message_handler)
                    self.set_position_mode(dualSidePosition='true')
                    self.set_margin_type(symbol=symbol)
                    
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set Broker-Ticker option : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
            
        
            
            #############################################
            # set order, order_way (OPEN) (order not exists)
            #############################################
            
            if pd.isnull(orderId):
                if not status_error:
                
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    self.table_trade.loc[self.table_trade['code'] == code, 'order'] = 1
                    self.table_trade.loc[self.table_trade['code'] == code, 'order_way'] = 'OPEN'
                    
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set order, order_way (OPEN) : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")

            #  orderId exists.
            else:
                # get_order_info. (update order_info)
                    # get_order_info should have valid symbol & orderId.                      
                order_info = get_order_info(self, 
                                        symbol,
                                        orderId)
                
                if order_info is not None:
                    # order 이 있는 경우에만 진행하는 것들
                        # 1 update table_trade 
                        # 2 update status + logging.
                        # 3 update remove_row
                        # 4 update order, order_way
                        # 5 check Expiry / Stop Loss / Closer (OPEN / CLOSE order 있는 경우에만 진행하는 것들)
                    

                    #############################################
                    # update table_trade
                    #############################################
                    
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    for k, v in order_info.items():                        
                        self.table_trade.loc[self.table_trade['code'] == code, k] = v
                        
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} update table_trade : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
                    # display(self.table_trade) 
                       
                    
                    
                    #############################################
                    # check status
                    #############################################
                    
                    self.sys_log.debug(f"status_prev : {status_prev}, order_info['status'] : {order_info['status']}")
                    if status_prev != order_info['status']:                    
                    
                        # set statusChangeTime 
                            # check if status has been changed.
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_trade.loc[self.table_trade['code'] == code, 'statusChangeTime'] = datetime.now().strftime('%Y%m%d%H%M%S%f')
                        self.push_msg("{} status has been changed. {} {}".format(code, row.order_way, order_info['status']))
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set statusChangeTime : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")      
                        # display(self.table_trade)
                            
                            
                        
                        # logging : transfer rows to Table - log.   
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_log = pd.concat([self.table_log, self.table_trade[self.table_trade['code'] == code]], ignore_index=True)
                        self.table_log['id'] = self.table_log.index + 1 # Update 'id' column to be the new index + 1
                        self.db_manager.replace_table(self.table_log, self.table_log_name)
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} replace_table table_log : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")  
                        # display(self.table_trade)
                        
                        
                    
                    #############################################
                    # set remove_row. 
                        # regardless status updated. just check realtime status.
                    #############################################  
                    
                    if (order_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']) or (row.order_way == 'CLOSE' and order_info['status'] == 'FILLED'):
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'] = 1
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set remove_row : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                        

                    
                    #############################################
                    # set order, order_way (for CLOSE)
                    #############################################
                    
                    if row.order_way == 'OPEN':         
                        if (abs(float(order_info['executedQty']) / float(order_info['origQty'])) >= self.config.bank.quantity_open_exec_ratio) or order_info['status'] == 'FILLED':           
                            
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()
                            
                            self.table_trade.loc[self.table_trade['code'] == code, 'order'] = 1
                            self.table_trade.loc[self.table_trade['code'] == code, 'order_way'] = 'CLOSE'
                            
                            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set order, order_way (CLOSE) : %.4fs" % (time.time() - start_time)) 
                            self.sys_log.debug("------------------------------------------------")
                            # display(self.table_trade)
                    
                    

                    #############################################
                    # Expiry / Stop Loss / Closer
                    #############################################
                    
                    price_realtime = get_price_realtime(self, symbol)
                    self.sys_log.debug(f"symbol : {symbol}")
                    self.sys_log.debug(f"price_realtime : {price_realtime}")

                    if not pd.isnull(price_realtime):
                        
                        price_realtime_prev = self.table_trade.loc[self.table_trade['code'] == code, 'price_realtime'].values[0]
                        
                        slippage_margin = 0.1 # 10 %
                        lower_bound = price_realtime_prev * (1 - slippage_margin)
                        upper_bound = price_realtime_prev * (1 + slippage_margin)                                        
                        self.sys_log.debug(f"price_realtime_prev: {price_realtime_prev:.6f}, lower_bound: {lower_bound:.6f}, upper_bound: {upper_bound:.6f}")
                                                                                    
                        
                        # price_realtime 정상 범위
                        if pd.isnull(price_realtime_prev) or (lower_bound <= price_realtime <= upper_bound):  
                            
                            self.table_trade.loc[self.table_trade['code'] == code, 'price_realtime'] = price_realtime                    
                    
                            # 01 Check Expiry
                            if row.order_way == 'OPEN' and order_info['status'] in ['NEW', 'PARTIALLY_FILLED']:
                                self.sys_log.debug("------------------------------------------------")
                                start_time = time.time()  
                                
                                if side_position == 'LONG':
                                    price_historical = historical_high
                                else:
                                    price_historical = historical_low      
                                
                                expired = check_expiration(self,
                                                        interval,
                                                        entry_timeIndex,
                                                        side_position,
                                                        price_realtime,
                                                        price_historical,
                                                        price_expiration,
                                                        ncandle_game=params.get('ncandle_game', 2))
                                    
                                if expired:
                                    
                                    # we need this, cause order should be canceled !                                            
                                    quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                                symbol,
                                                                                orderId)  
                                    
                                    # 어차피, get_income_info phase 에서 margin 내부에 (executed & unexecuted Margin 이 있다.)
                                    # self.table_account.loc[self.table_account['account'] == account, 'balance'] +=(quantity_unexecuted * row.price_entry / leverage)
                                    
                                    # Todo, messeage alertion needed ?
                                    self.push_msg("{} has been expired.".format(code))
                                    # self.user_text = 'watch'  # allowing message.
                                    
                                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check expiration for order_open : %.4fs" % (time.time() - start_time)) 
                                self.sys_log.debug("------------------------------------------------")
            
                        
                        
                            # 02 Check Stop Loss
                                # get stop_loss_on
                                # order_way == 'CLOSE' : Stop Loss or Liquidation
                            elif row.order_way == 'CLOSE':
                                self.sys_log.debug("------------------------------------------------")
                                start_time = time.time()
                                
                                if params.get('use_stop_loss', True) or params.get('use_stop_loss_barclose', False):                                
                                    stop_loss_on = check_stop_loss(self,
                                                                        side_open,
                                                                        price_realtime,
                                                                        price_liquidation,
                                                                        price_stop_loss)
                                
                                # if params.get('use_stop_loss_barclose', False) and stop_loss_on:
                                #     wait_barClosed(self, interval, entry_timeIndex)                                
                                
                                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check_stop_loss : %.4fs" % (time.time() - start_time)) 
                                self.sys_log.debug("------------------------------------------------")
                                
                               
                                
                    # 03 Check Closer
                        # Closer 시그널 발생 시, 해당 interval의 종료 시각까지 대기 후 Market 주문으로 포지션 종료
                    if row.order_way == 'CLOSE':
                        
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()  
                        
                        closer_on = check_closer(self, 
                                                    interval, 
                                                    entry_timeIndex,
                                                    ncandle_game=params.get('ncandle_game', 2))
                        
                        # if params.get('use_closer', False) and closer_on:
                        #     wait_barClosed(self, interval, entry_timeIndex)
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check_stop_loss : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                
                
                
            # ORDER 조건 확인
            # if not self.config.bank.backtrade and not order_motion:
            if self.table_trade.loc[self.table_trade['code'] == code, 'order'].values[0] == 1:

                ####################################
                # 전처리 : 가격 & 거래량 수집.                      
                ####################################   
                
                    # public set.
                order_type = 'LIMIT' # fixed for order_limit.
                side_position = side_position                
                                
                    # OPEN condition
                if self.table_trade.loc[self.table_trade['code'] == code, 'order_way'].values[0] == 'OPEN':
                                    
                    side_order = row.side_open
                    price = row.price_entry
                    quantity = row.quantity_open
    
                    # CLOSE condition
                else: #  self.table_trade.at[idx, 'order_way'] == 'CLOSE': # upper phase's condition use 'order' = 1                                
                    # we don't do partial like this anymore.
                        # each price info will be adjusted order by order.
                                        
                    side_order = row.side_close
                    price = row.price_take_profit
                    quantity = order_info['executedQty'] # OPEN's qty.      
            
            
                
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()                               
                                
                # 01 LIMIT
                order_result, \
                error_code = order_limit(self, 
                                        symbol,
                                        side_order, 
                                        side_position, 
                                        price, 
                                        quantity)
            
                # normal state : error_code = 0
                if not error_code:
                    self.table_trade.loc[self.table_trade['code'] == code, 'orderId'] = order_result['orderId']
                    
                else:
                    # CLOSE order_error should be cared more strictly.                        
                    self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'] = 1
                    self.table_trade.loc[self.table_trade['code'] == code, 'status'] =f"ERROR : order_limit {error_code}"
                    
                    # remove_row == 1로 실제로 거래가 종료되어 드롭되기 직전에 로깅                                        
                    self.table_log = pd.concat([self.table_log, self.table_trade[self.table_trade['code'] == code]], ignore_index=True)
                    self.table_log['id'] = self.table_log.index + 1 # Update 'id' column to be the new index + 1
                    self.db_manager.replace_table(self.table_log, self.table_log_name)
                    
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} replace_table table_log : %.4fs" % (time.time() - start_time)) 
                    
                    
                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} order_limit : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")
                
                
                
                # 02 STOP MARKET
                # if self.table_trade.at[idx, 'order_way'] == 'CLOSE':                    
                    
                #     self.sys_log.debug("------------------------------------------------")
                #     start_time = time.time()
                                
                #     order_result, \
                #     error_code = order_stop_market(self, 
                #                                 symbol,
                #                                 side_order,
                #                                 side_position,
                #                                 row.price_stop_loss,
                #                                 quantity)
                    
                #     # normal state : error_code = 0
                #     if not error_code:                        
                #         new_row = self.table_trade.loc[idx].copy()
                #         new_row['id'] = np.nan # set as a missing_id for "fill_missing_ids"
                #         new_row['orderId'] = order_result['orderId'] # this is key for updating order_info (change to CLOSE orderId)
                #         new_rows.append(new_row)
                                                
                #     self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} order_market : %.4fs" % (time.time() - start_time)) 
                #     self.sys_log.debug("------------------------------------------------")
                        
                        
           
            # 02 MARKET
                # prevent order_market duplication.
                    # order_way == 'CLOSE' : stop_loss / liquidation condition.
                        # repflect updated row.
            if self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'].values[0] != 1: # if not trade is done, it has np.nan or 1.
                if self.table_trade.loc[self.table_trade['code'] == code, 'order_way'].values[0] == 'CLOSE':
                    if stop_loss_on or closer_on:
                        
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                                    
                        quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                    symbol,
                                                                    orderId)          
                        
                        if quantity_unexecuted:
                            
                            order_result, \
                            error_code = order_market(self, 
                                                    symbol,
                                                    side_close,
                                                    side_position,
                                                    quantity_unexecuted)
                            
                            # normal state : error_code = 0
                                # this is key for updating order_info (change to CLOSE orderId)
                                    # order_info update 이후에 status 가 바뀌고, remove_row 진행하게 된다.
                            if not error_code:
                                self.table_trade.loc[self.table_trade['code'] == code, 'orderId'] = order_result['orderId']
                                                       
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} order_market : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
            
            else:
                
            #############################################
            # 행 제거 & 수익 계산 : status Update 이후.
            #############################################
            
            # if self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'].values[0] == 1:
                    
                # Drop rows
                    # this phase should be placed in the most below, else order_ will set value as np.nan in invalid rows. 
                        # 무슨 소린지 모르겠네.                  
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
                # drow remove_row = 1,
                self.table_trade.drop([idx], inplace=True) # we drop code, individually. (cause using loop.)

                if symbol not in self.table_trade.symbol.values:   
                    # 01 remove from websocket_client.     
                    self.websocket_client.stop_socket("{}@aggTrade".format(symbol.lower()))
                    
                    # 02 remove from price_market dictionary.
                    if symbol in self.price_market.keys():
                        self.price_market.pop(symbol)

                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} drop rows : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")  



                # 수익 계산.
                    # except canceled open order.
                        # 해당 루프는 비동기화 상태에서 loop_condition 과 별도로 동작하기에, 클래스 인스턴스로 접근해도 되지 않나.
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
                income, \
                self.income_accumulated, \
                profit, \
                self.profit_accumulated = get_income_info(self, 
                                                        self.table_log,
                                                        code,
                                                        side_position,
                                                        leverage,
                                                        self.income_accumulated,
                                                        self.profit_accumulated,
                                                        currency="USDT")
                
                self.table_account.loc[self.table_account['account'] == account, 'balance'] += margin
                self.table_account.loc[self.table_account['account'] == account, 'balance'] += income
                
                if self.config.bank.profit_mode == 'COMPOUND':
                    self.table_account.loc[self.table_account['account'] == account, 'balance_origin'] += income
                
                self.config.bank.income_accumulated = self.income_accumulated
                self.config.bank.profit_accumulated = self.profit_accumulated
                
                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} get_income_info : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")  
                # display(self.table_trade)
                
                
                
                # # add profit result by condition.
                # code_split = code.split('_')
                # condition = code_split[1]

                # if len(code_split) == 3 and len(condition) > 5: # reject idx placed instead conndition.
                    
                #     self.config.bank.condition_res.setdefault(condition, {
                #         'income_accumulated': 0,
                #         'profit_accumulated': 0
                #     })
                    
                #     self.config.bank.condition_res[condition]['income_accumulated'] += income
                #     self.config.bank.condition_res[condition]['profit_accumulated'] += profit 
                
                # with open(self.path_config, 'w') as f:
                #     json.dump(self.config, f, indent=4)                
                


            # Prevent losing orderId in every loop.      
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            self.db_manager.replace_table(self.table_account, self.table_account_name)
            self.db_manager.replace_table(self.table_trade, self.table_trade_name, mode=['UPDATE', 'DELETE'])
            
            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} replace_table account & trade : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")  

            
            
            # Control API flood.
            await asyncio.sleep(self.config.term.loop_table_trade)
            
        
        
        # add new_rows afther loop end.
        if new_rows:
            self.table_trade = pd.concat([self.table_trade, pd.DataFrame(new_rows)], ignore_index=True)
            


        # table update. save / send
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        self.db_manager.replace_table(self.table_account, self.table_account_name, send=self.config.database.usage)
        self.db_manager.replace_table(self.table_trade, self.table_trade_name, send=self.config.database.usage, mode=['UPDATE', 'DELETE'])
        self.db_manager.replace_table(self.table_log, self.table_log_name, send=self.config.database.usage)
        
        self.sys_log.debug(f"LoopTableTrade : elasped time, replace_table account & trade & log (send={self.config.database.usage}) : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        
        self.sys_log.debug(f"[loop_table_trade]     END   at {datetime.now().strftime('%H:%M:%S.%f')} (elapsed: {time.time() - loop_start:.2f}s)")
        # await asyncio.sleep(max(0, self.config.term.loop_table_trade - (time.time() - loop_start)))
        await asyncio.sleep(1) 


def loop_table_trade(self):
    
    """
    v1.2.2
        - restore check_stop_loss 20241123 1900
    v1.2.3
        - 주석 수정 및 정리 : 가독성 개선. 20250314 2208.        
        - row info 를 self. 로 받는 이류
            거래 종료 후, self. 객체에 담아서 한번에 필요한 정보 찍어내려고
                허나, 유효기간을 해당 루프내에서만으로 제한한다. 20250317 0850.
        - get_df_new 안한다. (expiry / SL / closer, 나머지 데이터로 진행해야한다.)
        - code has 4 value & 3 underBar (_). 20250318 1845.
        - remove add balance line in Expiry. 20250320 2137.
        - position 추적, balance > balance_orig error 수정 완료. 20250321 0540.
    v1.3.1
        - self. manual 수정 20250321 1907.
        - remove market_close_on --> stop_loss_on & closer_on 20250321 2251.
        - add code & orderId to time log phase. 2025032 2059.
    """
    
    if self.table_trade.empty:
        time.sleep(1)
    else:    
        new_rows = [] # exist for stop_market (following the orderInfo)
        
        for idx, row in self.table_trade.iterrows(): # for save iteration, use copy().
            
            # Init.            
            self.table_trade.at[idx, 'order'] = 0  # default.
            
            order_info = None
            
            # below vars. should not be (re) declared in bank method.
            expired = 0 # 0 / 1 / 2 : No / x / y
            stop_loss_on = False
            closer_on = False

            # update & clean table.
            # some data need to be transferred to Bank's instance, some don't.
            # 'row' means un-updated information. (by get_order_info)
            code = row.code
            symbol = row.symbol  # we have symbol column in table already.
            orderId = row.orderId  # type assertion.
            status_prev = row.status  # save before updating status.
            order_motion = row.order_motion

            side_open = row.side_open
            price_expiration = row.price_expiration
            price_stop_loss = row.price_stop_loss
            price_liquidation = row.price_liquidation

            side_close = row.side_close
            side_position = row.side_position
            
            leverage = row.leverage
            margin = row.margin
            account = row.account

            self.sys_log.debug("row : \n{}".format(row))
            self.sys_log.debug("row.dtypes : {}".format(row.dtypes))
            
            
            try:
                # 코드 형식이 {symbol}_{interval}_{entry_timeIndex}라고 가정합니다.
                _, interval, entry_timeIndex, _ = code.split('_')
            except Exception as e:
                msg = f"Error parsing code {code}: {e}"
                self.sys_log.error(msg)
                self.push_msg(msg)                
                continue # hard assertion.
            
            
            #############################################
            # Set Broker-Ticker option.
                # orderId 가 있더라고, price_market 가 없는 경우 가능 (프로그램 재시작 등.)
            #############################################

            if symbol not in self.price_market.keys() or pd.isnull(orderId): # initial (do only once.)
                
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
                self.websocket_client.agg_trade(symbol=symbol, id=1, callback=self.agg_trade_message_handler)
                self.set_position_mode(dualSidePosition='true')
                self.set_margin_type(symbol=symbol)
                
                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set Broker-Ticker option : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")
            
        
            
            #############################################
            # set order, order_way (OPEN) (order not exists)
            #############################################
            
            if pd.isnull(orderId):
                
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
                self.table_trade.at[idx, 'order'] = 1
                self.table_trade.at[idx, 'order_way'] = 'OPEN'
                
                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set order, order_way (OPEN) : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")


            #  order exists.
            else:
            # if not pd.isnull(orderId): # this means, status is not None. (order in settled)

                # get_order_info. (update order_info)
                    # get_order_info should have valid symbol & orderId.                      
                order_info = get_order_info(self, 
                                        symbol,
                                        orderId)
                
                if order_info is not None: # order should be exist.
                    

                    #############################################
                    # update table_trade
                    #############################################
                    
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    for k, v in order_info.items():
                        self.table_trade.at[idx, k] = v
                        
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} update table_trade : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
                    # display(self.table_trade) 
                       
                    
                    
                    #############################################
                    # check status
                    #############################################
                    
                    self.sys_log.debug(f"status_prev : {status_prev}, order_info['status'] : {order_info['status']}")
                    if status_prev != order_info['status']:                    
                    
                        # set statusChangeTime 
                            # check if status has been changed.
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        # self.table_trade.at[idx, 'statusChangeTime'] = int(time.time())
                        self.table_trade.at[idx, 'statusChangeTime'] = datetime.now().strftime('%Y%m%d%H%M%S%f')
                        self.push_msg("{} status has been changed. {} {}".format(code, row.order_way, order_info['status']))
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set statusChangeTime : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")      
                        # display(self.table_trade)
                            
                            
                        
                        # logging : transfer rows to Table - log.   
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                                        
                        # self.table_log = self.table_log.append(self.table_trade.loc[[idx]], ignore_index=True) # Append the new row to table_log
                        self.table_log = pd.concat([self.table_log, self.table_trade.loc[[idx]]], ignore_index=True)                       

                        self.table_log['id'] = self.table_log.index + 1 # Update 'id' column to be the new index + 1
                        self.db_manager.replace_table(self.table_log, self.table_log_name)
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} replace_table table_log : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")  
                        # display(self.table_trade)
                        
                        
                    
                    #############################################
                    # set remove_row. 
                        # regardless status updated. just check realtime status.
                    #############################################  
                    
                    if (order_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']) or (row.order_way == 'CLOSE' and order_info['status'] == 'FILLED'):
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_trade.at[idx, 'remove_row'] = 1
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set remove_row : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                        

                    
                    #############################################
                    # set order, order_way (for CLOSE)
                    #############################################
                    
                    if row.order_way == 'OPEN':         
                        if (abs(float(order_info['executedQty']) / float(order_info['origQty'])) >= self.config.bank.quantity_open_exec_ratio) or order_info['status'] == 'FILLED':           
                            
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()
                            
                            self.table_trade.at[idx, 'order'] = 1
                            self.table_trade.at[idx, 'order_way'] = 'CLOSE'
                            
                            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set order, order_way (CLOSE) : %.4fs" % (time.time() - start_time)) 
                            self.sys_log.debug("------------------------------------------------")
                            # display(self.table_trade)
                    
                    

                    #############################################
                    # Expiry / Stop Loss / Closer
                    #############################################
                    
                    price_realtime = get_price_realtime(self, symbol)
                    self.sys_log.debug("price_realtime : {}".format(price_realtime))

                    if not pd.isnull(price_realtime):
                        
                        price_realtime_prev = self.table_trade.at[idx, 'price_realtime']
                        
                        slippage_margin = 0.1 # 10 %
                        lower_bound = price_realtime_prev * (1 - slippage_margin)
                        upper_bound = price_realtime_prev * (1 + slippage_margin)                                        
                        self.sys_log.debug(f"price_realtime_prev: {price_realtime_prev:.6f}, lower_bound: {lower_bound:.6f}, upper_bound: {upper_bound:.6f}")
                                                                                    
                        
                        # price_realtime 정상 범위
                        if pd.isnull(price_realtime_prev) or (lower_bound <= price_realtime <= upper_bound):  
                            self.table_trade.at[idx, 'price_realtime'] = price_realtime                    
                    
                    
                            # 01 Check Expiry
                            if row.order_way == 'OPEN' and order_info['status'] in ['NEW', 'PARTIALLY_FILLED']:
                                self.sys_log.debug("------------------------------------------------")
                                start_time = time.time()  
                                
                                expired = check_expiration(self,
                                                        interval,
                                                        entry_timeIndex,
                                                        side_position,
                                                        price_realtime,
                                                        price_expiration,
                                                        ncandle_game=params.get('ncandle_game', 2))
                                    
                                if expired:                                                         
                                    quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                                symbol,
                                                                                orderId)  
                                    
                                    # 어차피, get_income_info phase 에서 margin 내부에 (executed & unexecuted Margin 이 있다.)
                                    # self.table_account.loc[self.table_account['account'] == account, 'balance'] +=(quantity_unexecuted * row.price_entry / leverage)
                                    
                                    # Todo, messeage alertion needed ?
                                    self.push_msg("{} has been expired.".format(code))
                                    # self.user_text = 'watch'  # allowing message.
                                    
                                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check expiration for order_open : %.4fs" % (time.time() - start_time)) 
                                self.sys_log.debug("------------------------------------------------")
            
                        
                        
                            # 02 Check Stop Loss
                                # get stop_loss_on
                                # order_way == 'CLOSE' : Stop Loss or Liquidation
                            elif row.order_way == 'CLOSE':
                                self.sys_log.debug("------------------------------------------------")
                                start_time = time.time()
                                
                                if params.get('use_stop_loss', True) or params.get('use_stop_loss_barclose', False):                                
                                    stop_loss_on = check_stop_loss(self,
                                                                        side_open,
                                                                        price_realtime,
                                                                        price_liquidation,
                                                                        price_stop_loss)
                                
                                # if params.get('use_stop_loss_barclose', False) and stop_loss_on:
                                #     wait_barClosed(self, interval, entry_timeIndex)                                
                                
                                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check_stop_loss : %.4fs" % (time.time() - start_time)) 
                                self.sys_log.debug("------------------------------------------------")
                                
                               
                                
                    # 03 Check Closer
                        # Closer 시그널 발생 시, 해당 interval의 종료 시각까지 대기 후 Market 주문으로 포지션 종료
                    if row.order_way == 'CLOSE':
                        
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()  
                        
                        closer_on = check_closer(self, 
                                                    interval, 
                                                    entry_timeIndex,
                                                    ncandle_game=params.get('ncandle_game', 2))
                        
                        # if params.get('use_closer', False) and closer_on:
                        #     wait_barClosed(self, interval, entry_timeIndex)
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check_stop_loss : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                
                
            
            # ORDER.
            if not self.config.bank.backtrade and not order_motion:

                # ORDER 조건.
                if self.table_trade.at[idx, 'order'] == 1:

                    ####################################
                    # 전처리 : 가격 & 거래량 수집.                      
                    ####################################   
                    
                        # public set.
                    order_type = 'LIMIT' # fixed for order_limit.
                    side_position = side_position                
                                    
                        # OPEN condition
                    if self.table_trade.at[idx, 'order_way'] == 'OPEN':
                                     
                        side_order = row.side_open
                        price = row.price_entry
                        quantity = row.quantity_open
        
                        # CLOSE condition
                    else: #  self.table_trade.at[idx, 'order_way'] == 'CLOSE': # upper phase's condition use 'order' = 1                                
                        # we don't do partial like this anymore.
                            # each price info will be adjusted order by order.
                                         
                        side_order = row.side_close
                        price = row.price_take_profit
                        quantity = order_info['executedQty'] # OPEN's qty.      
                
                
                    
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()                               
                                 
                    # 01 LIMIT
                    order_result, \
                    error_code = order_limit(self, 
                                            symbol,
                                            side_order, 
                                            side_position, 
                                            price, 
                                            quantity)
                
                    # normal state : error_code = 0
                    if not error_code:
                        # self.table_trade.at[idx, 'orderId'] = str(order_result['orderId']) # this is key for updating order_info (change to OPEN / CLOSE orderId)
                        self.table_trade.at[idx, 'orderId'] = order_result['orderId'] # this is key for updating order_info (change to OPEN / CLOSE orderId)
                        
                    else:
                        # CLOSE order_error should be cared more strictly.
                        self.table_trade.at[idx, 'remove_row'] = 1
                        
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} order_limit : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
                    
                    
                    
                    # 02 STOP MARKET
                    # if self.table_trade.at[idx, 'order_way'] == 'CLOSE':                    
                        
                    #     self.sys_log.debug("------------------------------------------------")
                    #     start_time = time.time()
                                    
                    #     order_result, \
                    #     error_code = order_stop_market(self, 
                    #                                 symbol,
                    #                                 side_order,
                    #                                 side_position,
                    #                                 row.price_stop_loss,
                    #                                 quantity)
                        
                    #     # normal state : error_code = 0
                    #     if not error_code:                        
                    #         new_row = self.table_trade.loc[idx].copy()
                    #         new_row['id'] = np.nan # set as a missing_id for "fill_missing_ids"
                    #         new_row['orderId'] = order_result['orderId'] # this is key for updating order_info (change to CLOSE orderId)
                    #         new_rows.append(new_row)
                                                    
                    #     self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} order_market : %.4fs" % (time.time() - start_time)) 
                    #     self.sys_log.debug("------------------------------------------------")
                        
                        
           
            # 02 MARKET
                # prevent order_market duplication.
                    # order_way == 'CLOSE' : stop_loss / liquidation condition.
                        # repflect updated row.
            if self.table_trade.at[idx, 'remove_row'] != 1: # if not trade is done, it has np.nan or 1.
                if self.table_trade.at[idx, 'order_way'] == 'CLOSE':
                    if stop_loss_on or closer_on:
                        
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                                    
                        quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                    symbol,
                                                                    orderId)          
                        
                        if quantity_unexecuted:
                            
                            order_result, \
                            error_code = order_market(self, 
                                                    symbol,
                                                    side_close,
                                                    side_position,
                                                    quantity_unexecuted)
                            
                            # normal state : error_code = 0
                                # this is key for updating order_info (change to CLOSE orderId)
                                    # order_info update 이후에 status 가 바뀌고, remove_row 진행하게 된다.
                            if not error_code:
                                self.table_trade.at[idx, 'orderId'] = order_result['orderId']
                                                       
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} order_market : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        


            #############################################
            # 행 제거 & 수익 계산 : status Update 이후.
            #############################################
            
            if self.table_trade.at[idx, 'remove_row'] == 1: 
                    
                    
                # Drop rows
                    # this phase should be placed in the most below, else order_ will set value as np.nan in invalid rows. 
                        # 무슨 소린지 모르겠네.                  
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
                # drow remove_row = 1,
                self.table_trade.drop([idx], inplace=True) # we drop code, individually. (cause using loop.)

                if symbol not in self.table_trade.symbol.values:   
                    # 01 remove from websocket_client.     
                    self.websocket_client.stop_socket("{}@aggTrade".format(symbol.lower()))
                    
                    # 02 remove from price_market dictionary.
                    if symbol in self.price_market.keys():
                        self.price_market.pop(symbol)

                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} drop rows : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")  



                # 수익 계싼
                    # except canceled open order.
                        # 해당 루프는 비동기화 상태에서 loop_condition 과 별도로 동작하기에, 클래스 인스턴스로 접근해도 되지 않나.
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
                income, \
                self.income_accumulated, \
                profit, \
                self.profit_accumulated = get_income_info(self, 
                                                        self.table_log,
                                                        code,
                                                        side_position,
                                                        leverage,
                                                        self.income_accumulated,
                                                        self.profit_accumulated,
                                                        currency="USDT")
                
                self.table_account.loc[self.table_account['account'] == account, 'balance'] += margin
                self.table_account.loc[self.table_account['account'] == account, 'balance'] += income
                
                if self.config.bank.profit_mode == 'COMPOUND':
                    self.table_account.loc[self.table_account['account'] == account, 'balance_origin'] += income
                
                self.config.bank.income_accumulated = self.income_accumulated
                self.config.bank.profit_accumulated = self.profit_accumulated
                
                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} get_income_info : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")  
                # display(self.table_trade)
                
                
                
                # # add profit result by condition.
                # code_split = code.split('_')
                # condition = code_split[1]

                # if len(code_split) == 3 and len(condition) > 5: # reject idx placed instead conndition.
                    
                #     self.config.bank.condition_res.setdefault(condition, {
                #         'income_accumulated': 0,
                #         'profit_accumulated': 0
                #     })
                    
                #     self.config.bank.condition_res[condition]['income_accumulated'] += income
                #     self.config.bank.condition_res[condition]['profit_accumulated'] += profit 
                
                # with open(self.path_config, 'w') as f:
                #     json.dump(self.config, f, indent=4)                
                


            # Prevent losing orderId in every loop.      
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            self.db_manager.replace_table(self.table_account, self.table_account_name)
            self.db_manager.replace_table(self.table_trade, self.table_trade_name, mode=['UPDATE', 'DELETE'])
            
            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} replace_table account & trade : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")  

            
            
            # Prevent API flood.
            time.sleep(self.config.term.loop_table_trade)
            
        
        
        # add new_rows afther loop end.
        if new_rows:
            self.table_trade = pd.concat([self.table_trade, pd.DataFrame(new_rows)], ignore_index=True)
            


        # table update. save / send
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        self.db_manager.replace_table(self.table_account, self.table_account_name, send=self.config.database.usage)
        self.db_manager.replace_table(self.table_trade, self.table_trade_name, send=self.config.database.usage, mode=['UPDATE', 'DELETE'])
        self.db_manager.replace_table(self.table_log, self.table_log_name, send=self.config.database.usage)
        
        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} replace_table account & trade & log (send={self.config.database.usage}) : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")  


