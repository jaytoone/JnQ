from bank import *
from idep import *
from params import * 


from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

from dataclasses import dataclass
# from typing import Optional, Any


# executor = ThreadPoolExecutor(max_workers=10)
# executor = ThreadPoolExecutor(max_workers=30)
executor = ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2))
# executor = ProcessPoolExecutor(max_workers=os.cpu_count() - 1)


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
        


async def loop_save_df_res_async(self):
    """
    v0.2
        - track batch save duration. 20250411 0843.
    """
    batch_start_time = None
    saving_count = 0
    
    self.sys_log.debug(f"[loop_save_df_res_async] START at {datetime.now().strftime('%H:%M:%S.%f')}")

    while True:
        try:
            if self.queue_df_res.qsize() > 0 and batch_start_time is None:
                batch_start_time = time.time()
                saving_count = self.queue_df_res.qsize()

            item = await self.queue_df_res.get()

            start_time = time.time()

            df_res = item['df']
            symbol = item['symbol']
            interval = item['interval']

            if df_res is not None:
                try:
                    c_index = self.config.bank.complete_index
                    path_dir_df_res = f"{self.path_dir_df_res}\\{str(df_res.index[c_index]).split(' ')[-1].replace(':', '')}"
                    os.makedirs(path_dir_df_res, exist_ok=True)

                    path_df_res = f"{path_dir_df_res}\\{symbol}_{interval}.csv"
                    df_res.to_csv(path_df_res, index=True)
                    
                    self.sys_log.debug(f"[loop_save_df_res_async] Saved {symbol} {interval} "
                                       f"(elapsed: {time.time() - start_time:.4f}s)")
                    
                except Exception as io_err:
                    self.sys_log.error(f"[loop_save_df_res_async] File I/O error: {io_err}")
            else:
                self.sys_log.warning(f"[loop_save_df_res_async] Received None df_res for {symbol}-{interval}, skipping.")

        except Exception as e:
            self.sys_log.error(f"[loop_save_df_res_async] General error: {e}")
        finally:
            self.queue_df_res.task_done()

            # Check if queue is empty after processing
            if self.queue_df_res.empty() and batch_start_time is not None:
                batch_elapsed = time.time() - batch_start_time                
                self.sys_log.debug(f"[loop_save_df_res_async] END Items: {saving_count}, at {datetime.now().strftime('%H:%M:%S.%f')} (elapsed: {batch_elapsed:.2f}s)")

                batch_start_time = None
                saving_count = 0

        
async def loop_table_condition_async(self, drop=False, debugging=False):
    """
    v2.1.4
        - Async Full Refactor. 20250410 1214.
    v2.1.5
        - reorder save df_res phase. 20250410 1638.
        - adj. get_trade_info_sync v0.2.2
    """

    data_len = 100
    kilne_limit = 100

    while True:
        loop_start = time.time()
        self.sys_log.debug(f"[loop_table_condition] START at {datetime.now().strftime('%H:%M:%S.%f')}")

        # Check available tokens before proceeding
        # tokens_required = len(self.table_condition)
        # self.sys_log.debug(f"[loop_table_condition] Tokens needed: {tokens_required}")
        # self.token_bucket.wait_for_tokens(tokens_required)  # BLOCKS until enough tokens are available
        
        if self.table_condition.empty:
            await asyncio.sleep(1)
            continue


        # STEP 1: 수집 대상 필터링 및 fetch_df 준비
        semaphore = asyncio.Semaphore(32)  # API 제한 고려
        tasks_step1 = []
        job_list = []  # 유효한 작업 후보 저장
        
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        for idx, row in self.table_condition.iterrows():
            # symbol = row.symbol
            interval = row.interval
            interval_number = itv_to_number(interval)
            timestamp_current = int(time.time())

            # interval 중복 방지
            if timestamp_current - int(row.timestampLastUsed) < interval_number * 60:
                continue
            else:
                timestamp_anchor = 1718236800  # 기준 anchor
                timestamp_remain = (timestamp_current - timestamp_anchor) % (interval_number * 60)
                timestamp_current -= timestamp_remain
                self.table_condition.at[idx, 'timestampLastUsed'] = timestamp_current

                days = data_len / (60 * 24 / interval_number)
                job_list.append((idx, row, days))

        # STEP 2-1: df 수집 (get_df_new_async)
        async with aiohttp.ClientSession() as session:
            async def fetch_task(idx_, symbol_, interval_, days_):
                async with semaphore:
                    return idx_, await get_df_new_async(self, session, symbol_, interval_, days_, limit=kilne_limit)

            tasks_step1 = [
                fetch_task(idx, row.symbol, row.interval, days)
                for idx, row, days in job_list
            ]
            df_results = await asyncio.gather(*tasks_step1)
        
        self.sys_log.debug(f"LoopTableCondition : elasped time, get_df_new_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")


        # STEP 2-2: 조건 분석 (get_trade_info_sync → run_in_executor 사용한 병렬 처리)
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        loop = asyncio.get_running_loop()
        tasks_step2 = [
            loop.run_in_executor(
                executor,              # ThreadPoolExecutor 또는 ProcessPoolExecutor
                get_trade_info_sync,   # 동기화된 조건 분석 함수
                self,                  # 클래스 인스턴스 (context)
                idx,                   # 조건 테이블 인덱스
                df_res,                # 수집된 df 결과
                row.symbol,            # 종목
                row.position,          # 포지션
                row.interval,          # 인터벌
                row.divider,           # 분할 비율
                row.account,           # 계좌
                row.side_open,         # 진입 방향
                debugging              # 디버깅 플래그
            )
            for (idx, df_res), (idx_, row, _) in zip(df_results, job_list)
            if isinstance(df_res, pd.DataFrame) and not df_res.empty
        ]
        
        # tasks_step2 = [
        #     get_trade_info_async(
        #         self,                  # 클래스 인스턴스 (context)
        #         idx,                   # 조건 테이블 인덱스
        #         df_res,                # 수집된 df 결과
        #         row.symbol,            # 종목
        #         row.position,          # 포지션
        #         row.interval,          # 인터벌
        #         row.divider,           # 분할 비율
        #         row.account,           # 계좌
        #         row.side_open,         # 진입 방향
        #         debugging              # 디버깅 플래그
        #     )
        #     for (idx, df_res), (idx_, row, _) in zip(df_results, job_list)
        #     if isinstance(df_res, pd.DataFrame) and not df_res.empty
        # ]

        trade_infos = await asyncio.gather(*tasks_step2)        
        
        self.sys_log.debug(f"LoopTableCondition : elasped time, get_trade_info_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")


        # self.sys_log.debug("------------------------------------------------")
        # start_time = time.time()
        
        # async for completed in as_completed(tasks_step2):
        #     trade_info = await completed
        #     # trade_infos.append(trade_info)

        #     if trade_info.position:
        #         await init_table_trade_async(self, trade_info)
            
        #     if isinstance(trade_info.df_res, pd.DataFrame):
        #         await enqueue_df_res_async(self, trade_info.df_res, trade_info.symbol, trade_info.interval)

        # self.sys_log.debug(f"LoopTableCondition : elapsed time (as_completed): %.4fs" % (time.time() - start_time))
        # self.sys_log.debug("------------------------------------------------")
        
        
        # STEP 3: 테이블 초기화 (position 존재하는 경우만)
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        tasks_step3 = [
            init_table_trade_async(self, trade_info)
            for trade_info in trade_infos if trade_info.position
        ]
        await asyncio.gather(*tasks_step3)
        
        self.sys_log.debug(f"LoopTableCondition : elasped time, init_table_trade_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")        
        
        
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        for ti in trade_infos:
            if ti and isinstance(ti.df_res, pd.DataFrame):
                await enqueue_df_res_async(self, ti.df_res, ti.symbol, ti.interval)
                
        self.sys_log.debug(f"LoopTableCondition : elasped time, enqueue_df_res_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")


        # STEP 4: DB 반영
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        async with self.lock_trade:
            self.db_manager.replace_table(
                self.table_condition,
                self.table_condition_name,
                send=self.config.database.usage
            )
            
        self.sys_log.debug(f"LoopTableCondition : elasped time, replace_table condition (send={self.config.database.usage}) : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")

        loop_elapsed_time = time.time() - loop_start
        self.sys_log.debug(f"[loop_table_condition] END   at {datetime.now().strftime('%H:%M:%S.%f')} (elapsed: {loop_elapsed_time:.2f}s)")
        await asyncio.sleep(max(0, 1 - loop_elapsed_time))
          
        
def get_trade_info_sync(self, idx, df_res, symbol, position, interval, divider, account, side_open, debugging):
    
    """
    v0.2.1
        - remove df_res fetch. 20250410 1243.
        - exclude df_res save for speed. 20250410 1415.
    v0.2.2
        - return default TradeInfo for df_res. 20250410 2248.
    v0.2.4
        - add fast return mode. 20250411 0726. --> 효과 크게 없다.
    """
    
    timestamp_millis = datetime.now().strftime('%f')
    c_index = self.config.bank.complete_index
    code = f"{symbol}_{interval}_{str(df_res.index[c_index])}_{timestamp_millis}"
    
    side_open = None
    position = None
    
    price_expiration = None
    price_take_profit = None
    price_entry = None
    price_stop_loss = None

    try:
        if df_res is None or isinstance(df_res, str) or len(df_res) < abs(c_index):
            msg = f"Warning in df_res : {df_res}"
            self.sys_log.warning(msg)
            self.push_msg(msg)
            return TradeInfo(symbol, side_open, position, interval, divider, account,
                             df_res, code,
                             price_expiration, price_take_profit, price_entry, price_stop_loss)

        modes = params.get("modes", [])


        # Indicator Phase
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        df_res = (
            df_res
            .pipe(get_DC, params.get("DC_period", 5))
            .pipe(get_DCconsecutive)
            .pipe(lambda df: get_CCI(df, params.get("CCI_period", 21), params.get("CCIMA_period", 14)) if 'adjCCIMAGapStarter' in modes else df)
            .pipe(lambda df: get_BB(df, params.get("BB_period", 20), params.get("BB_multiple", 2), level=params.get("BB_level", 2)) if 'adjBBOpen' in modes or 'adjBBCandleBodyAnchor' in modes else df)
            .pipe(get_CandleFeatures)
            .pipe(get_SAR, params.get("SAR_start", 0.02), params.get("SAR_increment", 0.02), params.get("SAR_maximum", 0.2))
        )

        self.sys_log.debug(f"[get_trade_info_sync] {symbol} {interval} elapsed time: get_indicators = {time.time() - start_time:.4f}s")
        self.sys_log.debug("------------------------------------------------")


        # Price Channel + Opener Phase
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        df_res = (
            df_res
            .pipe(get_PriceChannel, **params)
            .pipe(get_Opener, **params)
        )

        row = df_res.iloc[c_index]
        opener_long, opener_short = row['OpenerLong'], row['OpenerShort']
        self.sys_log.debug(f"Opener status: Long = {opener_long}, Short = {opener_short}")
        self.sys_log.debug(f"[get_trade_info_sync] {symbol} {interval} elapsed time: get_price_channel & opener = {time.time() - start_time:.4f}s")
        self.sys_log.debug("------------------------------------------------")

        if not (opener_long or opener_short or debugging):
            return TradeInfo(symbol, side_open, position, interval, divider, account,
                             df_res, code,
                             price_expiration, price_take_profit, price_entry, price_stop_loss)


        # Signal Phase
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        df_res = get_Direction(df_res, **params)
        if not df_res['DirectionLong'].any() and not df_res['DirectionShort'].any():
            return TradeInfo(symbol, side_open, position, interval, divider, account,
                             df_res, code,
                             price_expiration, price_take_profit, price_entry, price_stop_loss)

        df_res = get_Momentum(df_res, **params)
        if not df_res['MomentumLong'].any() and not df_res['MomentumShort'].any():
            return TradeInfo(symbol, side_open, position, interval, divider, account,
                             df_res, code,
                             price_expiration, price_take_profit, price_entry, price_stop_loss)

        df_res = get_Anchor(df_res, **params)
        df_res = get_Starter(df_res, **params)
        df_res = get_SR(df_res, **params)
        df_res = get_Expiry(df_res, **params)

        df_res['long'] = df_res['OpenerLong'] & df_res['StarterLong'] & df_res['AnchorLong'] & df_res['SRLong'] & df_res['MomentumLong'] & df_res['DirectionLong']
        df_res['short'] = df_res['OpenerShort'] & df_res['StarterShort'] & df_res['AnchorShort'] & df_res['SRShort'] & df_res['MomentumShort'] & df_res['DirectionShort']

        self.sys_log.debug(f"[get_trade_info_sync] {symbol} {interval} elapsed time: get_signal = {time.time() - start_time:.4f}s")
        self.sys_log.debug("------------------------------------------------")


        # PriceBox Extraction Phase
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        row = df_res.iloc[c_index]
        if row['long']:
            position, side_col, side_open = 'LONG', 'long', 'BUY'
        elif row['short']:
            position, side_col, side_open = 'SHORT', 'short', 'SELL'

        if position:
            price_expiration   = row[f'price_expiry_{side_col}']
            price_take_profit  = row[f'price_take_profit_{side_col}']
            price_entry        = row[f'price_entry_{side_col}']
            price_stop_loss    = row[f'price_stop_loss_{side_col}']

        self.sys_log.debug(f"[get_trade_info_sync] {symbol} {interval} elapsed time: get_price_box = {time.time() - start_time:.4f}s")
        self.sys_log.debug("------------------------------------------------")

    except Exception as e:
        self.sys_log.error(f"Error in get_trade_info_sync for {symbol}: {e}")

    return TradeInfo(
        symbol, side_open, position, interval, divider, account,
        df_res, code,
        price_expiration, price_take_profit, price_entry, price_stop_loss
    )


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
    v2.6
        - adj. asyncio queue. 20250406 1941.
        - add row.leverage_limit_user 20250406 2058.
        - modify historical_high / low 20250407 1209.
    v2.6.1
        - init margin as 0. 20250409 0554.
        - add balance_available. 20250409 0643.
    v2.6.2
        - adj. leverage_limit_const. 20250415 0038.
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
            
    self.sys_log.debug("price_take_profit : {}".format(price_take_profit))
    self.sys_log.debug("price_entry : {}".format(price_entry))
    self.sys_log.debug("price_stop_loss : {}".format(price_stop_loss))
    self.sys_log.debug("price_liquidation : {}".format(price_liquidation))
    self.sys_log.debug("price_expiration : {}".format(price_expiration))
    
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
    balance_origin, \
    balance_available = get_account_normality(self, 
                                            account)
    
    self.sys_log.debug("account_normality : {}".format(account_normality))
    self.sys_log.debug("balance_ : {}".format(balance_))
    self.sys_log.debug("balance_origin : {}".format(balance_origin))
    self.sys_log.debug("balance_available : {}".format(balance_available))
    
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_account_normality : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
    
    
    
    ##################################################
    # add to TableTrade
    ##################################################
    
    # self.sys_log.debug("------------------------------------------------")
    # start_time = time.time()  
    
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
    
    complete_index = self.config.bank.complete_index
    latest_index = self.config.bank.latest_index

    historical_high = df_res['high'].iloc[complete_index:latest_index + 1].max()
    historical_low = df_res['low'].iloc[complete_index:latest_index + 1].min()

    table_trade_new_row.historical_high = historical_high   # added.  
    table_trade_new_row.historical_low = historical_low     # added.  
    
    table_trade_new_row.account = account
    table_trade_new_row.balance = balance_                 # added.  
    table_trade_new_row.balance_origin = balance_origin    # added.      
    
    table_trade_new_row.margin = 0    # margin 은 balance 와 다시 합해지는 것을 고려해 0 으로 초기화.
    
    
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
        
        self.sys_log.debug("target_loss : {}".format(target_loss))
        self.sys_log.debug("loss : {}".format(loss))
        self.sys_log.debug("quantity_open : {}".format(quantity_open))
        self.sys_log.debug("amount_entry : {}".format(amount_entry))
            
        table_trade_new_row.target_loss = target_loss          # added.  
        table_trade_new_row.quantity_open = quantity_open 
        table_trade_new_row.amount_entry = amount_entry
        
        self.sys_log.debug(f"InitTableTrade : elasped time, {code} get margin : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        
        
        # func3 : check margin normality (> min balance)
        if amount_entry > self.config.broker.amount_min:     
            
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
            
            leverage_limit_const = leverage_limit_user / leverage_limit_server            
            margin = amount_entry / leverage            
                        
            self.sys_log.debug("leverage_limit_user : {}".format(leverage_limit_user))
            self.sys_log.debug("leverage_limit_server : {}".format(leverage_limit_server))      
            self.sys_log.debug("leverage_limit_const : {}".format(leverage_limit_const))  # add      
            self.sys_log.debug("leverage : {}".format(leverage))
            self.sys_log.debug("margin : {}".format(margin))            
            
            table_trade_new_row.leverage_limit_user = leverage_limit_user
            table_trade_new_row.leverage_limit_server = leverage_limit_server     # add
            table_trade_new_row.leverage_limit_const = leverage_limit_const       # add
            table_trade_new_row.leverage = leverage
            table_trade_new_row.margin = margin
            
            self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_leverage : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
            # check leverage_limit_const
            leverage_limit_const_max = params.get('leverage_limit_const', 2)
            if leverage_limit_const < leverage_limit_const_max:
                
                # check if total balance available for the margin.
                if balance_available > margin:    
                    
                    # set leverage
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                                        
                    self.set_leverage(symbol, leverage)
                        
                    self.sys_log.debug(f"InitTableTrade : elasped time, {code} set_leverage : %.4fs" % (time.time() - start_time))
                    self.sys_log.debug("------------------------------------------------")
                    
                    
                    # func4 : subtract margin if normal on 'balance'.
                    self.table_account.loc[self.table_account['account'] == account, 'balance'] -= margin
                else:
                    table_trade_new_row.status = 'ERROR : balance_available <= margin'    # added.
                    table_trade_new_row.remove_row = 1              
            else: 
                table_trade_new_row.status = f"ERROR : leverage_limit_const > {leverage_limit_const_max}"
                table_trade_new_row.remove_row = 1
            
        else:
            table_trade_new_row.status = 'ERROR : amount_entry <= amount_min'    # added.
            table_trade_new_row.remove_row = 1
                    
    else:
        table_trade_new_row.status = 'ERROR : account_normality'    # added.
        table_trade_new_row.remove_row = 1
        
        
    # append row.
    await self.queue_trade_init.put(table_trade_new_row)
    # async with self.lock_trade:
    #     self.table_trade = pd.concat([self.table_trade, table_trade_new_row]).reset_index(drop=True)            
    
    self.sys_log.debug(f"[QUEUE] {code} row enqueued to queue_trade_init.") 
        
        
    # remove. table_trade_async 에서 진행.
    # self.sys_log.debug("------------------------------------------------")
    # start_time = time.time()
    
    # self.db_manager.replace_table(self.table_account, self.table_account_name, send=self.config.database.usage)
    # self.db_manager.replace_table(self.table_trade, self.table_trade_name, send=self.config.database.usage, mode=['UPDATE', 'DELETE'])
    
    # self.sys_log.debug(f"InitTableTrade : elasped time, {code} replace_table (send={self.config.database.usage})  : %.4fs" % (time.time() - start_time)) 
    # self.sys_log.debug("------------------------------------------------")
    
    self.sys_log.debug(f"[init_table_trade] END   at {datetime.now().strftime('%H:%M:%S.%f')} (elapsed: {time.time() - loop_start:.2f}s)")
  
     
async def enqueue_df_res_async(self, df_res, symbol, interval):
    """
    df_res 저장용 항목을 queue_df_res에 enqueue
    """
    try:
        save_item = {
                "df": df_res,
                "symbol": symbol,
                "interval": interval
            }
        await self.queue_df_res.put(save_item)
        self.sys_log.debug(f"[enqueue_df_res_async] Item queued: {symbol} {interval}")
    except Exception as e:
        self.sys_log.error(f"[enqueue_df_res_async] Queue error: {e}")
 
 
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
        - add table_snapshot for race_condition. 20250406 0031.
    v2.1.4
        - update status. 20250406 1934.
    v2.2
        - add asyncio queue. 20250406 1938.
        - add row.leverage_limit_user 20250406 2058.
    v2.2.1
        - add new col (status_2) 20250406 2331.
        - modify statusChangeTime format. 20250408 2300.
        - modify status_error logic, make it more clear. 20250409 0605.
        - table_trade update 는 해당 루프에만 존재하기 때문에, lock 이 반드시 필요하지는 않는다. 20250411 0644.
        - modify await time by table_empty condition. 20250412 0912.
    """
    
    
    while True:
        
        loop_start = time.time()
        self.sys_log.debug(f"[loop_table_trade]     START at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        while not self.queue_trade_init.empty():
            table_trade_new_row = await self.queue_trade_init.get()
            async with self.lock_trade:
                self.table_trade = pd.concat([self.table_trade, table_trade_new_row]).reset_index(drop=True)    
                code = table_trade_new_row.code.values[0]
                self.sys_log.debug(f"[QUEUE_CONSUMED] {code} row inserted into table_trade")    
        
        async with self.lock_trade:
            # if self.table_trade.empty:
            #     await asyncio.sleep(1)
            #     continue

            table_snapshot = self.table_trade.copy()
        
        new_rows = [] # exist for stop_market (following the orderInfo)
        
        for idx, row in table_snapshot.iterrows(): # for save iteration, use copy().
            
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
            remove_row = row.remove_row
            order_way = row.order_way
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
            leverage_limit_user = row.leverage_limit_user
            margin = row.margin
            account = row.account
                            
            # status_error = 1 if 'ERROR' in str(status_prev) else 0
            status_error = remove_row == 1 and pd.isnull(order_way)
            self.sys_log.debug(f"status_error : {status_error}")
            
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
                
                if order_info:
                # if order_info is not None:
                    # order 이 있는 경우에만 진행하는 것들
                        # 1 update table_trade 
                        # 3 update remove_row, status
                        # 2 update statusChangeTime + table_logging.
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
                        # set remove_row, status
                            # regardless status updated. just check realtime status.
                    #############################################                      
                    self.sys_log.debug(f"status_prev : {status_prev}, order_info['status'] : {order_info['status']}")
                    
                    if (order_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']) or (row.order_way == 'CLOSE' and order_info['status'] == 'FILLED'):
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'] = 1 # 아래에 로깅 구간이 있기 때문에, 로깅 불필요하다.
                        
                        if order_info['status'] == 'FILLED' and order_info['type'] == 'LIMIT':
                            self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'TP'    
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set remove_row : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                    
                    
                    
                    #############################################
                    # set statusChangeTime & table_logging.
                    #############################################
                    if status_prev != order_info['status']:                    
                    
                        # set statusChangeTime 
                            # check if status has been changed.
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        # self.table_trade.loc[self.table_trade['code'] == code, 'statusChangeTime'] = datetime.now().strftime('%Y%m%d%H%M%S%f')
                        self.table_trade.loc[self.table_trade['code'] == code, 'statusChangeTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")                                                
                        self.push_msg("{} status has been changed. {} {}".format(code, row.order_way, order_info['status']))
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set statusChangeTime : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")      
                        # display(self.table_trade)
                            
                            
                        
                        # logging : transfer rows to Table - log.   
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_log = pd.concat([self.table_log, self.table_trade[self.table_trade['code'] == code]], ignore_index=True)
                        self.table_log['id'] = self.table_log.index + 1 # Update 'id' column to be the new index + 1
                        
                        # 추후 도입 예정
                        if len(self.table_log) > 10000:
                            self.table_log = self.table_log.iloc[-7000:].copy()
                        
                        self.db_manager.replace_table(self.table_log, self.table_log_name)
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} replace_table table_log : %.4fs" % (time.time() - start_time)) 
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
                                    self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = f"Expired_{'x' if expired == 1 else 'y'}"
                                    
                                    # we need this, cause order should be canceled !                                            
                                    quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                                symbol,
                                                                                orderId)  
                                    
                                    # 어차피, get_income_info phase 에서 margin 내부에 (executed & unexecuted Margin 이 있다.)
                                    # self.table_account.loc[self.table_account['account'] == account, 'balance'] +=(quantity_unexecuted * row.price_entry / leverage)
                                    
                                    # Todo, messeage alertion needed ?
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
                                    
                                    if stop_loss_on:
                                        self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'SL'
                                
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
                                
                        if closer_on:
                            self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'Closer'
                        
                        # if params.get('use_closer', False) and closer_on:
                        #     wait_barClosed(self, interval, entry_timeIndex)
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check_closer : %.4fs" % (time.time() - start_time)) 
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
                           
            #############################################
            # 행 제거 & 수익 계산 : status Update 이후.
            #############################################
            else:            
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
                                                        leverage_limit_user,
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


        # # table update. save / send
        # self.sys_log.debug("------------------------------------------------")
        # start_time = time.time()
        
        # self.db_manager.replace_table(self.table_account, self.table_account_name, send=self.config.database.usage)
        # self.db_manager.replace_table(self.table_trade, self.table_trade_name, send=self.config.database.usage, mode=['UPDATE', 'DELETE'])
        # self.db_manager.replace_table(self.table_log, self.table_log_name, send=self.config.database.usage)
        
        # self.sys_log.debug(f"LoopTableTrade : elasped time, replace_table account & trade & log (send={self.config.database.usage}) : %.4fs" % (time.time() - start_time)) 
        # self.sys_log.debug("------------------------------------------------")
        
        loop_elapsed_time = time.time() - loop_start
        self.sys_log.debug(f"[loop_table_trade]     END   at {datetime.now().strftime('%H:%M:%S.%f')} (elapsed: {loop_elapsed_time:.2f}s)")
        
        if self.table_trade.empty and self.queue_trade_init.empty():
            await asyncio.sleep(max(0, 1 - loop_elapsed_time))
        # await asyncio.sleep(1)



