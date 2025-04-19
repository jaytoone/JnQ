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
    v2.4
        - Refactored to use as_completed for faster per-symbol processing. 20250419
    v2.4.1
        - ClientSession delayed until job_list confirmed, 20250419 1023
        - row caching applied, 
        - batch init/enqueue maintained. 
        - 2000 row 기준 5초를 더 허용하되, 일괄보다 낫겠다 이거야.
    """

    data_len = 100
    kilne_limit = 100

    while True:
        loop_start = time.time()
        self.sys_log.debug(f"[loop_table_condition] START at {datetime.now().strftime('%H:%M:%S.%f')}")

        if self.table_condition.empty:
            await asyncio.sleep(1)
            continue

        # STEP 1: 수집 대상 필터링 및 fetch_df 준비
        semaphore = asyncio.Semaphore(32)  # API 제한 고려
        job_list = []  # 유효한 작업 후보 저장

        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        for idx, row in self.table_condition.iterrows():
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


        if job_list:
                        
            # row 캐시 (pandas loc 제거 목적)
            job_dict = {idx: row for idx, row, _ in job_list}

            # STEP 2-1: df 수집 (get_df_new_async)
            async with aiohttp.ClientSession() as session:
                async def fetch_task(idx_, symbol_, interval_, days_):
                    async with semaphore:
                        return idx_, await get_df_new_async(self, session, symbol_, interval_, days_, limit=kilne_limit)

                tasks_step1 = [
                    fetch_task(idx, row.symbol, row.interval, days)
                    for idx, row, days in job_list
                ]

                # STEP 2-2: 조건 분석 및 테이블 초기화 병렬 수행
                loop = asyncio.get_running_loop()
                tasks_init, tasks_enqueue = [], []

                for completed in asyncio.as_completed(tasks_step1):
                    idx, df_res = await completed
                    row = job_dict[idx]

                    if not isinstance(df_res, pd.DataFrame) or df_res.empty:
                        continue

                    # 조건 분석 (get_trade_info_sync → run_in_executor 사용)
                    trade_info = await loop.run_in_executor(
                        executor,
                        get_trade_info_sync,
                        self,
                        idx,
                        df_res,
                        row.symbol,
                        row.position,
                        row.interval,
                        row.divider,
                        row.account,
                        row.side_open,
                        debugging
                    )

                    # 테이블 초기화
                    if trade_info.position:
                        tasks_init.append(init_table_trade_async(self, trade_info))

                    # df_res enqueue
                    if isinstance(trade_info.df_res, pd.DataFrame):
                        tasks_enqueue.append(enqueue_df_res_async(self, trade_info.df_res, trade_info.symbol, trade_info.interval))

                # STEP 3: 테이블 초기화 및 df_res enqueue 병렬 처리
                await asyncio.gather(*tasks_init)
                await asyncio.gather(*tasks_enqueue)

                self.sys_log.debug(f"LoopTableCondition : elapsed time (total loop): %.4fs" % (time.time() - start_time))
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
    v2.6
        - adj. asyncio queue. 20250406 1941.
        - add row.leverage_limit_user 20250406 2058.
        - modify historical_high / low 20250407 1209.
    v2.6.1
        - init margin as 0. 20250409 0554.
        - add balance_available. 20250409 0643.
    v2.6.2
        - adj. leverage_limit_const. 20250415 0038.
        - modify historical_high / low slicing. 20250417 0825.
        - modify margin phase. (if margin calculated, subtract from balance) 20250417 2033.
            - if error, margin = 0. 20250418 0032.
    """
    
    
    loop_start = time.time()
    self.sys_log.debug(f"[init_table_trade] START at {datetime.now().strftime('%H:%M:%S.%f')}")
           
    # ──────────────────────────────────────────────
    # INIT
    # ──────────────────────────────────────────────
    
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
    
    
    # ──────────────────────────────────────────────
    # Get side.
    # ──────────────────────────────────────────────
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
        
    side_close, \
    side_position = get_side_info(self, 
                                    side_open)
        
    self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_side_info : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
     
    
    # ──────────────────────────────────────────────
    # Pricing
    # ──────────────────────────────────────────────
    
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

    
    # ──────────────────────────────────────────────
    # balance.
    # ──────────────────────────────────────────────
    
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
    
    
    
    # ──────────────────────────────────────────────
    # add to TableTrade
    # ──────────────────────────────────────────────
    
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

    table_trade_new_row.historical_high = df_res['high'].iloc[self.config.bank.complete_index:].max()   # added. * [-2 : 0] doesn't work.
    table_trade_new_row.historical_low = df_res['low'].iloc[self.config.bank.complete_index:].min()     # added.  
    
    table_trade_new_row.account = account
    table_trade_new_row.balance = balance_                 # added.  
    table_trade_new_row.balance_origin = balance_origin    # added.      
    
    table_trade_new_row.margin = 0    # margin 은 balance 와 다시 합해지는 것을 고려해 0 으로 초기화.
    
    
    if account_normality:                  
    
        # ──────────────────────────────────────────────
        # quantity.
        # ──────────────────────────────────────────────
        
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
            
            self.sys_log.debug(f"InitTableTrade : elasped time, {code} get_leverage : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
            # check leverage_limit_const
            leverage_limit_const_max = params.get('leverage_limit_const', 2)
            if leverage_limit_const < leverage_limit_const_max:
                
                # balance_ 와 margin 비교시 병렬 처리 중 margin 소모 대응 가능해진다.
                if balance_ > margin:    
                    
                    # set leverage
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                                        
                    self.set_leverage(symbol, leverage)
                        
                    self.sys_log.debug(f"InitTableTrade : elasped time, {code} set_leverage : %.4fs" % (time.time() - start_time))
                    self.sys_log.debug("------------------------------------------------")
                    
                    
                    # 거래 진행 시에만 margin 차감 (에러 발생 시 margin = 0)
                        # 종료 시 margin은 balance에 복구됨
                    table_trade_new_row.margin = margin                        
                    self.table_account.loc[self.table_account['account'] == account, 'balance'] -= margin
                    
                else:
                    table_trade_new_row.status = 'ERROR : balance_ <= margin'    # added.
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
        - rewrite json. 20250417 2044.
    v2.3
        - reorder phase. 20250417 2221.
        - log status_error state. 20250418 0018
        - remove stop_market annot. (expiry check 로 price_realtime 이 필요하기 때문에 사용하지 않습니다.) 20250418 1652.
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
            
            self.table_trade.loc[self.table_trade['code'] == code, 'order'] = 0 # init as 0.            
            
            try:
                # 코드 형식이 {symbol}_{interval}_{entry_timeIndex}라고 가정합니다.
                _, interval, entry_timeIndex, _ = code.split('_')
            except Exception as e:
                msg = f"Error parsing code {code}: {e}"
                self.sys_log.error(msg)
                self.push_msg(msg)                
                continue # hard assertion.
            

            # self.sys_log.debug("row : \n{}".format(row))
            # self.sys_log.debug("row.dtypes : {}".format(row.dtypes))            
            
                            
            # status_error = 1 if 'ERROR' in str(status_prev) else 0
            status_error = remove_row == 1 and pd.isnull(order_way)
            self.sys_log.debug(f"status_error : {status_error}")
            
            if status_error:
                # remove_row == 1로 실제로 거래가 종료되어 드롭되기 직전에 로깅
                self.table_log = pd.concat([self.table_log, self.table_trade[self.table_trade['code'] == code]], ignore_index=True)
                self.table_log['id'] = self.table_log.index + 1 # Update 'id' column to be the new index + 1                    
                self.db_manager.replace_table(self.table_log, self.table_log_name)
            
            
            # ──────────────────────────────────────────────
            # Broker 설정 & 초기 주문 처리
            # ──────────────────────────────────────────────

            is_null_id  = pd.isnull(orderId)
            need_broker_init = symbol not in self.price_market or is_null_id 

            if not status_error:
                if need_broker_init:
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    self.websocket_client.agg_trade(symbol=symbol, id=1, callback=self.agg_trade_message_handler)
                    self.set_position_mode(dualSidePosition='true')
                    self.set_margin_type(symbol=symbol)
                    
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set Broker-Ticker option : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")

                if is_null_id :
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    self.table_trade.loc[self.table_trade['code'] == code, 'order'] = 1
                    self.table_trade.loc[self.table_trade['code'] == code, 'order_way'] = 'OPEN'
                    
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set order, order_way (OPEN) : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")


            if not is_null_id:
                
                # get_order_info. (update order_info)                   
                order_info = get_order_info(self, 
                                        symbol,
                                        orderId)
                
                if order_info:
                    # order 이 있는 경우에만 진행하는 것들
                        # 1 update table_trade 
                        # 3 update remove_row, status
                        # 2 update statusChangeTime + table_logging.
                        # 4 update order, order_way
                        # 5 check Expiry / Stop Loss / Closer (OPEN / CLOSE order 있는 경우에만 진행하는 것들)
                    

                    # ──────────────────────────────────────────────
                    # update table_trade
                    # ──────────────────────────────────────────────
                    
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    for k, v in order_info.items():                        
                        self.table_trade.loc[self.table_trade['code'] == code, k] = v
                        
                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} update table_trade : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
                    # display(self.table_trade) 
                       
                        
                    
                    self.sys_log.debug(f"status_prev : {status_prev}, order_info['status'] : {order_info['status']}")
                    
                    # ──────────────────────────────────────────────
                    # set remove_row
                        # go to drop row & get_income_info status
                    # ──────────────────────────────────────────────
                    if (order_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']) or (row.order_way == 'CLOSE' and order_info['status'] == 'FILLED'):
                    
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()       
                        
                        # 부분 체결 후 취소된 경우는 제외하고, 나머지는 제거 플래그 설정 (remove_row)
                            # 1 CLOSE order_way 로 교체.
                            # 2 즉시 체결 안된 partial filled --> closer 처리, 이미 status_2 = expired 상태.
                        if (order_info['status'] == 'CANCELED' and float(order_info['executedQty']) > 0):
                            self.table_trade.loc[self.table_trade['code'] == code, 'order_way'] = 'CLOSE' 
                            closer_on = True                                                               
                        else:                        
                            self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'] = 1 # 아래에 로깅 구간이 있기 때문에, 로깅 불필요하다.
                        
                        
                        if order_info['status'] == 'FILLED' and order_info['type'] == 'LIMIT':
                            self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'TP'    
                            
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set remove_row : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                    
                    
                    
                    # ──────────────────────────────────────────────
                    # set statusChangeTime & table_logging.
                    # ──────────────────────────────────────────────
                    if status_prev != order_info['status']:                    
                    
                        # Set statusChangeTime 
                            # check if status has been changed.
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_trade.loc[self.table_trade['code'] == code, 'statusChangeTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")                                                
                        self.push_msg("{} status has been changed. {} {}".format(code, row.order_way, order_info['status']))
                        
                        self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set statusChangeTime : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")      
                        # display(self.table_trade)
                            
                            
                        
                        # Send row to table_log.
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
                        

                    
                    # ──────────────────────────────────────────────
                    # set order, order_way (for CLOSE)
                    # ──────────────────────────────────────────────
                    
                    if row.order_way == 'OPEN':         
                        
                        if (abs(float(order_info['executedQty']) / float(order_info['origQty'])) >= self.config.bank.quantity_open_exec_ratio) or order_info['status'] == 'FILLED':
                            
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()
                            
                            # LIMIT CLOSE 진행.
                            self.table_trade.loc[self.table_trade['code'] == code, 'order'] = 1
                            self.table_trade.loc[self.table_trade['code'] == code, 'order_way'] = 'CLOSE'
                            
                            # 미체결 잔량 존재 시 해당 주문 취소 후 진행.
                            if order_info['status'] != 'FILLED':                                 
                                quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                            symbol,
                                                                            orderId)  
                                self.sys_log.debug(f"{code} {orderId} quantity_unexecuted : {quantity_unexecuted}")                            
                            
                            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} set order, order_way (CLOSE) : %.4fs" % (time.time() - start_time)) 
                            self.sys_log.debug("------------------------------------------------")
                            # display(self.table_trade)
                    
                    

                    # ──────────────────────────────────────────────
                    # PHASE SUMMARY (order == 0 일 때)
                    # 1. 실시간 가격 조회 및 슬리피지 확인
                    # 2. Expiry 체크 (OPEN 상태 + 실시간 가격 정상 + order_info 있음)
                    # 3. Stop Loss 체크 (CLOSE 상태 + 가격 정상)
                    # 4. Closer 체크 (CLOSE 상태, 가격 무관)
                    # 5. MARKET 주문 실행 (Stop Loss 또는 Closer 조건 발생 시)
                    #  ──────────────────────────────────────────────
                    if self.table_trade.loc[self.table_trade['code'] == code, 'order'].values[0] == 0:

                        order_way_current = row.order_way

                        # ──────────────────────────────────────────────
                        # 1 실시간 가격 조회 및 슬리피지 확인
                        # ──────────────────────────────────────────────
                        price_realtime = get_price_realtime(self, symbol)
                        self.sys_log.debug(f"{symbol} price_realtime : {price_realtime}")

                        price_ok = False
                        if not pd.isnull(price_realtime):
                            price_realtime_prev = self.table_trade.loc[self.table_trade['code'] == code, 'price_realtime'].values[0]
                            slippage_margin = 0.1
                            lower_bound = price_realtime_prev * (1 - slippage_margin)
                            upper_bound = price_realtime_prev * (1 + slippage_margin)

                            self.sys_log.debug(f"price_realtime_prev: {price_realtime_prev:.6f}, lower_bound: {lower_bound:.6f}, upper_bound: {upper_bound:.6f}")

                            if pd.isnull(price_realtime_prev) or (lower_bound <= price_realtime <= upper_bound):
                                price_ok = True
                                self.table_trade.loc[self.table_trade['code'] == code, 'price_realtime'] = price_realtime


                        # ────────────────────────────────────────────── 
                        # 2 Expiry 체크 (OPEN 상태 + 실시간 가격 정상 + order_info 있음)
                        # ──────────────────────────────────────────────
                        if order_way_current == 'OPEN' and price_ok and order_info and order_info['status'] in ['NEW', 'PARTIALLY_FILLED']:
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()

                            price_historical = historical_high if side_position == 'LONG' else historical_low
                            expired = check_expiration(
                                self,
                                interval,
                                entry_timeIndex,
                                side_position,
                                price_realtime,
                                price_historical,
                                price_expiration,
                                ncandle_game=params.get('ncandle_game', 2)
                            )

                            if expired:
                                self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = f"Expired_{'x' if expired == 1 else 'y'}"
                                quantity_unexecuted = get_quantity_unexecuted(self, symbol, orderId)

                            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check expiration for order_open : %.4fs" % (time.time() - start_time))
                            self.sys_log.debug("------------------------------------------------")


                        # ──────────────────────────────────────────────
                        # 3 Stop Loss 체크 (CLOSE 상태 + 가격 정상)
                        # ──────────────────────────────────────────────
                        if order_way_current == 'CLOSE' and price_ok and (params.get('use_stop_loss', True) or params.get('use_stop_loss_barclose', True)):
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()

                            stop_loss_on = check_stop_loss(
                                self,
                                side_open,
                                price_realtime,
                                price_liquidation,
                                price_stop_loss
                            )

                            if stop_loss_on:
                                self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'SL'

                            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check_stop_loss : %.4fs" % (time.time() - start_time))
                            self.sys_log.debug("------------------------------------------------")


                        # ──────────────────────────────────────────────
                        # 4 Closer 체크 (CLOSE 상태, 가격 무관)
                        # ──────────────────────────────────────────────
                        if order_way_current == 'CLOSE' and params.get('use_closer', True):
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()

                            closer_on = check_closer(
                                self,
                                interval,
                                entry_timeIndex,
                                ncandle_game=params.get('ncandle_game', 2)
                            )

                            if closer_on:
                                self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'Closer'

                            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} check_closer : %.4fs" % (time.time() - start_time))
                            self.sys_log.debug("------------------------------------------------")


                        # ──────────────────────────────────────────────
                        # 5 MARKET 주문 실행 (Stop Loss 또는 Closer 조건 발생 시)
                        # ──────────────────────────────────────────────
                        if stop_loss_on or closer_on:
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()

                            quantity_unexecuted = get_quantity_unexecuted(self, symbol, orderId)
                            if quantity_unexecuted:
                                order_result, error_code = order_market(
                                    self,
                                    symbol,
                                    side_close,
                                    side_position,
                                    quantity_unexecuted
                                )
                                if not error_code:
                                    self.table_trade.loc[self.table_trade['code'] == code, 'orderId'] = order_result['orderId']

                            self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} order_market : %.4fs" % (time.time() - start_time))
                            self.sys_log.debug("------------------------------------------------")

                
                
                
            # order_info 가 꼭 유효하지 않아도 되는 구간.   
                
            # ──────────────────────────────────────────────
            # LIMIT 주문 실행 (OPEN / CLOSE 구분 처리)
            # ──────────────────────────────────────────────
            if self.table_trade.loc[self.table_trade['code'] == code, 'order'].values[0] == 1:

                order_way_current = self.table_trade.loc[self.table_trade['code'] == code, 'order_way'].values[0]
                side_position = row.side_position

                # === OPEN: order_info 불필요 ===
                if order_way_current == 'OPEN':
                    side_order = row.side_open
                    price = row.price_entry
                    quantity = row.quantity_open

                # === CLOSE: order_info 반드시 필요 ===
                elif order_way_current == 'CLOSE':
                    if order_info is None:
                        self.sys_log.error(f"[{code}] CLOSE 주문 실행 실패: order_info 없음")
                        continue  # 또는 적절한 fallback 처리

                    side_order = row.side_close
                    price = row.price_take_profit
                    quantity = order_info['executedQty']

                else:
                    self.sys_log.error(f"[{code}] 알 수 없는 order_way: {order_way_current}")
                    continue

                # === 주문 실행 ===
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()

                order_result, error_code = order_limit(
                    self,
                    symbol,
                    side_order,
                    side_position,
                    price,
                    quantity
                )

                if not error_code:
                    self.table_trade.loc[self.table_trade['code'] == code, 'orderId'] = order_result['orderId']
                else:
                    self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'] = 1
                    self.table_trade.loc[self.table_trade['code'] == code, 'status'] = f"ERROR : order_limit {error_code}"

                    # 오류 row → 로그 테이블로 이동
                    self.table_log = pd.concat([self.table_log, self.table_trade[self.table_trade['code'] == code]], ignore_index=True)
                    self.table_log['id'] = self.table_log.index + 1
                    self.db_manager.replace_table(self.table_log, self.table_log_name)

                    self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} replace_table table_log : %.4fs" % (time.time() - start_time))

                self.sys_log.debug(f"LoopTableTrade : elasped time, {code} {orderId} order_limit : %.4fs" % (time.time() - start_time))
                self.sys_log.debug("------------------------------------------------")
                    
                        
                           
            # ──────────────────────────────────────────────
            # DROP ROW & GET INCOME INFO
                # status_error case 를 고려해서 indent level 유지.
            # ──────────────────────────────────────────────
            
            if self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'].values[0] == 1:         
                           
                # 01 DROP ROW
                    # 이 drop 단계는 가장 마지막에 실행해야 합니다.
                    # 먼저 실행할 경우, 위쪽 phase에서 해당 row를 접근할 수 없고,
                    # orderId나 기타 필드가 NaN으로 설정되거나 오류가 발생할 수 있습니다.             
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



                # 02 GET INCOME INFO
                    # 취소된 OPEN 주문은 계산에 해당되지 않습니다.
                        # --> 해당 루프는 비동기화 상태에서 loop_condition 과 별도로 동작하기에, 클래스 인스턴스 ( self.table_log ? ) 로 접근해도 되지 않나.
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
                
                
                with open(self.path_config, 'w') as f:
                    json.dump(self.config, f, indent=4)                
                


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



