from bank import *
from idep import *
from params import * 


from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor

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
    v0.2.2
        - Safe loop: CancelledError, Exception 모두 while True 유지. 20250427 1400
        - modify time logging. 20250503 2053
    """
    batch_start_time = None
    # saving_count = 0

    # self.sys_log.debug(f"LoopSaveDfRes : START at {datetime.now().strftime('%H:%M:%S.%f')}")

    while True:
        try:
            if self.queue_df_res.qsize() > 0 and batch_start_time is None:
                batch_start_time = time.time()
                # saving_count = self.queue_df_res.qsize()

            try:
                item = await self.queue_df_res.get()
            except asyncio.CancelledError:
                self.sys_log.warning("LoopSaveDfRes : CancelledError during get(), continue loop")
                await asyncio.sleep(0.1)  # 너무 빠르게 도는 걸 방지
                continue

            try:
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
                        
                    except Exception as io_err:
                        self.sys_log.error(f"LoopSaveDfRes : File I/O error: {io_err}")
                else:
                    self.sys_log.warning(f"LoopSaveDfRes : Received None df_res for {symbol}-{interval}, skipping.")

            finally:
                self.queue_df_res.task_done()

            if self.queue_df_res.empty() and batch_start_time is not None:                
                batch_elapsed = time.time() - batch_start_time
                batch_start_time = None
                # saving_count = 0
                self.sys_log.debug(f"[loop_save_df_res_async] elapsed : {batch_elapsed:.4f}s")

        except asyncio.CancelledError:
            self.sys_log.warning("LoopSaveDfRes : CancelledError outer loop, continue loop")
            await asyncio.sleep(0.1)
            continue

        except Exception as e:
            self.sys_log.error(f"LoopSaveDfRes : General outer loop error: {e}")
            await asyncio.sleep(0.1)
            continue
       

async def update_table_condition_async(self,):
    
    """
    v0.1
        - init. 20250421 2229.
        - add try wrapper on exchange_info. 20250422 1442.
            - modify to asyncio. 20250422 1953.
    """
    
    # STEP 1: 실시간 심볼 정보 수집
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, self.exchange_info)  # 동기 → 비동기 안전 전환
        symbol_infos = result['data']['symbols']
    except Exception as e:
        self.sys_log.error(f"Error in exxhange_info : {e}")
        return self.table_condition  # 기존 상태 유지 or empty 반환
    
    symbols = [
        info['symbol'] for info in symbol_infos
        if info.get("contractType") == "PERPETUAL" and info['symbol'].endswith("USDT")
    ]

    # STEP 2: 전체 interval 조합
    intervals = ['15m', '30m', '1h', '2h', '4h']
    symbol_interval_pairs = list(itertools.product(symbols, intervals))
    table_new = pd.DataFrame(symbol_interval_pairs, columns=["symbol", "interval"])
    
    # --- dtype 명시적 선언 ---
    dtypes = {
        'id': 'int64',
        'symbol': 'object',
        'side_open': 'float64',
        'price_take_profit': 'float64',
        'price_entry': 'float64',
        'price_stop_loss': 'float64',
        'leverage': 'float64',
        'priceBox_indicator': 'float64',
        'priceBox_value': 'float64',
        'point_mode': 'float64',
        'point_indicator': 'float64',
        'point_value': 'float64',
        'zone_indicator': 'float64',
        'zone_value': 'float64',
        'position': 'float64',
        'interval': 'object',
        'RRratio': 'float64',
        'timestampLastUsed': 'int64',
        'divider': 'int64',
        'account': 'object'
    }

    # STEP 3: 기본값 설정 (timestampLastUsed는 merge로 유지)
    defaults = {
        "side_open": np.nan, "price_take_profit": np.nan, "price_entry": np.nan, "price_stop_loss": np.nan,
        "leverage": np.nan, "priceBox_indicator": np.nan, "priceBox_value": np.nan,
        "point_mode": np.nan, "point_indicator": np.nan, "point_value": np.nan,
        "zone_indicator": np.nan, "zone_value": np.nan, "position": np.nan, "RRratio": np.nan,
        "divider": self.config.bank.divider, "account": self.config.bank.account
    }

    for col, val in defaults.items():
        table_new[col] = val

    # STEP 4: 기존 테이블이 없다면 그대로 사용
    if not hasattr(self, 'table_condition') or self.table_condition.empty:
        table_new['timestampLastUsed'] = 0
        table_new.insert(0, "id", range(1, len(table_new) + 1))
        return table_new.astype(dtypes)

    # STEP 5: 기존 table_condition과 merge → timestampLastUsed 유지
    existing = self.table_condition[['symbol', 'interval', 'timestampLastUsed']]
    table_new = pd.merge(table_new, existing, on=['symbol', 'interval'], how='left')
    table_new['timestampLastUsed'] = table_new['timestampLastUsed'].fillna(0)

    # STEP 6: 기존 table_condition에서 사라진 심볼 제거
    existing_keys = set(zip(self.table_condition['symbol'], self.table_condition['interval']))
    updated_keys = set(zip(table_new['symbol'], table_new['interval']))
    removed_keys = existing_keys - updated_keys

    if removed_keys:
        self.table_condition = self.table_condition[
            ~self.table_condition[['symbol', 'interval']].apply(tuple, axis=1).isin(removed_keys)
        ]

    # STEP 7: 최종 테이블 덮어쓰기
    table_new.insert(0, "id", range(1, len(table_new) + 1))
    
    return table_new.astype(dtypes)
                
        
async def loop_table_condition_async(self, drop=False, debugging=False):
    """
    v2.4.3
        - adj. v2.1.5, 해당 버전은 OPEN 하고 save_df_res 하니까. 20250503 2151.
            - init 까지는 빠르게 진행. 20250503 2237.
    v2.4.4
        - rollback v2.1.5's 일괄 처리. 20250507 2001.
            - 1h 까지 36초.
            - 15m 까지 20초
        - add len(df) > 2 condition. 20250509 2321.
    v3.0
        - adj. asyncio funcs. 20250510 2031.
    """

    data_len = 100
    kilne_limit = 100

    while True:
        
        if self.table_condition.empty:
            await asyncio.sleep(1)
            continue
        
        loop_start = time.time()
        # self.sys_log.debug(f"[loop_table_condition] START at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        
        # STEP 0: 실시간 ticker 기반으로 table_condition 갱신
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        self.table_condition = await update_table_condition_async(self)
    
        self.sys_log.debug(f"LoopTableCondition : elapsed time, update_table_condition_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
        

        # STEP 1: 수집 대상 필터링 및 fetch_df 준비        
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()      
        
        job_list = []  # 유효한 작업 후보 저장          
        timestamp_current = int(time.time()) # 모두 동일한 시작 timestamp 기준. --> loop 나눠서 진행하는 것을 방지한다.
        
        for idx, row in self.table_condition.iterrows():
            # symbol = row.symbol
            interval = row.interval
            interval_number = itv_to_number(interval)
            interval_sec = interval_number * 60
            timestamp_last = int(row.timestampLastUsed)

            # interval 중복 방지
            if timestamp_current - timestamp_last < interval_sec:
                # self.sys_log.debug(f"[SKIP] {row.symbol} {interval} / last: {timestamp_last}, now: {timestamp_current}")
                continue
            else:
                timestamp_anchor = 1718236800  # 기준 anchor, '2024-06-13 09:00:00'
                timestamp_remain = (timestamp_current - timestamp_anchor) % interval_sec
                timestamp_current -= timestamp_remain
                self.table_condition.at[idx, 'timestampLastUsed'] = timestamp_current

                days = data_len / (60 * 24 / interval_number)
                job_list.append((idx, row, days))
                
        self.sys_log.debug(f"LoopTableCondition : elapsed time, check timestampLastUsed : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
                
 
        # STEP 2: df 수집 (get_df_new_async)
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()      
        
        semaphore = asyncio.Semaphore(32)  # API 제한 고려
        tasks_step1 = []
        
        await self.token_bucket.wait_for_tokens(len(job_list))
        
        async with aiohttp.ClientSession() as session:
            async def fetch_task(idx_, symbol_, interval_, days_):
                await asyncio.sleep(0.1)
                async with semaphore:
                    return idx_, await get_df_new_async(self, session, symbol_, interval_, days_, limit=kilne_limit)

            tasks_step1 = [
                fetch_task(idx, row.symbol, row.interval, days)
                for idx, row, days in job_list
            ]
            df_results = await asyncio.gather(*tasks_step1)
        
        self.sys_log.debug(f"LoopTableCondition : elapsed time, get_df_new_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")


        # STEP 3: get_trade_info_sync → run_in_executor 사용한 동기 함수의 비동기 병렬 처리)
            # ProcessPoolExecutor 사용 시, pure function 보장해야한다.
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
            if isinstance(df_res, pd.DataFrame) and not df_res.empty and len(df_res) > 1
        ]

        trade_infos = await asyncio.gather(*tasks_step2)        
        
        self.sys_log.debug(f"LoopTableCondition : elapsed time, get_trade_info_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
        
        
        # STEP 4: init_table_trade_async (position 존재하는 경우만)
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        tasks_step3 = [
            init_table_trade_async(self, trade_info)
            for trade_info in trade_infos if trade_info.position
        ]
        await asyncio.gather(*tasks_step3)
        
        self.sys_log.debug(f"LoopTableCondition : elapsed time, init_table_trade_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")      
        
                
        # STEP 5: enqueue_df_res_async
            # enqueue_df_res_async 는 consumer 가 따라오지 못할 가능성 때문에 직렬로 설정.
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        for ti in trade_infos:
            if ti and isinstance(ti.df_res, pd.DataFrame):
                await enqueue_df_res_async(self, ti.df_res, ti.symbol, ti.interval)
                
        self.sys_log.debug(f"LoopTableCondition : elapsed time, enqueue_df_res_async : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
        
        
        # STEP 6: DB 반영
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        async with self.lock_trade:
            self.db_manager.replace_table(
                self.table_condition,
                self.table_condition_name,
                send=self.config.database.usage
            )
            
        self.sys_log.debug(f"LoopTableCondition : elapsed time, replace_table condition (send={self.config.database.usage}) : %.4fs" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
        

        loop_elapsed_time = time.time() - loop_start
        self.sys_log.debug(f"[loop_table_condition] elapsed : {loop_elapsed_time:.4f}s")
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
    
    loop_start = time.time()
    
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

        self.sys_log.debug(f"GetTradeInfo : elapsed time, {symbol} {interval} get_indicators : {time.time() - start_time:.4f}s")
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
        self.sys_log.debug(f"GetTradeInfo : elapsed time, {symbol} {interval} get_price_channel & opener : {time.time() - start_time:.4f}s")
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

        self.sys_log.debug(f"GetTradeInfo : elapsed time, {symbol} {interval} get_signal : {time.time() - start_time:.4f}s")
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

        self.sys_log.debug(f"GetTradeInfo : elapsed time, {symbol} {interval} get_price_box : {time.time() - start_time:.4f}s")
        self.sys_log.debug("------------------------------------------------")

    except Exception as e:
        self.sys_log.error(f"Error in get_trade_info_sync for {symbol}: {e}")
        
    # self.sys_log.debug(f"[get_trade_info_sync] elapsed : {time.time() - loop_start:.2f}s")

    return TradeInfo(
        symbol, side_open, position, interval, divider, account,
        df_res, code,
        price_expiration, price_take_profit, price_entry, price_stop_loss
    )


async def init_table_trade_async(self, trade_info: TradeInfo):
    
    """    
    v2.6.3
        - ban Open Expired_x. 20250427 1352.
        - rollback to use latest_index in historical high / low . 20250501 1851.
    v3.0
        - adj. asyncio funcs. 20250510 1801.
    """
    
    
    loop_start = time.time()
    # self.sys_log.debug(f"[init_table_trade] START at {datetime.now().strftime('%H:%M:%S.%f')}")
           
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
    # amount_min = 5
    
    
    # ──────────────────────────────────────────────
    # Get side.
    # ──────────────────────────────────────────────
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
        
    side_close, \
    side_position = get_side_info(self, 
                                    side_open)
        
    self.sys_log.debug(f"InitTableTrade : elapsed time, {code} get_side_info : %.4fs" % (time.time() - start_time)) 
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
    
    self.sys_log.debug(f"InitTableTrade : elapsed time, {code} get_price_entry : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
        

    # get price_liquidation
        # get_price_liquidation requires leverage.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   

    # we are using target_loss_pct now, so liquidation has no more meaning.
        # target_loss_pct max is 100.
    price_liquidation = price_stop_loss
    
    self.sys_log.debug(f"InitTableTrade : elapsed time, {code} get_price_liquidation : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    # # get price_expiration
    # self.sys_log.debug("------------------------------------------------")
    # start_time = time.time()  
    
    # price_expiration = price_take_profit # temporarily fix to price_take_profit.
    
    # self.sys_log.debug(f"InitTableTrade : elapsed time, {code} get_price_expiration : %.4fs" % (time.time() - start_time)) 
    # self.sys_log.debug("------------------------------------------------")

    
    # adj. precision_price & quantity
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()    
        
    precision_price, \
    precision_quantity = await get_precision(self, 
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
    
    self.sys_log.debug(f"InitTableTrade : elapsed time, {code} adj. precision_price : %.4fs" % (time.time() - start_time)) 
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
    balance_available = await get_account_normality(self, account)
    
    self.sys_log.debug("account_normality : {}".format(account_normality))
    self.sys_log.debug("balance_ : {}".format(balance_))
    self.sys_log.debug("balance_origin : {}".format(balance_origin))
    self.sys_log.debug("balance_available : {}".format(balance_available))
    
    self.sys_log.debug(f"InitTableTrade : elapsed time, {code} get_account_normality : %.4fs" % (time.time() - start_time)) 
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

    table_trade_new_row.historical_high = df_res['high'].iloc[self.config.bank.latest_index]   # added. * [-2 : 0] doesn't work.
    table_trade_new_row.historical_low = df_res['low'].iloc[self.config.bank.latest_index]     # added.  
    
    table_trade_new_row.account = account
    table_trade_new_row.balance = balance_                 # added.  
    table_trade_new_row.balance_origin = balance_origin    # added.      
    
    # table_trade_new_row.margin = 0            # margin 은 balance 와 다시 합해지는 것을 고려해 0 으로 초기화.
    table_trade_new_row.statusChangeTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")      # balance 순차 소비 확인을 위해 (sys_log 와 sync.) 실시간으로 초기화.
    
    
    if account_normality:         
        
        # Expiration 사전 체크 (historical 가격 기준 만료 검사)
        expired = 0
        
        if side_position == 'LONG':
            if (table_trade_new_row.historical_high.values[0]) >= price_expiration:
                expired = 1
        else:
            if (table_trade_new_row.historical_low.values[0]) <= price_expiration:
                expired = 1


        if expired != 1:    
                
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
            
            self.sys_log.debug(f"InitTableTrade : elapsed time, {code} get margin : %.4fs" % (time.time() - start_time)) 
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
                
                self.sys_log.debug(f"InitTableTrade : elapsed time, {code} get_leverage : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")
                
                
                # check leverage_limit_const
                leverage_limit_const_max = params.get('leverage_limit_const', 5)
                if leverage_limit_const < leverage_limit_const_max:
                    
                    # balance_ 와 margin 비교시 병렬 처리 중 margin 소모 대응 가능해진다.
                    if balance_ > margin:    
                        
                        # set leverage
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                                            
                        await self.set_leverage(symbol, leverage)
                            
                        self.sys_log.debug(f"InitTableTrade : elapsed time, {code} set_leverage : %.4fs" % (time.time() - start_time))
                        self.sys_log.debug("------------------------------------------------")
                        
                        
                        # 거래 진행 시에만 margin 차감 (에러 발생 시 margin = 0)
                            # 종료 시 margin은 balance에 복구됨                     
                        self.table_account.loc[self.table_account['account'] == account, 'balance'] -= margin
                        
                    else:
                        table_trade_new_row.status = 'ERROR : balance < margin'    # added.
                        table_trade_new_row.remove_row = 1         
                        
                else: 
                    table_trade_new_row.status = f"ERROR : leverage_limit_const > {leverage_limit_const_max}"
                    table_trade_new_row.remove_row = 1
                
            else:
                table_trade_new_row.status = 'ERROR : amount_entry < broker.amount_min'    # added.
                table_trade_new_row.remove_row = 1
                
        else:
            table_trade_new_row.status = 'ERROR : Open Expired_x'   # added.
            table_trade_new_row.status_2 = 'Open Expired_x'
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
    
    # self.sys_log.debug(f"InitTableTrade : elapsed time, {code} replace_table (send={self.config.database.usage})  : %.4fs" % (time.time() - start_time)) 
    # self.sys_log.debug("------------------------------------------------")
    
    # self.sys_log.debug(f"[init_table_trade] elapsed : {time.time() - loop_start:.2f}s")
  
     
async def enqueue_df_res_async(self, df_res, symbol, interval):
    """
    v0.1
        df_res 저장용 항목을 queue_df_res에 enqueue. 20250503 2042.
    """
    
    loop_start = time.time()
    
    try:
        save_item = {
                "df": df_res,
                "symbol": symbol,
                "interval": interval
            }
        await self.queue_df_res.put(save_item)
        self.sys_log.debug(f"EnqueueDfRes : Item queued : {symbol} {interval}")
    except Exception as e:
        self.sys_log.error(f"EnqueueDfRes : Queue error : {e}")
        
    # self.sys_log.debug(f"[enqueue_df_res] elapsed : {time.time() - loop_start:.2f}s")
    
 
 
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
        - modify order_market condi. 20250420 2045.
        - if status_error, skip get_income_info. 20250426 2029.
        - modify table_log to be sorted by statusChangeTime. 20250426 2136.
            - log every init status. (null_orderId) 20250427 0836.
    v2.3.1
        - modify websocket agg_trade to kline. 20250502 2047.
        - check init state by candle_cache. 20250503 0154.
        - add status_error push_msg. 20250506 1017.
    """
    
    
    while True:
        
        loop_start = time.time()
        # self.sys_log.debug(f"[loop_table_trade]     START at {datetime.now().strftime('%H:%M:%S.%f')}")
        
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
            
            # historical_high =  row.historical_high
            # historical_low =  row.historical_low

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
            if status_error:
                msg = f"{code} status_error : {status_prev}"
                self.sys_log.debug(msg)
                self.push_msg(msg)
            
            
            # ──────────────────────────────────────────────
            # Broker 설정 & 초기 주문 처리
            # ──────────────────────────────────────────────

            is_null_id  = pd.isnull(orderId)
            need_broker_init = (
                not hasattr(self, 'candle_cache') or
                symbol not in self.candle_cache or
                is_null_id
            )
            
            
            # balance 변화 추이를 위해 모든 init status 로깅.
                # is_null_id = init 된 상태.
            # if status_error:
            if is_null_id:
                self.table_log = pd.concat([self.table_log, self.table_trade[self.table_trade['code'] == code]], ignore_index=True)                
                self.table_log = self.table_log.sort_values('statusChangeTime').reset_index(drop=True)      # 추가: statusChangeTime 기준 정렬
                self.table_log['id'] = self.table_log.index + 1                                             # Update 'id' column to be the new index + 1                    
                self.db_manager.replace_table(self.table_log, self.table_log_name)
                

            if not status_error:
                if need_broker_init:
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    # self.websocket_client.agg_trade(symbol=symbol, id=1, callback=self.agg_trade_message_handler)
                    self.websocket_client.kline(symbol=symbol, id=idx + 1, interval=interval, callback=self.kline_message_handler)
                    await self.set_position_mode(dualSidePosition='true')
                    await self.set_margin_type(symbol=symbol)
                    
                    self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} set Broker-Ticker option : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")

                if is_null_id :
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    self.table_trade.loc[self.table_trade['code'] == code, 'order'] = 1
                    self.table_trade.loc[self.table_trade['code'] == code, 'order_way'] = 'OPEN'
                    
                    self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} set order, order_way (OPEN) : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")


            if not is_null_id:
                
                # get_order_info. (update order_info)                   
                order_info = await get_order_info(self, symbol, orderId)
                
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
                        
                    self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} update table_trade : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
                    # display(self.table_trade) 
                       
                        
                    
                    self.sys_log.debug(f"status_prev : {status_prev}, order_info['status'] : {order_info['status']}")
                    
                    # ──────────────────────────────────────────────
                    # set remove_row, trade end.
                        # if remove_row, directly go to drop row & get_income_info status after logged.
                    # ──────────────────────────────────────────────
                    if (order_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']) or (row.order_way == 'CLOSE' and order_info['status'] == 'FILLED'):
                    
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()       
                        
                        # 부분 체결 후 취소된 경우는 제외하고, 나머지는 제거 플래그 설정 (remove_row)
                            # partial_filled + expired --> closer 처리 (어찌되었든 거래 종료는 시켜야하니까.), 
                            # status_2 = expired --> ERROR 처리.
                        if (order_info['status'] == 'CANCELED' and float(order_info['executedQty']) > 0):  # PARTIALLY_FILLED 는 현 구간에 진입 불가, executedQty 로 확인
                            self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'ERROR : not enough executedQty'
                            closer_on = True   
                        else:
                            self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'] = 1 # 아래에 로깅 구간이 있기 때문에, 로깅 불필요하다.
                        
                        if order_info['status'] == 'FILLED' and order_info['type'] == 'LIMIT':
                            self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'TP'                                
                        
                        self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} set remove_row : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                    
                    
                    
                    # ──────────────────────────────────────────────
                    # set statusChangeTime / update table_log / set PARTIALLY_FILLED Closer.
                    # ──────────────────────────────────────────────
                    if status_prev != order_info['status']:                    
                    
                        # Set statusChangeTime 
                            # check if status has been changed.
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_trade.loc[self.table_trade['code'] == code, 'statusChangeTime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        self.push_msg("{} status has been changed. {} {}".format(code, row.order_way, order_info['status']))
                        
                        self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} set statusChangeTime : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")      
                        # display(self.table_trade)
                            
                            
                        
                        # Send row to table_log.
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_log = pd.concat([self.table_log, self.table_trade[self.table_trade['code'] == code]], ignore_index=True)  
                        self.table_log = self.table_log.sort_values('statusChangeTime').reset_index(drop=True)      # 추가: statusChangeTime 기준 정렬
                        self.table_log['id'] = self.table_log.index + 1 # Update 'id' column to be the new index + 1
                        
                        # 추후 도입 예정
                        if len(self.table_log) > 10000:
                            self.table_log = self.table_log.iloc[-7000:].copy()
                        
                        self.db_manager.replace_table(self.table_log, self.table_log_name)
                        
                        self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} replace_table table_log : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")  
                        # display(self.table_trade)
                                                

                    
                    # ──────────────────────────────────────────────
                    # check CLOSE LIMIT.
                    # ──────────────────────────────────────────────
                    
                    if row.order_way == 'OPEN':       
                        # 해당 조건 전 가능 status : NEW / PARTIALLY_FILLED
                        if (abs(float(order_info['executedQty']) / float(order_info['origQty'])) >= self.config.bank.quantity_open_exec_ratio) or order_info['status'] == 'FILLED':
                            
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()
                            
                            # LIMIT CLOSE 진행.
                            self.table_trade.loc[self.table_trade['code'] == code, 'order'] = 1
                            self.table_trade.loc[self.table_trade['code'] == code, 'order_way'] = 'CLOSE'
                            
                            # 주문 실패 - 미체결 잔량 존재 시 해당 주문 취소 후 CLOSE LIMIT 진행.
                            if order_info['status'] != 'FILLED':  # = PARTIALLY_FILLED          
                                if await order_cancel(self, symbol, orderId):                                
                                    order_info = await get_order_info(self, symbol, orderId) # update order_info for executedQty on CLOSE LIMIT.
                                    self.sys_log.debug(f"{code} {orderId} order_info['executedQty'] : {order_info['executedQty']}")         
                            
                            self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} check CLOSE LIMIT : %.4fs" % (time.time() - start_time)) 
                            self.sys_log.debug("------------------------------------------------")
                            # display(self.table_trade)
                    
                    

                    # ──────────────────────────────────────────────
                    # Exit Phase Summary. (order == 0 일 때)
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
                        price_high, price_low = get_price_realtime(self, symbol, interval)
                        self.sys_log.debug(f"{symbol} price_high, price_low : {price_high}, {price_low}")

                        price_ok = False
                        if not pd.isnull(price_high):
                            price_high_prev = self.table_trade.loc[self.table_trade['code'] == code, 'price_high'].values[0]
                            
                            slippage_margin = 0.1
                            lower_bound = price_high_prev * (1 - slippage_margin)
                            upper_bound = price_high_prev * (1 + slippage_margin)
                            self.sys_log.debug(f"price_high_prev: {price_high_prev:.6f}, lower_bound: {lower_bound:.6f}, upper_bound: {upper_bound:.6f}")

                            if pd.isnull(price_high_prev) or (lower_bound <= price_high <= upper_bound):
                                price_ok = True
                                self.table_trade.loc[self.table_trade['code'] == code, 'price_high'] = price_high
                                self.table_trade.loc[self.table_trade['code'] == code, 'price_low'] = price_low


                        # ────────────────────────────────────────────── 
                        # 2 Expiry 체크 (OPEN 상태 + 실시간 가격 정상 + order_info 있음)
                        # ──────────────────────────────────────────────
                        if order_way_current == 'OPEN' and price_ok and order_info and order_info['status'] in ['NEW', 'PARTIALLY_FILLED']:
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()

                            price_expiration_check = price_high if side_position == 'LONG' else price_low                            
                            expired = check_expiration(
                                self,
                                interval,
                                entry_timeIndex,
                                side_position,
                                price_expiration_check,
                                price_expiration,
                                ncandle_game=params.get('ncandle_game', 2)
                            )
                            
                            if expired:
                                await order_cancel(self, symbol, orderId) # just cancel. no need for quantity_unexecuted
                                # _ = get_quantity_unexecuted(self, symbol, orderId)
                                self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = f"Expired_{'x' if expired == 1 else 'y'}"

                            self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} check expiration for order_open : %.4fs" % (time.time() - start_time))
                            self.sys_log.debug("------------------------------------------------")


                        # ──────────────────────────────────────────────
                        # 3 Stop Loss 체크 (CLOSE 상태 + 가격 정상)
                        # ──────────────────────────────────────────────
                        if order_way_current == 'CLOSE' and price_ok and (params.get('use_stop_loss', True) or params.get('use_stop_loss_barclose', True)):
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()

                            price_stop_loss_check = price_low if side_position == 'LONG' else price_high
                            stop_loss_on = check_stop_loss(
                                self,
                                side_open,
                                price_stop_loss_check,
                                price_liquidation,
                                price_stop_loss
                            )

                            if stop_loss_on:
                                self.table_trade.loc[self.table_trade['code'] == code, 'status_2'] = 'SL'

                            self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} check_stop_loss : %.4fs" % (time.time() - start_time))
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

                            self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} check_closer : %.4fs" % (time.time() - start_time))
                            self.sys_log.debug("------------------------------------------------")


                        # ──────────────────────────────────────────────
                        # 5 MARKET 주문 실행 (Stop Loss 또는 Closer 조건 발생 시)
                            # remove_row != 1, (이 조건을 달지 않을 경우, order_market 을 반복할 수 있다.)
                                # status Update 를 위에서 하기 때문에.
                        # ──────────────────────────────────────────────
                        if self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'].values[0] != 1:
                            if stop_loss_on or closer_on:
                                self.sys_log.debug("------------------------------------------------")
                                start_time = time.time()

                                quantity_unexecuted = await get_quantity_unexecuted(self, symbol, orderId)
                                
                                # 잔여 미체결랑 처리
                                if quantity_unexecuted:
                                    order_result, error_code = await order_market(
                                        self,
                                        symbol,
                                        side_close,
                                        side_position,
                                        quantity_unexecuted
                                    )
                                    if not error_code:
                                        self.table_trade.loc[self.table_trade['code'] == code, 'orderId'] = order_result['orderId'] # return to order_info update.

                                self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} order_market : %.4fs" % (time.time() - start_time))
                                self.sys_log.debug("------------------------------------------------")

                
                
                
            # order_info 가 꼭 유효하지 않아도 되는 구간.   
                
            # ──────────────────────────────────────────────
            # LIMIT 주문 실행 (OPEN / CLOSE 구분 처리)
            # ──────────────────────────────────────────────
            if self.table_trade.loc[self.table_trade['code'] == code, 'order'].values[0] == 1:
                
                # === 주문 실행 ===
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
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
                        self.sys_log.error(f"[{code}] CLOSE order_info = {order_info}.")
                        continue  # 또는 적절한 fallback 처리

                    side_order = row.side_close
                    price = row.price_take_profit
                    quantity = order_info['executedQty']

                else:
                    self.sys_log.error(f"[{code}] order_way = {order_way_current}")
                    continue

                order_result, error_code = await order_limit(
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
                    # Closer 로도 Exit 는 되니까, 일단은 CLOSE 에 대한 에러 처리 대기.
                    self.table_trade.loc[self.table_trade['code'] == code, 'remove_row'] = 1
                    self.table_trade.loc[self.table_trade['code'] == code, 'status'] = f"ERROR : order_limit {error_code}"

                    # 오류 row → 로그 테이블로 이동, remove_row =1 로 바로 drop 하기 때문에.
                    self.table_log = pd.concat([self.table_log, self.table_trade[self.table_trade['code'] == code]], ignore_index=True)  
                    self.table_log = self.table_log.sort_values('statusChangeTime').reset_index(drop=True)      # 추가: statusChangeTime 기준 정렬
                    self.table_log['id'] = self.table_log.index + 1
                    self.db_manager.replace_table(self.table_log, self.table_log_name)

                self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} order_limit : %.4fs" % (time.time() - start_time))
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

                self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} drop rows : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")  

                

                if not status_error:
                    
                    # 02 GET INCOME INFO
                        # 취소된 OPEN 주문은 계산에 해당되지 않습니다.
                            # --> 해당 루프는 비동기화 상태에서 loop_condition 과 별도로 동작하기에, 클래스 인스턴스 ( self.table_log ? ) 로 접근해도 되지 않나.
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    income, \
                    self.income_accumulated, \
                    profit, \
                    self.profit_accumulated = await get_income_info(self, 
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
                    
                    self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} get_income_info : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")  
                    # display(self.table_trade)
                    
                    
                    with open(self.path_config, 'w') as f:
                        json.dump(self.config, f, indent=4)                
                


            # Prevent losing orderId in every loop.      
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            self.db_manager.replace_table(self.table_account, self.table_account_name)
            self.db_manager.replace_table(self.table_trade, self.table_trade_name, mode=['UPDATE', 'DELETE'])
            
            self.sys_log.debug(f"LoopTableTrade : elapsed time, {code} {orderId} replace_table account & trade : %.4fs" % (time.time() - start_time)) 
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
        
        # self.sys_log.debug(f"LoopTableTrade : elapsed time, replace_table account & trade & log (send={self.config.database.usage}) : %.4fs" % (time.time() - start_time)) 
        # self.sys_log.debug("------------------------------------------------")
        
        loop_elapsed_time = time.time() - loop_start
        self.sys_log.debug(f"[loop_table_trade] elapsed : {loop_elapsed_time:.4f}s")
        
        if self.table_trade.empty and self.queue_trade_init.empty():
            await asyncio.sleep(max(0, 1 - loop_elapsed_time))
        # await asyncio.sleep(1)



