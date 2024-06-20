import numpy as np
import pandas as pd
import time
from datetime import datetime

from funcs.public.indicator_ import *
from funcs.public.broker import itv_to_number


def loop_messenger(self,):

    """
    v1.0
        init
            consider dtypes in payload (float & str)
                default type = str.
        
    last confirmed at, 20240530 0808.
    """
    
    if self.user_text is not None:

        # default msg = invalid payload.
        msg = "error in self.user_text : invalid payload."
        user_text_upper_case = self.user_text.upper()
        payload = user_text_upper_case.replace('  ', ' ').split(' ')  # allow doubled-space.
        # payload_ = user_text_upper_case.replace('  ', ' ').split('/')  # allow doubled-space.

        payload_len = len(payload)

        if payload_len == 2:

            # asset change
            if 'ASSET' in user_text_upper_case:
                self.balance_available = int(self.user_text.split(' ')[-1])
                user_text_upper_case = 'WATCH'

        # set user_data
        elif payload_len == 6:
            
            symbol, side_open, price_take_profit, price_entry, price_stop_loss, leverage = payload
            symbol += 'USDT'

            self.get_tickers(self,)

            # validation
                # symbol
            if symbol in self.tickers_available:
                
                # side_open
                if side_open in ['BUY', 'SELL']:

                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()  
                    table_condition_new_row = pd.DataFrame(data = np.array([np.nan] * len(self.table_condition.columns)).reshape(-1, 1).T, columns=self.table_condition.columns, dtype=object)
                
                    # init table_condition.
                    table_condition_new_row.symbol = symbol 
                    table_condition_new_row.side_open = side_open 
                    table_condition_new_row.price_take_profit = float(price_take_profit)
                    table_condition_new_row.price_entry = float(price_entry)
                    table_condition_new_row.price_stop_loss = float(price_stop_loss)
                    table_condition_new_row.leverage = float(leverage)
                    
                    # append row.
                    self.table_condition = pd.concat([self.table_condition, table_condition_new_row]).reset_index(drop=True)        
                    self.sys_log.debug("InitTableCondition : elasped time, append row : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
                    
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()    
                    self.set_tables(self, mode='CONDITION')
                    self.sys_log.debug("InitTableCondition : elasped time, set_tables  : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")  
                    
                    user_text_upper_case = 'WATCH'
                                    
        # watch
        if 'WATCH' in user_text_upper_case:
            msg = "{}".format(self.table_condition)


        # show message.
        self.push_msg(self, msg)


        # reset user_text.
        self.user_text = None






def loop_table_condition(self, drop=False):

    """
    v1.0
        modify to data_table mode.
        update with one trade in a minute.
        add push_msg for initTableTrade.
        modify code name to code_{} using datetime.
        remove realtime_term for this loop
            term already exist in get_new_df
    v1.2
        add drop param.
    v1.3
        use messenger mode.
            add user_data. (symbol & price ...)
        add condition
    v1.4
        modify to TableCondition_v0.3 verison.
        modify drop to False. 
        remove elapsed time +,
    v1.5
        add column 'timestampLastUsed' in tableCondition.
            apply remaining minutes to zero.
        add timeframe_value (days)
        
    last confirmed at, 20240615 0644.
    """

    start_time_loop = time.time()
    for idx, row in self.table_condition.iterrows(): # idx can be used, cause table_condition's index are not reset.

        # init.        
        self.table_condition_row = row
           
        self.symbol = row.symbol
        self.side_open = row.side_open
        # self.price_take_profit = row.price_take_profit
        # self.price_entry = row.price_entry
        # self.price_stop_loss = row.price_stop_loss
        # self.leverage = row.leverage
        self.position = row.position
        self.timeframe = row.timeframe
        self.timeframe_value = row.timeframe_value
        self.priceBox = row.priceBox
        self.priceBox_value = row.priceBox_value
        self.point = row.point
        self.point_value = row.point_value
        self.zone = row.zone
        self.zone_value = row.zone_value
        

        # do not repeat it in interval.
        interval_number = itv_to_number(self.timeframe)
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

            # maximum residential duration for TableCondition.
                # to proceed TableTrade.
            if time.time() - start_time_loop > self.config.trader_set.loop_duration:
                self.sys_log.warning("LoopTableCondition : break by loop_duration {}s.".format(self.config.trader_set.loop_duration))
                break
            self.sys_log.debug("LoopTableCondition : elasped time, check timestampLastUsed : {:.2f}s".format(time.time() - start_time))
            self.sys_log.debug("------------------------------------------------")
        

        # get df_res
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        if self.config.trader_set.backtrade:
            # self.get_streamer(self)
            self.get_df_new_by_streamer(self)
        else:
            self.get_df_new(self, 
                            interval=self.timeframe, 
                            days=self.timeframe_value, 
                            timesleep=0.1)
            
            # prevent delaying by last_index_watch validation from loop_duration.
            if time.time() - start_time > self.config.trader_set.loop_duration:
                start_time = time.time()
     
        self.sys_log.info("LoopTableCondition : check df_res last row timestamp : {}".format(self.df_res.index[-1]))
        self.sys_log.debug("LoopTableCondition : elasped time, get_df_new : {:.2f}s".format(time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
 
        if self.df_res is None:
            continue     
            

        # # set timeframe (1)     
        # self.sys_log.debug("------------------------------------------------")
        # start_time = time.time()  
        # self.sys_log.debug("LoopTableCondition : elasped time, set timeframe : %.4fs" % (time.time() - start_time)) 
        # self.sys_log.debug("------------------------------------------------")


        
        # set point (3)   
            # set side_open considered. (set priceBox automation require side_open)
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()     
        self.df_res = get_II(self.df_res, period=self.point_value)
        self.sys_log.debug("{}".format(self.df_res.tail()))
    
        iiSource = self.df_res.iiSource.to_numpy()
        iiSource_back_1 = self.df_res.iiSource.shift(1).to_numpy()
        # iiSource_back_2 = self.df_res.iiSource.shift(2).to_numpy()
        
        cross_over = np.where((iiSource > 0) & (0 > iiSource_back_1), 1, 0) # LONG
        cross_under = np.where((iiSource < 0) & (0 < iiSource_back_1), 1, 0) # SHORT  
        
        # point_long_index = np.argwhere(cross_over).ravel()
        # point_short_index = np.argwhere(cross_under).ravel() 
        
        self.sys_log.debug("cross_over[self.config.trader_set.complete_index] : {}".format(cross_over[self.config.trader_set.complete_index]))
        self.sys_log.debug("cross_under[self.config.trader_set.complete_index] : {}".format(cross_under[self.config.trader_set.complete_index]))

        if self.position == 'LONG':
            if cross_over[self.config.trader_set.complete_index]:
                self.side_open = 'BUY'
        else:
            if cross_under[self.config.trader_set.complete_index]:
                self.side_open = 'SELL'
        self.sys_log.debug("LoopTableCondition : elasped time, set point : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
            

        
        if not pd.isnull(self.side_open): # default = np.nan
            
            # set priceBox automation (2)
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            try:
                self.df_res = get_DC(self.df_res, period=self.priceBox_value, interval=self.timeframe) #.tail()    
                # self.df_res = get_DC_perentage(self.df_res, period=self.priceBox_value, interval=self.timeframe) #.tail()     
            
                DC_upper = self.df_res['DC_{}{}_upper'.format(self.timeframe, self.priceBox_value)].to_numpy()
                DC_lower = self.df_res['DC_{}{}_lower'.format(self.timeframe, self.priceBox_value)].to_numpy()
                close = self.df_res.close.to_numpy()    
                
                if self.side_open == 'SELL':    
                    self.price_take_profit = DC_lower[self.config.trader_set.complete_index]
                    self.price_entry = close[self.config.trader_set.complete_index]
                    self.price_stop_loss = DC_upper[self.config.trader_set.complete_index]
                else:
                    self.price_take_profit = DC_upper[self.config.trader_set.complete_index]
                    self.price_entry = close[self.config.trader_set.complete_index]
                    self.price_stop_loss = DC_lower[self.config.trader_set.complete_index] 

                # get RRratio
                    # currently, we are using set_price from other functions.
                        # replace below function later.
                self.set_price_and_open_signal(self, env='BANK') # env select not to use adj_price_unit().       
                    
            except Exception as e:
                msg = "error in set priceBox automation : {}".format(e)
                self.sys_log.error(msg)
                # self.push_msg(self, msg)
                continue            
            self.sys_log.debug("LoopTableCondition : elasped time, set priceBox automation : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
    
    
            
            
            # set zone (4)
                # RRratio phase should be after set_price_and_open_signal.
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            RRratio_adj_fee_category = self.df_res['RRratio_adj_fee_category'].to_numpy()         
            
            self.sys_log.debug("RRratio_adj_fee_category[self.config.trader_set.complete_index] : {}".format(RRratio_adj_fee_category[self.config.trader_set.complete_index]))
            self.sys_log.debug("self.zone_value.split(';') : {}".format(self.zone_value.split(';')))
            
            if RRratio_adj_fee_category[self.config.trader_set.complete_index] in self.zone_value.split(';'):
                pass
            else:
                self.side_open = np.nan            
            # self.get_side_open(self, open_num=1)            
            self.sys_log.debug("self.side_open : {}".format(self.side_open))
            self.sys_log.debug("LoopTableCondition : elasped time, set zone : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
    

            
            if not pd.isnull(self.side_open): # default = np.nan
    
                # set symbol & code.
                self.code = "{}_{}".format(self.symbol, datetime.now().strftime('%Y%m%d%H%M%S%f'))
                init_table_trade(self) 
                self.push_msg(self, "{} row insert into TableTrade.".format(self.code))
    
                if drop:
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()   
                    self.table_condition.drop([idx], inplace=True)    
                    self.sys_log.debug("LoopTableCondition : elasped time, drop rows  : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")  

            
        # saving table takes some time, so we do not persist this phase more front.
            # check timestampLastUsed
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()    
        self.set_tables(self, mode='CONDITION')
        self.sys_log.debug("LoopTableCondition : elasped time, set_tables  : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")

            
        # save df_res.
            # to use not-updated df_res in LoopTableTrade's REPLACEMENT phase.
            # check df_res integrity.
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()            
        self.path_df_res = "{}\\{}.ftr".format(self.path_dir_df_res, self.symbol)
        self.df_res.reset_index(drop=False).to_feather(self.path_df_res , compression='lz4')
        self.sys_log.debug("LoopTableCondition : elasped time, save df_res : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")




def init_table_trade(self, ):

    """
    v1.0
        rm OrderSide, PositionSide.
    v2.0
        move get_balance to the top.
        modify to vivid input & output.
    
    last confirmed at, 20240613 1335.
    """
        
    # param init.    
    
    # get balance
        # Stock 특성상, balance_min = self.price_entry 로 설정함.
        # open_data_dict.pop 은 watch_dict 의 일회성 / 영구성과 무관하다.    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()    
    self.get_balance_info(self, balance_min=10)   
    self.sys_log.debug("InitTableTrade : elasped time, get_balance_info : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    if not self.balance_insufficient:
    
        self.order_motion = 0  # for consuming time without trading.
        
        # get prices for order.
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        self.price_entry, \
        self.price_stop_loss, \
        self.price_take_profit = self.get_price(self, 
                                              self.side_open, 
                                              self.df_res)
        self.sys_log.debug("InitTableTrade : elasped time, get_price : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        
        
        # get
            # self.side_close
            # self.side_position
            # self.price_entry open execution, 
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()    
        self.price_open = self.df_res['open'].to_numpy()[-1]  # open 은 latest_index 의 open 사용
        if self.side_open == 'BUY':
            self.side_close = 'SELL'
            self.side_position = 'LONG'
            if self.config.pos_set.long_fake:
                self.order_motion = 1
            self.price_entry = min(self.price_open, self.price_entry)
        else:
            self.side_close = 'BUY'
            self.side_position = 'SHORT'
            if self.config.pos_set.short_fake:
                self.order_motion = 1
            self.price_entry = max(self.price_open, self.price_entry)
        self.sys_log.debug("InitTableTrade : elasped time, side_close, side_position, price_entry open execution : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
            
    
        # get leverage
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time() 
        self.loss, \
        self.leverage_limit_user, \
        self.leverage_limit_server, \
        self.leverage = self.get_leverage_limit(self, 
                                              self.symbol, 
                                              self.price_entry, 
                                              self.price_stop_loss, 
                                              self.config.trader_set.fee_market,
                                              self.config.trader_set.fee_market)
        # self.get_leverage(self)
        # if self.leverage is None:
        #     self.sys_log.warning("leverage is None : {}".format(self.leverage))
        #     return
        self.sys_log.debug("InitTableTrade : elasped time, get_leverage : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
    
         
        # get price_liquidation
            # get_price_liquidation requires leverage.
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()   
        self.price_liquidation = self.get_price_liquidation(self.side_open, 
                                                       self.price_entry, 
                                                       self.config.trader_set.fee_limit, 
                                                       self.config.trader_set.fee_market,
                                                       self.leverage)
        self.sys_log.debug("InitTableTrade : elasped time, get_price_liquidation : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
    
        
        # get price_expiration
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()  
        self.get_price_expiration(self)
        self.sys_log.debug("InitTableTrade : elasped time, get_price_expiration : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
    
        
        # adj. precision_price & quantity
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()        
        self.precision_price, \
        self.precision_quantity = self.get_precision(self, 
                                                    self.symbol)
        self.price_take_profit, \
        self.price_entry, \
        self.price_stop_loss, \
        self.price_liquidation = [self.calc_with_precision(price_, self.precision_price) for price_ in [self.price_take_profit, 
                                                                                                        self.price_entry, 
                                                                                                        self.price_stop_loss,
                                                                                                        self.price_liquidation]]  # includes half-dynamic self.price_take_profit
        self.sys_log.debug("InitTableTrade : elasped time, adj. precision price & quantity : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
    
        
        # set leverage
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()  
        if not self.config.trader_set.backtrade:
            # [ RealTrade ]
            self.set_leverage(self)      
        self.sys_log.debug("InitTableTrade : elasped time, set_leverage : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
    
        
        # get quantity_open
        self.quantity_open = self.calc_with_precision(self.balance_available / self.price_entry * self.leverage, self.precision_quantity)
        
        self.sys_log.info("balance_available : {}".format(self.balance_available))
        self.sys_log.info("price_take_profit : {}".format(self.price_take_profit))
        self.sys_log.info("price_entry : {}".format(self.price_entry))
        self.sys_log.info("price_stop_loss : {}".format(self.price_stop_loss))
        self.sys_log.info("price_liquidation : {}".format(self.price_liquidation))
        self.sys_log.info("price_expiration : {}".format(self.price_expiration))
        self.sys_log.info("leverage : {}".format(self.leverage))
        self.sys_log.info("quantity_open : {}".format(self.quantity_open))
        
        # self.sys_log.debug("LoopTableCondition : elapsed time, init_table_trade %.5f" % (time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
    
        
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()  
        table_trade_new_row = pd.DataFrame(data = np.array([np.nan] * len(self.table_trade.columns)).reshape(-1, 1).T, columns=self.table_trade.columns, dtype=object)
    
        # # save df_res.
        # self.path_df_res = "{}\\{}.ftr".format(self.path_dir_df_res, self.code)
        # self.df_res.reset_index(drop=True).to_feather(self.path_df_res , compression='lz4')
    
        # init table_trade.
        table_trade_new_row.code = self.code 
        table_trade_new_row.path_df_res = self.path_df_res 
        table_trade_new_row.order_motion = self.order_motion 
        table_trade_new_row.side_open = self.side_open 
        table_trade_new_row.side_close = self.side_close 
        table_trade_new_row.side_position = self.side_position 
        table_trade_new_row.price_take_profit = self.price_take_profit 
        table_trade_new_row.price_entry = self.price_entry 
        table_trade_new_row.price_stop_loss = self.price_stop_loss 
        table_trade_new_row.price_liquidation = self.price_liquidation 
        table_trade_new_row.price_expiration = self.price_expiration 
        table_trade_new_row.leverage = self.leverage 
        table_trade_new_row.balance_available = self.balance_available 
        table_trade_new_row.quantity_open = self.quantity_open 
        table_trade_new_row.precision_price = self.precision_price 
        table_trade_new_row.precision_quantity = self.precision_quantity 
        
        table_trade_new_row.symbol = self.symbol 
    
        # self.sys_log.info("table_trade_new_row : \n{}".format(table_trade_new_row.iloc[0]))
    
            # append row.
        self.table_trade = pd.concat([self.table_trade, table_trade_new_row]).reset_index(drop=True)        
        self.sys_log.debug("InitTableTrade : elasped time, append row : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")



def loop_table_trade(self, ):

    """
    v1.0
        values come from rows should be static.
        prevent order_market dup.
        rm self.table_trade.at[idx, 'status'] = self.order_res['status'] in order phase.
            considering status_prev. cannot be here, causing no diffenece before / after order_info update.
            self.table_trade.at[idx, 'statusChangeTime'] can be replaced with order_res's updateTime
        use loc istead of iloc for concat table_log (stay comprehension with .at)
        secure orderId dtype
    v1.1
        add, if self.table_trade.at[idx, 'remove_row'] != 1: # if not trade is done,
            if self.table_trade.at[idx, 'order_way'] == 'CLOSE':
                if self.order_market_on:    
        replace to feather ver (orderId type to int64)   
        add side_open to check_stop_loss upper phase.
        reorder statusChangeTime & OPEN CLOSE
        modify statusChangeTime format
        add remove_row = 1, self.error_code is not None.
        add side_position on get_income_info().
        modify remove from websocket_client
            remove symbol from price_market
        modify get_price_realtime phase. (consider price_realtime error).
        
        add preventing losing orderId by set set_table in this function.
    v1.1.1
        modify to vivid input / output
    
    last confirmed at, 20240614 1325.
    """ 
    
    for idx, row in self.table_trade.iterrows(): # for save iteration, use copy().
        
        # init.
        self.order_info = None
        self.expired = 0
        self.order_market_on = 0
        self.table_trade.at[idx, 'order'] = 0 # default.        
        
        # update & clean table.
            # 'row' means un-updated information. (by get_order_info)
        self.code = row.code
        self.symbol = row.symbol # we have symbol column in table already.
        self.orderId = row.orderId
        self.status_prev = row.status # save before updateing status.        
        self.order_motion = row.order_motion

        self.sys_log.debug("row : {}".format(row))
        self.sys_log.debug("row.dtypes : {}".format(row.dtypes))
    

        # set order, order_way (OPEN) (order not exists)
        if pd.isnull(self.orderId):
            
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            self.table_trade.at[idx, 'order'] = 1
            self.table_trade.at[idx, 'order_way'] = 'OPEN'

            if self.symbol not in self.price_market.keys(): # only once.
                self.websocket_client.agg_trade(symbol=self.symbol, id=1, callback=self.agg_trade_message_handler)
                self.set_position_mode(self)
                self.set_margin_type(self)
            
            self.sys_log.debug("LoopTableTrade : elasped time, set order, order_way (OPEN) : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")

        #  order exists.
        else:
        # if not pd.isnull(self.orderId): # this means, status is not None. (order in settled)

            # get_order_info. (update order_info)
                # get_order_info should have valid symbol & orderId.   
            self.get_order_info(self,)   
            
            if self.order_info is not None: # order should be exist.
                
                # update table_trade
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                for k, v in self.order_info.items():
                    self.table_trade.at[idx, k] = v
                self.sys_log.debug("LoopTableTrade : elasped time, update table_trade : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")
                # display(self.table_trade)    
                
                # set statusChangeTime 
                    # check if status has been changed.
                if self.status_prev != self.order_info['status']:
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    # self.table_trade.at[idx, 'statusChangeTime'] = int(time.time())
                    self.table_trade.at[idx, 'statusChangeTime'] = datetime.now().strftime('%Y%m%d%H%M%S%f')
                    self.push_msg(self, "{} status has been changed. {} {}".format(self.code, row.order_way, self.order_info['status']))
                    self.sys_log.debug("LoopTableTrade : elasped time, set statusChangeTime  : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")      
                    # display(self.table_trade)
                    
                    # set remove_row. 
                        # use updated status. 
                    if (self.order_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']) or (row.order_way == 'CLOSE' and self.order_info['status'] == 'FILLED'):
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        self.table_trade.at[idx, 'remove_row'] = 1
                        self.sys_log.debug("LoopTableTrade : elasped time, set remove_row  : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                    
                    # logging : transfer rows to Table - log.        
                    self.table_log = pd.concat([self.table_log, self.table_trade.loc[[idx]]]) # list input persist dataframe oubalance_availableut.
                    self.sys_log.debug("LoopTableTrade : elasped time, + logging  : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")  
                    # display(self.table_trade)

                
                # set order, order_way (CLOSE)
                if row.order_way == 'OPEN':         
                    if (abs(float(self.order_info['executedQty']) / float(self.order_info['origQty'])) >= self.config.trader_set.quantity_open_exec_ratio) or self.order_info['status'] == 'FILLED':           
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        self.table_trade.at[idx, 'order'] = 1
                        self.table_trade.at[idx, 'order_way'] = 'CLOSE'
                        self.sys_log.debug("LoopTableTrade : elasped time, set order, order_way (CLOSE) : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                
                
                # check expiration for order_open.
                    # set price_reatlime
                        # for, open expiry & check_stop_loss
                self.get_price_realtime(self,)

                if not pd.isnull(self.price_realtime):                    
                    self.sys_log.debug("self.price_realtime : {}".format(self.price_realtime))
                    self.table_trade.at[idx, 'price_realtime'] = self.price_realtime      
                    
            
                        # row.side_position has 2 state (LONG / SHORT) in status below 2 case.
                    if row.order_way == 'OPEN' and self.order_info['status'] in ['NEW', 'PARTIALLY_FILLED']:
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.price_expiration = row.price_expiration
                        if row.side_position == 'SHORT':
                            if self.price_realtime <= self.price_expiration:
                                self.expired = 1
                        else: # elif row.side_position == 'LONG':
                            if self.price_realtime >= self.price_expiration:
                                self.expired = 1                
                            
                        if self.expired:      
                            self.order_cancel(self,)
                            
                            # Todo, messeage alertion needed ?
                            self.push_msg(self, "{} has been expired.".format(self.code))
                            self.user_text = 'watch'  # allowing message.
                        self.sys_log.debug("LoopTableTrade : elasped time, check expiration for order_open  : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
        
                    
                    # check stop_loss
                        # get order_market_on
                        # order_way == 'CLOSE' : stop_loss / liquidation condition.
                    elif row.order_way == 'CLOSE':
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()   
                        self.side_open = row.side_open
                        self.price_stop_loss = row.price_stop_loss
                        self.price_liquidation = row.price_liquidation
                        self.check_stop_loss(self,)
                        # self.check_stop_loss_by_signal(self,)
                        self.sys_log.debug("LoopTableTrade : elasped time, check_stop_loss  : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
            
        
        # order_limit
        if not self.config.trader_set.backtrade and not self.order_motion:

           # reflect updated row.
           if self.table_trade.at[idx, 'order'] == 1:

                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                # public set.
                order_type = 'LIMIT' # fixed for order_limit.
                side_position = row.side_position
            
                # open condition
                if self.table_trade.at[idx, 'order_way'] == 'OPEN':
                
                    # open                    
                    side_order = row.side_open
                    price = row.price_entry
                    quantity = row.quantity_open
    
                # close condition
                else: #  self.table_trade.at[idx, 'order_way'] == 'CLOSE': # upper phase's condition use 'order' = 1
                            
                    # we don't do partial like this anymore.
                        # each price info will be adjusted order by order.
                    # self.order_limit_multiple(self, order_res_list)
                    
                    # close                    
                    side_order = row.side_close
                    price = row.price_take_profit
                    quantity = self.order_info['executedQty']
                    
                # [ API - Trading ]                
                self.order_limit(self, order_type, side_order, side_position, price, quantity)
            
                # KIWOOM version.
                    # 주문 단가가 상한선인 경우, 주문 가능할 때까지 keep.
                        # self.price_stop_loss check 는 지속적으로 수행함.
                        # 한번만 append 해야할 것, close_data 에서는 pop 하지 않음.
                            # 이것도 누적인가 ?
               
                # normal state : self.error_code = 0
                if not self.error_code:
                    # self.table_trade.at[idx, 'orderId'] = str(self.order_res['orderId']) # this is key for updating order_info (change to OPEN / CLOSE orderId)
                    self.table_trade.at[idx, 'orderId'] = self.order_res['orderId'] # this is key for updating order_info (change to OPEN / CLOSE orderId)
                else:
                    self.table_trade.at[idx, 'remove_row'] = 1
                self.sys_log.debug("LoopTableTrade : elasped time, order_limit  : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")

        
        # order_market
            # prevent order_market duplication.
                # order_way == 'CLOSE' : stop_loss / liquidation condition.
                    # repflect updated row.
        if self.table_trade.at[idx, 'remove_row'] != 1: # if not trade is done,
            if self.table_trade.at[idx, 'order_way'] == 'CLOSE':
                if self.order_market_on:
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()                
                    self.side_close = row.side_close
                    self.side_position = row.side_position
                    self.order_market(self,)
                    self.table_trade.at[idx, 'orderId'] = self.order_res['orderId'] # this is key for updating order_info (change to CLOSE orderId)
                    self.sys_log.debug("LoopTableTrade : elasped time, order_market  : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")

        
        # drop rows
            # this phase should be placed in the most below, else order_ will set value as np.nan in invalid rows.
        else:
        # if self.table_trade.at[idx, 'remove_row'] == 1:      
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()      
            # drow remove_row = 1,
            self.table_trade.drop([idx], inplace=True)
            # display(self.table_trade)

            # remove from websocket_client.
            if self.symbol not in self.table_trade.symbol.values:        
                self.websocket_client.stop_socket("{}@aggTrade".format(self.symbol.lower()))
                
                if self.symbol in self.price_market.keys():
                    self.price_market.pop(self.symbol)

            self.sys_log.debug("LoopTableTrade : elasped time, drop rows  : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")  

            # except open canceld,            
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            self.income, \
            self.income_accumulated, \
            self.profit, \
            self.profit_accumulated = self.get_income_info(self, 
                                                            self.table_log,
                                                            self.code,
                                                            row.side_position,
                                                            row.leverage,
                                                            self.income_accumulated,
                                                            self.profit_accumulated,                    
                                                            mode="PROD", 
                                                            currency="USDT")      
            self.sys_log.debug("LoopTableTrade : elasped time, get_income_info  : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")  
            # display(self.table_trade)


        # preventing losing orderId      
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()    
        self.set_tables(self, mode='TRADE')
        self.sys_log.debug("LoopTableTrade : elasped time, set_tables  : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")  

        
        # preventing API flood.
        time.sleep(0.1)
        
        
