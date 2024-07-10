from bank import *
from idep import *



def loop_messenger(self,):

    """
    v1.0
        init
            consider dtypes in payload (float & str)
                default type = str.
    v1.0.1
        apply dbms
        
    last confirmed at, 20240705 2330.
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
                    
                    self.replace_table(self.table_condition, self.table_condition_name)
                    
                    self.sys_log.debug("InitTableCondition : elasped time, replace_table  : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")  
                    
                    user_text_upper_case = 'WATCH'
                                    
        # watch
        if 'WATCH' in user_text_upper_case:
            msg = "{}".format(self.table_condition)


        # show message.
        self.push_msg(msg)


        # reset user_text.
        self.user_text = None
        
def loop_table_condition(self, drop=False, debug=False):

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
        add interval_value (days)
        
        v1.5.1
            apply functional idep.py.
            add target_loss, account            
        v1.5.2
            apply dbms
            
            v1.5.2.1
                divide send / save. 
                add debug mode.
                api_count + 1    
                
        v1.5.3
            apply TokenBucket.
            remove loop_duration.
        
    last confirmed at, 20240710 1016.
    """

    start_time_loop = time.time()
    for idx, row in self.table_condition.iterrows(): # idx can be used, cause table_condition's index are not reset.

        # init.        
        self.table_condition_row = row # declared for using in entry_point_location. (deprecated)
           
        self.symbol = row.symbol
        self.side_open = row.side_open
        # self.price_take_profit = row.price_take_profit
        # self.price_entry = row.price_entry
        # self.price_stop_loss = row.price_stop_loss
        # self.leverage = row.leverage
        self.position = row.position
        self.interval = row.interval
        self.interval_value = row.interval_value
        self.priceBox_indicator = row.priceBox_indicator
        self.priceBox_value = row.priceBox_value
        self.point_indicator = row.point_indicator
        self.point_value = row.point_value
        self.zone_indicator = row.zone_indicator
        self.zone_value = row.zone_value
        self.target_loss = row.target_loss
        self.account = row.account
        

        # do not repeat it in interval.
        interval_number = itv_to_number(self.interval)
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
                
            self.sys_log.debug("LoopTableCondition : elasped time, check timestampLastUsed : {:.2f}s".format(time.time() - start_time))
            self.sys_log.debug("------------------------------------------------")
        

        # get df_res
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()

        self.token_bucket.wait_for_token_consume(1)        
        self.df_res = get_df_new(self, 
                               interval=self.interval, 
                               days=self.interval_value, 
                               limit=1500, 
                               timesleep=0.0) # changed to 0.0
                
        self.sys_log.info("LoopTableCondition : check df_res last row timestamp : {}".format(self.df_res.index[-1]))
        self.sys_log.debug("LoopTableCondition : elasped time, get_df_new : {:.2f}s".format(time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
 
        if self.df_res is None:
            continue     
            
        
        # set point (3)   
            # set side_open considered. (set priceBox_indicator automation require side_open)
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        self.df_res, \
        cross_over, \
        cross_under = get_point_index(self.df_res,
                                        row.point_mode,
                                        row.point_indicator,
                                        row.point_value,
                                        row.interval,
                                        return_bool=True)
        
        point_index_long = np.argwhere(cross_over).ravel() # input for get_price_arr
        point_index_short = np.argwhere(cross_under).ravel()   

        point_min_len = abs(self.config.trader_set.complete_index)
        if len(cross_over) >= point_min_len and len(cross_under) >= point_min_len:
            self.sys_log.debug("cross_over[self.config.trader_set.complete_index] : {}".format(cross_over[self.config.trader_set.complete_index]))
            self.sys_log.debug("cross_under[self.config.trader_set.complete_index] : {}".format(cross_under[self.config.trader_set.complete_index]))
    
            if self.position == 'LONG':
                if cross_over[self.config.trader_set.complete_index] and not debug: # debugging.
                    self.side_open = 'BUY'
            else:
                if cross_under[self.config.trader_set.complete_index] and not debug: # debugging.
                    self.side_open = 'SELL' 
                    
        self.sys_log.debug("LoopTableCondition : elasped time, set point : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
            

        
        if not pd.isnull(self.side_open): # default = np.nan
            
            # set priceBox_indicator automation (2)
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()       
            
            try:                
                priceBox_upper, \
                priceBox_lower, \
                close = get_priceBox(self.df_res,
                                     row.priceBox_indicator,
                                     row.priceBox_value,
                                     row.interval,
                                     self.side_open)
            
                
                # price_arr use priceBox_indicator differed by side_open.
                price_take_profit_arr, \
                price_entry_arr, \
                price_stop_loss_arr, \
                index_valid_bool = get_price_arr(self.side_open, 
                                             priceBox_upper, 
                                             priceBox_lower,
                                             close,
                                             point_index_short,
                                             point_index_long)

                self.sys_log.debug("index_valid_bool : {}".format(index_valid_bool))
                
                if len(index_valid_bool):    
                    # we have save df_res, so not using conitnue.
                        # use lastest index, -1.
                            # cannot be debugged for price protection.
                    if not index_valid_bool[-1]:
                        self.side_open = np.nan                   
                     
                    self.price_take_profit = price_take_profit_arr[-1]
                    self.price_entry = price_entry_arr[-1]
                    self.price_stop_loss = price_stop_loss_arr[-1]
    
                    # get RRratio
                        # currently, we are using set_price from other functions.
                            # replace below function later.
                    set_price_and_open_signal(self, env='BANK') # env select not to use adj_price_unit().       
                    
                    RRratio_adj_fee_category = self.df_res['RRratio_adj_fee_category'].to_numpy()            
                    self.sys_log.debug("RRratio_adj_fee_category[self.config.trader_set.complete_index] : {}".format(RRratio_adj_fee_category[self.config.trader_set.complete_index]))
                    self.sys_log.debug("self.zone_value.split(';') : {}".format(self.zone_value.split(';')))
                    
                    # debugging.
                    if RRratio_adj_fee_category[self.config.trader_set.complete_index] not in self.zone_value.split(';') and not debug:
                        self.side_open = np.nan                        
                else:
                    self.side_open = np.nan  
                    
                self.sys_log.debug("self.side_open : {}".format(self.side_open))
                    
            except Exception as e:
                msg = "error in get_priceBox, get_price_arr : {}".format(e)
                self.sys_log.error(msg)
                # self.push_msg(msg)
                continue 
                
            self.sys_log.debug("LoopTableCondition : elasped time, get_priceBox, get_price_arr : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
    
    
            
            # # set zone (4)
            #     # RRratio phase should be after set_price_and_open_signal.
            # self.sys_log.debug("------------------------------------------------")
            # start_time = time.time()
            
            # self.sys_log.debug("LoopTableCondition : elasped time, set zone : %.4fs" % (time.time() - start_time)) 
            # self.sys_log.debug("------------------------------------------------")
    

            
            if not pd.isnull(self.side_open): # default = np.nan
    
                # set symbol & code.
                self.code = "{}_{}".format(self.symbol, datetime.now().strftime('%Y%m%d%H%M%S%f'))
                init_table_trade(self) 
    
                if drop:
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()   
                    
                    self.table_condition.drop([idx], inplace=True)    
                    
                    self.sys_log.debug("LoopTableCondition : elasped time, drop rows  : %.4fs" % (time.time() - start_time)) 
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

        
        # saving table takes some time, so we do not persist this phase more front.
            # check timestampLastUsed
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()   
        
        self.replace_table(self.table_condition, self.table_condition_name)
        
        self.sys_log.debug("LoopTableCondition : elasped time, replace_table  : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        


    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
    
    self.replace_table(self.table_condition, self.table_condition_name, send=True)
    
    self.sys_log.debug("LoopTableCondition : elasped time, replace_table (send=True)  : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
    
    
def init_table_trade(self, ):

    """
    v1.0
        rm OrderSide, PositionSide.
    v2.0
        move get_balance to the top.
        modify to vivid input & output.
        
        v2.1
            vivid mode.
            add account & margin.
        v2.2
            apply dbms.
             
            v2.2.1
                divide send / save. 
                api_count + 3
        v2.3
            apply TokenBucket.
    
    last confirmed at, 20240710 1019.
    """
           
    
    self.order_motion = 0  # for consuming time without trading.

    # we don't use get_price anymore...
        # cause priceBox values are gotten already from LoopTableCondition. 
    # # get prices for order.
    # self.sys_log.debug("------------------------------------------------")
    # start_time = time.time()
    # self.price_entry, \
    # self.price_stop_loss, \
    # self.price_take_profit = self.get_price(self, 
    #                                       self.side_open, 
    #                                       self.df_res)
    # self.sys_log.debug("InitTableTrade : elasped time, get_price : %.4fs" % (time.time() - start_time)) 
    # self.sys_log.debug("------------------------------------------------")
    
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
        
    self.side_close, \
    self.side_position = get_side_info(self, 
                                       self.side_open)
        
    self.sys_log.debug("InitTableTrade : elasped time, get_side_info : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
    
    self.price_entry = get_price_entry(self.df_res,
                                       self.side_open,
                                       self.price_entry)
    
    self.sys_log.debug("InitTableTrade : elasped time, get_price_entry : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
        

    # get price_liquidation
        # get_price_liquidation requires leverage.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   

    # we are using target_loss_pct now, so liquidation has no more meaning.
        # target_loss_pct max is 100.
    self.price_liquidation = self.price_stop_loss
    
    self.sys_log.debug("InitTableTrade : elasped time, get_price_liquidation : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    # get price_expiration
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()  
    
    self.price_expiration = self.price_take_profit # temporarily fix to price_take_profit.
    
    self.sys_log.debug("InitTableTrade : elasped time, get_price_expiration : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    # adj. precision_price & quantity
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()    
    
    self.token_bucket.wait_for_token_consume(1)   
    
    self.precision_price, \
    self.precision_quantity = get_precision(self, 
                                            self.symbol)
    self.price_take_profit, \
    self.price_entry, \
    self.price_stop_loss, \
    self.price_liquidation, \
    self.price_expiration = [self.calc_with_precision(price_, self.precision_price) for price_ in [self.price_take_profit, 
                                                                                                    self.price_entry, 
                                                                                                    self.price_stop_loss,
                                                                                                    self.price_liquidation,
                                                                                                    self.price_expiration]]  # add price_expiration.
    
    self.sys_log.debug("InitTableTrade : elasped time, adj. precision_price : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

            
    # get leverage
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    self.token_bucket.wait_for_token_consume(1)   
    
    self.loss, \
    self.leverage_limit_user, \
    self.leverage_limit_server, \
    self.leverage = get_leverage_limit(self, 
                                       self.symbol, 
                                       self.price_entry, 
                                       self.price_stop_loss, 
                                       self.config.trader_set.fee_market,
                                       self.config.trader_set.fee_market)
    # self.get_leverage(self)
    
    self.sys_log.debug("InitTableTrade : elasped time, get_leverage : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

            
    # set leverage
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()  
    
    if not self.config.trader_set.backtrade:
        
        self.token_bucket.wait_for_token_consume(1)   
        
        self.set_leverage(self.symbol,
                          self.leverage)      
        
    self.sys_log.debug("InitTableTrade : elasped time, set_leverage : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    # get quantity_open   
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()  
    
    if self.target_loss:
        self.quantity_open = self.target_loss / self.loss
    else:
        self.quantity_open = 15 / self.price_entry
    self.quantity_open = self.calc_with_precision(self.quantity_open, self.precision_quantity)
    
    # Calculate entry and exit amounts
        # amount_entry cannot be larger than the target_loss.
    self.amount_entry = self.price_entry * self.quantity_open
    self.margin = self.amount_entry / self.leverage
    
    self.sys_log.debug("InitTableTrade : elasped time, get margin : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

            
    self.sys_log.info("price_take_profit : {}".format(self.price_take_profit))
    self.sys_log.info("price_entry : {}".format(self.price_entry))
    self.sys_log.info("price_stop_loss : {}".format(self.price_stop_loss))
    self.sys_log.info("price_liquidation : {}".format(self.price_liquidation))
    self.sys_log.info("price_expiration : {}".format(self.price_expiration))
    self.sys_log.info("quantity_open : {}".format(self.quantity_open))
    self.sys_log.info("amount_entry : {}".format(self.amount_entry))
    self.sys_log.info("leverage : {}".format(self.leverage))
    self.sys_log.info("margin : {}".format(self.margin))

    
    # get balance
        # Stock 특성상, balance_min = self.price_entry 로 설정함.
        # open_data_dict.pop 은 watch_dict 의 일회성 / 영구성과 무관하다.    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()

    self.token_bucket.wait_for_token_consume(1)   
    
    margin_consistency = get_margin_consistency(self, 
                                                self.margin,
                                                self.account,
                                                mode="PROD")
    
    self.sys_log.debug("InitTableTrade : elasped time, get_margin_consistency : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

    
    if margin_consistency:    
    
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()  
        
        table_trade_new_row = pd.DataFrame(data = np.array([np.nan] * len(self.table_trade.columns)).reshape(-1, 1).T, columns=self.table_trade.columns, dtype=object)
    
        # # save df_res.
        self.path_df_res_open = "{}/{}.ftr".format(self.path_dir_df_res, self.code)
        # self.df_res.reset_index(drop=True).to_feather(self.path_df_res , compression='lz4')
    
        # init table_trade.
        table_trade_new_row.symbol = self.symbol 
        table_trade_new_row.code = self.code 
        table_trade_new_row.path_df_res_open = self.path_df_res_open 
        table_trade_new_row.order_motion = self.order_motion 
        table_trade_new_row.side_open = self.side_open 
        table_trade_new_row.side_close = self.side_close 
        table_trade_new_row.side_position = self.side_position 
        table_trade_new_row.precision_price = self.precision_price 
        table_trade_new_row.precision_quantity = self.precision_quantity 
        table_trade_new_row.price_take_profit = self.price_take_profit 
        table_trade_new_row.price_entry = self.price_entry 
        table_trade_new_row.price_stop_loss = self.price_stop_loss 
        table_trade_new_row.price_liquidation = self.price_liquidation 
        table_trade_new_row.price_expiration = self.price_expiration 
        table_trade_new_row.quantity_open = self.quantity_open 
        table_trade_new_row.amount_entry = self.amount_entry # added.
        table_trade_new_row.leverage = self.leverage 
        table_trade_new_row.margin = self.margin             # added.
        table_trade_new_row.account = self.account           # added. 
    
        # self.sys_log.info("table_trade_new_row : \n{}".format(table_trade_new_row.iloc[0]))
    
        # append row.
        self.table_trade = pd.concat([self.table_trade, table_trade_new_row]).reset_index(drop=True)            
        
        self.sys_log.debug("InitTableTrade : elasped time, append row : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        
        self.push_msg("{} row insert into TableTrade.".format(self.code))        
        
        
    # preventing losing table_account info.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    self.replace_table(self.table_account, self.table_account_name, send=True)
    self.replace_table(self.table_trade, self.table_trade_name, send=True)
    
    self.sys_log.debug("InitTableTrade : elasped time, replace_table (send=True)  : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")  
    
    
def loop_table_trade(self, ):

    """
    v1.0
        values come from rows should be static.
        prevent order_market dup.
        rm self.table_trade.at[idx, 'status'] = self.order_result['status'] in order phase.
            considering status_prev. cannot be here, causing no diffenece before / after order_info update.
            self.table_trade.at[idx, 'statusChangeTime'] can be replaced with order_result's updateTime
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
            add \n to row : {}.
        v1.1.2
            vivid mode after v1.1.1
                decide to use without self. (if self. is needless)
        v1.1.3
            apply dbms.

            v1.1.3.1
                divide send / server.
                api_count + 8. (dynamic)
        v1.1.4
            apply TokenBucket.
    
    last confirmed at, 20240710 1020.
    """ 
    
    for idx, row in self.table_trade.iterrows(): # for save iteration, use copy().
        
        # init.
        self.order_info = None
        self.expired = 0
        self.order_market_on = 0
        self.table_trade.at[idx, 'order'] = 0 # default.        
        
        # update & clean table.
            # some data need to be transfered to Bank's instance, some don't.
                # 'row' means un-updated information. (by get_order_info)
        self.code = row.code
        self.symbol = row.symbol # we have symbol column in table already.
        self.orderId = row.orderId
        self.status_prev = row.status # save before updateing status.        
        self.order_motion = row.order_motion
        
        self.price_expiration = row.price_expiration 
        
        self.side_open = row.side_open
        self.price_stop_loss = row.price_stop_loss
        self.price_liquidation = row.price_liquidation
                        
        self.side_close = row.side_close
        self.side_position = row.side_position        

        self.sys_log.debug("row : \n{}".format(row))
        self.sys_log.debug("row.dtypes : {}".format(row.dtypes))
    

        # set order, order_way (OPEN) (order not exists)
        if pd.isnull(self.orderId):
            
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            self.table_trade.at[idx, 'order'] = 1
            self.table_trade.at[idx, 'order_way'] = 'OPEN'

            if self.symbol not in self.price_market.keys(): # only once.
                
                self.token_bucket.wait_for_token_consume(3)
                self.websocket_client.agg_trade(symbol=self.symbol, 
                                                id=1, 
                                                callback=self.agg_trade_message_handler)
                self.set_position_mode(dualSidePosition='true')
                self.set_margin_type(symbol=self.symbol)
            
            self.sys_log.debug("LoopTableTrade : elasped time, set order, order_way (OPEN) : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")

        #  order exists.
        else:
        # if not pd.isnull(self.orderId): # this means, status is not None. (order in settled)

            # get_order_info. (update order_info)
                # get_order_info should have valid symbol & orderId.        
            self.token_bucket.wait_for_token_consume(1)            
            self.order_info = get_order_info(self, 
                                       self.symbol,
                                       self.orderId)
            
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
                    self.push_msg("{} status has been changed. {} {}".format(self.code, row.order_way, self.order_info['status']))
                    
                    self.sys_log.debug("LoopTableTrade : elasped time, set statusChangeTime : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")      
                    # display(self.table_trade)
                    
                    # set remove_row. 
                        # use updated status. 
                    if (self.order_info['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']) or (row.order_way == 'CLOSE' and self.order_info['status'] == 'FILLED'):
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_trade.at[idx, 'remove_row'] = 1
                        
                        self.sys_log.debug("LoopTableTrade : elasped time, set remove_row : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                    
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    # logging : transfer rows to Table - log.        
                    self.table_log = pd.concat([self.table_log, self.table_trade.loc[[idx]]]) # list input persist dataframe oubalance_availableut.
                    self.table_log.reset_index(drop=True, inplace=True) # for indexing row by 'loc'    
                    self.replace_table(self.table_log, self.table_log_name)
                    
                    self.sys_log.debug("LoopTableTrade : elasped time, replace_table : %.4fs" % (time.time() - start_time)) 
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
                self.token_bucket.wait_for_token_consume(1)   
                self.price_realtime = get_price_realtime(self, 
                                                        self.symbol)
                self.sys_log.debug("self.price_realtime : {}".format(self.price_realtime))

                if not pd.isnull(self.price_realtime):                    
                    self.table_trade.at[idx, 'price_realtime'] = self.price_realtime                    
            
                        # row.side_position has 2 state (LONG / SHORT) in status below 2 case.
                    if row.order_way == 'OPEN' and self.order_info['status'] in ['NEW', 'PARTIALLY_FILLED']:
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()  
                                
                        self.expired = check_expiration(self.side_position, 
                                                        self.price_realtime, 
                                                        self.price_expiration)
                            
                        if self.expired:
                            
                            self.token_bucket.wait_for_token_consume(1)   
                            quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                        self.symbol,
                                                                        self.orderId)  
                            
                            self.table_account.loc[self.table_account['account'] == row.account, 'balance'] +=(quantity_unexecuted * row.price_entry / row.leverage)
                            
                            # Todo, messeage alertion needed ?
                            self.push_msg("{} has been expired.".format(self.code))
                            self.user_text = 'watch'  # allowing message.
                            
                        self.sys_log.debug("LoopTableTrade : elasped time, check expiration for order_open : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
        
                    
                    # check stop_loss
                        # get order_market_on
                        # order_way == 'CLOSE' : stop_loss / liquidation condition.
                    elif row.order_way == 'CLOSE':
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()  
                        
                        self.order_market_on = check_stop_loss(self,
                                                            self.side_open,
                                                            self.price_realtime,
                                                            self.price_liquidation,
                                                            self.price_stop_loss)
                        
                        self.sys_log.debug("LoopTableTrade : elasped time, check_stop_loss : %.4fs" % (time.time() - start_time)) 
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
                    
                    # close                    
                    side_order = row.side_close
                    price = row.price_take_profit
                    quantity = self.order_info['executedQty'] # OPEN's qty.

            
                self.token_bucket.wait_for_token_consume(1)   
                self.order_result, \
                self.error_code = order_limit(self, 
                                            self.symbol,
                                            side_order, 
                                            side_position, 
                                            price, 
                                            quantity)
            
                # KIWOOM version.
                    # If the order price is at the upper limit, keep it until the order is possible.
                        # Continuously perform self.price_stop_loss check.
                        # Append only once; do not pop in close_data.
                            # Is this also cumulative?
               
                # normal state : self.error_code = 0
                if not self.error_code:
                    # self.table_trade.at[idx, 'orderId'] = str(self.order_result['orderId']) # this is key for updating order_info (change to OPEN / CLOSE orderId)
                    self.table_trade.at[idx, 'orderId'] = self.order_result['orderId'] # this is key for updating order_info (change to OPEN / CLOSE orderId)
                    
                else:
                    self.table_trade.at[idx, 'remove_row'] = 1
                    
                    if self.table_trade.at[idx, 'order_way'] == 'OPEN': # if open fail, deposit withdrew margin.
                        self.table_account.loc[self.table_account['account'] == row.account, 'balance'] += row.margin
                    
                self.sys_log.debug("LoopTableTrade : elasped time, order_limit : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")

        
        # order_market
            # prevent order_market duplication.
                # order_way == 'CLOSE' : stop_loss / liquidation condition.
                    # repflect updated row.
        if self.table_trade.at[idx, 'remove_row'] != 1: # if not trade is done, it has np.nan or 1.
            if self.table_trade.at[idx, 'order_way'] == 'CLOSE':
                if self.order_market_on:
                    
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                    
                    self.token_bucket.wait_for_token_consume(2)                    
                    quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                self.symbol,
                                                                self.orderId)                    

                    self.order_result, \
                    self.error_code = order_market(self, 
                                                 self.symbol,
                                                 self.side_close,
                                                 self.side_position,
                                                 quantity_unexecuted)
                    
                    # normal state : self.error_code = 0
                    if not self.error_code:
                        self.table_trade.at[idx, 'orderId'] = self.order_result['orderId'] # this is key for updating order_info (change to CLOSE orderId)                                
                    self.sys_log.debug("LoopTableTrade : elasped time, order_market : %.4fs" % (time.time() - start_time)) 
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

            self.sys_log.debug("LoopTableTrade : elasped time, drop rows : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")  

            # except open canceld,            
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            self.income, \
            self.income_accumulated, \
            self.profit, \
            self.profit_accumulated = get_income_info(self, 
                                                    self.table_log,
                                                    self.code,
                                                    row.side_position,
                                                    row.leverage,
                                                    self.income_accumulated,
                                                    self.profit_accumulated,                    
                                                    mode="PROD", 
                                                    currency="USDT")
            
            self.table_account.loc[self.table_account['account'] == row.account, 'balance'] += self.income
            
            self.sys_log.debug("LoopTableTrade : elasped time, get_income_info : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")  
            # display(self.table_trade)


        # preventing losing orderId      
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        self.replace_table(self.table_account, self.table_account_name)
        self.replace_table(self.table_trade, self.table_trade_name)
        
        self.sys_log.debug("LoopTableTrade : elasped time, replace_table : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")  

        
        # # preventing API flood.
        # time.sleep(0.1)

    # use send=True instead time.sleep for API flood.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    self.replace_table(self.table_account, self.table_account_name, send=True)
    self.replace_table(self.table_trade, self.table_trade_name, send=True)
    self.replace_table(self.table_log, self.table_log_name, send=True)
    
    self.sys_log.debug("LoopTableTrade : elasped time, replace_table (send=True) : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")  
