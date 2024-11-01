from bank import *
from idep import *



def loop_messenger(self,):

    """
    v2.0
        simple messenger for check program alive.
        
    Last confirmed, 20240925 0950.
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
        

def loop_table_condition(self, drop=False, debugging=False):
   
    """     
    v1.4:
        - Modify to TableCondition_v0.3 version.
        - Modify drop to False.
        - Remove elapsed time +.
    v1.5:
        - Add column 'timestampLastUsed' in tableCondition.
        - Apply remaining minutes to zero.
        - Add interval_value (days).
    v1.5.1:
        - Apply functional idep.py.
        - Add target_loss, account.
    v1.5.2:
        - Apply dbms.
    v1.5.2.1:
        - Divide send / save.
        - Add debug mode.
        - api_count + 1.
    v1.5.3:
        - Apply TokenBucket.
    v1.5.4:
        - Remove loop duration.
        - Move token info to internal function.
    v1.5.5:
        - Use db_manager.
    v1.5.6:
        - Replace target_loss to divider.
    v1.5.8:
        - modify
            table_condition
                remove interval value. 
        - update required
            value usage between self & row.
    v1.6
        - update
            use point_bool & zone_bool.
            add account info to code.
        - modify
            check last row index.
            add idx to code.
                modify ids to condition string.
            pusg_msg point & RRratio.
            add 180 to calc days.
            add msg for self.df_res.

    20241028 2317.
    """

    
    for idx, row in self.table_condition.iterrows(): # idx can be used, cause table_condition's index are not reset.

        # init.        
        self.table_condition_row = row # declared for using in entry_point_location. (deprecated)
           
        self.symbol = row.symbol
        self.side_open = row.side_open
        # self.price_take_profit = row.price_take_profit
        # self.price_entry = row.price_entry
        # self.price_stop_loss = row.price_stop_loss
        # self.leverage = row.leverage
        
        self.priceBox_indicator = row.priceBox_indicator
        self.priceBox_value = row.priceBox_value
        self.point_mode = row.point_mode
        self.point_indicator = row.point_indicator
        self.point_value = row.point_value
        self.zone_indicator = row.zone_indicator
        self.zone_value = row.zone_value
        
        self.position = row.position
        self.interval = row.interval
        self.RRratio = row.RRratio
        
        self.divider = row.divider
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
        
        self.sys_log.debug(f"idx: {idx}")        
        
        values = [180, self.priceBox_value, self.point_value, self.zone_value] # add 180 for recursive indiator.
        days = np.nanmax([value if value is not None else np.nan for value in values]) / (1440 / itv_to_number(self.interval)) + 1        
        self.sys_log.debug(f"days: {days}")
  
        self.df_res = get_df_new(self, 
                               interval=self.interval, 
                               days=days, 
                               limit=1500, 
                               timesleep=0.0) # changed to 0.0
                
        self.sys_log.debug("LoopTableCondition : elasped time, get_df_new : {:.2f}s".format(time.time() - start_time))
        self.sys_log.debug("------------------------------------------------")
 
        if self.df_res is None or type(self.df_res) is str or len(self.df_res) < abs(self.config.bank.complete_index):
            msg = f"Warning in df_res : {self.df_res}"
            self.sys_log.warning(msg)
            self.push_msg(msg)
            
            if type(self.df_res) == str:
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()   
                
                self.table_condition.drop([idx], inplace=True)    
                
                self.sys_log.debug(f"LoopTableCondition : elasped time, drop invalid idx {idx}: %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")
            continue
        else:
            self.sys_log.debug("LoopTableCondition : check df_res last row timestamp : {}".format(self.df_res.index[-1]))
                           
        
        # set point (3)   
            # set side_open considered. (set priceBox_indicator automation require side_open)
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        self.df_res, \
        point_bool_long, \
        point_bool_short = get_point_bool(self.df_res,
                                        row.point_mode,
                                        row.point_indicator,
                                        row.point_value,
                                        row.interval)
        self.sys_log.debug("point_bool_long[self.config.bank.complete_index] : {}".format(point_bool_long[self.config.bank.complete_index]))
        self.sys_log.debug("point_bool_short[self.config.bank.complete_index] : {}".format(point_bool_short[self.config.bank.complete_index]))
                
                
        if row.zone_indicator == 'NULL': 
            zone_bool_long = point_bool_long
            zone_bool_short = point_bool_short
        else:   
            self.df_res, \
            zone_bool_long, \
            zone_bool_short = get_zone_bool(self.df_res, 
                                        row.zone_indicator,
                                        row.zone_value, 
                                        row.interval,)
            self.sys_log.debug("zone_bool_long[self.config.bank.complete_index] : {}".format(zone_bool_long[self.config.bank.complete_index]))
            self.sys_log.debug("zone_bool_short[self.config.bank.complete_index] : {}".format(zone_bool_short[self.config.bank.complete_index]))
            
        
        cross_over = zone_bool_long * point_bool_long
        cross_under = zone_bool_short * point_bool_short        
        self.sys_log.debug("cross_over[self.config.bank.complete_index] : {}".format(cross_over[self.config.bank.complete_index]))
        self.sys_log.debug("cross_under[self.config.bank.complete_index] : {}".format(cross_under[self.config.bank.complete_index]))
        
        
        point_index_long = np.argwhere(cross_over).ravel() # input for get_price_arr
        point_index_short = np.argwhere(cross_under).ravel()
        
        
        # set symbol & code.
        serial_number = ''.join(row.loc['priceBox_indicator':'RRratio'].astype(str).values)                
        self.code = "{}_{}_{}".format(self.symbol, serial_number, datetime.now().strftime('%Y%m%d%H%M%S%f'))       
            

        if self.position == 'LONG':
            if cross_over[self.config.bank.complete_index] or debugging:
                self.side_open = 'BUY'
                
                msg = f"{self.code} {self.side_open}"
                self.push_msg(msg)
        else:
            if cross_under[self.config.bank.complete_index] or debugging:
                self.side_open = 'SELL' 
                
                msg = f"{self.code} {self.side_open}"
                self.push_msg(msg)
                    
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

                msg = "index_valid_bool : {}".format(index_valid_bool)
                self.sys_log.debug(msg)
                self.push_msg(msg)
                
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
                    
                    RRratio_adj_fee = self.df_res['RRratio_adj_fee'].to_numpy() 
                    msg = "RRratio_adj_fee[self.config.bank.complete_index] : {}".format(RRratio_adj_fee[self.config.bank.complete_index])
                    self.sys_log.debug(msg)            
                    self.push_msg(msg)
                    
                    RRratio_adj_fee_category = self.df_res['RRratio_adj_fee_category'].to_numpy()
                    self.sys_log.debug("RRratio_adj_fee_category[self.config.bank.complete_index] : {}".format(RRratio_adj_fee_category[self.config.bank.complete_index]))
                    self.sys_log.debug("self.RRratio.split(';') : {}".format(self.RRratio.split(';')))
                    
                    # msg = f"self.df_res: {self.df_res.tail(2).to_string()}"
                    # self.sys_log.debug(msg)            
                    # self.push_msg(msg)
                    
                    if not (RRratio_adj_fee_category[self.config.bank.complete_index] in self.RRratio.split(';') or debugging):
                        self.side_open = np.nan                      
                else:
                    self.side_open = np.nan  
                    
                self.sys_log.debug("self.side_open : {}".format(self.side_open))
                    
            except Exception as e:
                msg = "Error in get_priceBox, get_price_arr : {}".format(e)
                self.sys_log.error(msg)
                self.push_msg(msg)
                # continue 
                
            self.sys_log.debug("LoopTableCondition : elasped time, get_priceBox, get_price_arr : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
            if not pd.isnull(self.side_open): # default = np.nan
                                
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
            
        self.path_df_res = "{}\\{}.csv".format(self.path_dir_df_res, self.symbol)
        self.df_res.to_csv(self.path_df_res, index=True) # leave datetime index.
        
        self.sys_log.debug("LoopTableCondition : elasped time, save df_res : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")

        
        # saving table takes some time, so we do not persist this phase more front.
            # check timestampLastUsed
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()   
        
        self.db_manager.replace_table(self.table_condition, self.table_condition_name)
        
        self.sys_log.debug("LoopTableCondition : elasped time, replace_table  : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        
        # # preventing API flood.
        time.sleep(self.config.term.loop_table_condition)


    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
    
    self.db_manager.replace_table(self.table_condition, self.table_condition_name, send=True)
    
    self.sys_log.debug("LoopTableCondition : elasped time, replace_table (send=True)  : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
  

def init_table_trade(self, ):

    """    
    v2.3
        apply TokenBucket.        
    v2.3.1
        move TokenBucket to function internal.
    v2.3.2
        use db_manger.
    v2.3.3 
        - replace
            margin_consistency to account_normality.
    v2.3.4 
        - modify
            divide balance into balance_origin (margin unconsidered.)
    v2.3.5
        - modify
            config info.
            add account to push_msg.
    v2.3.6
        - update
            phase reorder.
            use amount_entry as an input for leverage_limit.
    
    Last confirmed, 20241012 1432.
    """
           
    ##################################################
    # INIT
    ##################################################
    
    self.order_motion = 0  # for consuming time without trading.
    self.amount_min = 5
    
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
        
    self.side_close, \
    self.side_position = get_side_info(self, 
                                       self.side_open)
        
    self.sys_log.debug("InitTableTrade : elasped time, get_side_info : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")

     
    
    ##################################################
    # Pricing
    ##################################################
    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()   
    
    self.price_entry = get_price_entry(self, 
                                       self.df_res,
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

    
    ##################################################
    # balance.
    ##################################################
    
    # get balance
        # Stock 특성상, balance_min = self.price_entry 로 설정함.
        # open_data_dict.pop 은 watch_dict 의 일회성 / 영구성과 무관하다.    
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    account_normality, \
    self.balance_, \
    self.balance_origin = get_account_normality(self, 
                                        self.account,)
    
    self.sys_log.debug("InitTableTrade : elasped time, get_account_normality : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
    
    
    if account_normality:  
                
    
        ##################################################
        # quantity.
        ##################################################
        
        # get quantity_open   
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()  
        
        self.target_loss = self.balance_origin / self.divider # divider's target is balance_origin.
            
        price_diff = abs(self.price_entry - self.price_stop_loss)
        fee_entry = self.price_entry * self.config.broker.fee_market
        fee_stop_loss = self.price_stop_loss * self.config.broker.fee_market        
        self.loss = price_diff + fee_entry + fee_stop_loss
        
        self.quantity_open = self.target_loss / self.loss
        self.quantity_open = self.calc_with_precision(self.quantity_open, self.precision_quantity)
        
        # Calculate entry and exit amounts
            # amount_entry cannot be larger than the target_loss.
        self.amount_entry = self.price_entry * self.quantity_open
        
        self.sys_log.debug("InitTableTrade : elasped time, get margin : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")
        
        
        # func3 : check margin normality (> min balance)
        if self.amount_entry > self.amount_min:     
            
            # get leverage
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
                
            self.leverage_limit_user, \
            self.leverage_limit_server, \
            self.leverage = get_leverage_limit(self, 
                                            self.symbol, 
                                            self.amount_entry, 
                                            self.loss, 
                                            self.price_entry,
                                            )
            
            self.margin = self.amount_entry / self.leverage
            
            self.sys_log.debug("InitTableTrade : elasped time, get_leverage : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
            # set leverage
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()  
            
            if not self.config.bank.backtrade:
                                
                self.set_leverage(self.symbol,
                                self.leverage)      
                
            self.sys_log.debug("InitTableTrade : elasped time, set_leverage : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            
            # func4 : subtract margin if normal on 'balance'.
            self.table_account.loc[self.table_account['account'] == self.account, 'balance'] -= self.margin
            
        
            ##################################################
            # output bill.
            ##################################################
            
            self.sys_log.info("price_take_profit : {}".format(self.price_take_profit))
            self.sys_log.info("price_entry : {}".format(self.price_entry))
            self.sys_log.info("price_stop_loss : {}".format(self.price_stop_loss))
            self.sys_log.info("price_liquidation : {}".format(self.price_liquidation))
            self.sys_log.info("price_expiration : {}".format(self.price_expiration))
            
            self.sys_log.info("balance : {}".format(self.balance_))
            self.sys_log.info("balance_origin : {}".format(self.balance_origin))
            
            self.sys_log.info("target_loss : {}".format(self.target_loss))
            self.sys_log.info("loss : {}".format(self.loss))
            self.sys_log.info("quantity_open : {}".format(self.quantity_open))
            self.sys_log.info("amount_entry : {}".format(self.amount_entry))
            
            self.sys_log.info("leverage : {}".format(self.leverage))
            self.sys_log.info("margin : {}".format(self.margin))
            
        
        
            ##################################################
            # add to TableTrade
            ##################################################
            
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
            
            table_trade_new_row.account = self.account
            table_trade_new_row.balance = self.balance_                 # added.  
            table_trade_new_row.balance_origin = self.balance_origin    # added.  
            
            table_trade_new_row.target_loss = self.target_loss          # added.  
            table_trade_new_row.quantity_open = self.quantity_open 
            table_trade_new_row.amount_entry = self.amount_entry
            
            table_trade_new_row.leverage = self.leverage
            table_trade_new_row.margin = self.margin
        
            # self.sys_log.info("table_trade_new_row : \n{}".format(table_trade_new_row.iloc[0]))
        
            # append row.
            self.table_trade = pd.concat([self.table_trade, table_trade_new_row]).reset_index(drop=True)            
            
            self.sys_log.debug("InitTableTrade : elasped time, append row to TableTrade. : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")
            
            self.push_msg("{} {} row insert into TableTrade.".format(self.code, self.account))
        
        
    # preventing losing table_account info.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    self.db_manager.replace_table(self.table_account, self.table_account_name, send=True)
    self.db_manager.replace_table(self.table_trade, self.table_trade_name, send=True, mode=['UPDATE', 'DELETE'])
    
    self.sys_log.debug("InitTableTrade : elasped time, replace_table (send=True)  : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")
    


def loop_table_trade(self, ):
    """
    v1.2
        - update
            use stop_market

    20241101 0736.
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
    
        
        #############################################
        # set order, order_way (OPEN) (order not exists)
        #############################################
        
        if pd.isnull(self.orderId):
            
            self.sys_log.debug("------------------------------------------------")
            start_time = time.time()
            
            self.table_trade.at[idx, 'order'] = 1
            self.table_trade.at[idx, 'order_way'] = 'OPEN'

            if self.symbol not in self.price_market.keys(): # only once.
                
                self.websocket_client.agg_trade(symbol=self.symbol, id=1, callback=self.agg_trade_message_handler)
                self.set_position_mode(dualSidePosition='true')
                self.set_margin_type(symbol=self.symbol)
            
            self.sys_log.debug("LoopTableTrade : elasped time, set order, order_way (OPEN) : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")

        #  order exists.
        else:
        # if not pd.isnull(self.orderId): # this means, status is not None. (order in settled)

            # get_order_info. (update order_info)
                # get_order_info should have valid symbol & orderId.              
            self.order_info = get_order_info(self, 
                                       self.symbol,
                                       self.orderId)
            
            if self.order_info is not None: # order should be exist.
                

                #############################################
                # update table_trade
                #############################################
                
                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
                
                for k, v in self.order_info.items():
                    self.table_trade.at[idx, k] = v
                    
                self.sys_log.debug("LoopTableTrade : elasped time, update table_trade : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")
                # display(self.table_trade)    
                
                
                #############################################
                # check status
                #############################################
                
                if self.status_prev != self.order_info['status']:                    
                
                    # set statusChangeTime 
                        # check if status has been changed.
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
                        
                    
                    # logging : transfer rows to Table - log.   
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                                     
                    self.table_log = self.table_log.append(self.table_trade.loc[[idx]], ignore_index=True) # Append the new row to table_log
                    self.table_log['id'] = self.table_log.index + 1 # Update 'id' column to be the new index + 1
                    self.db_manager.replace_table(self.table_log, self.table_log_name)
                    
                    self.sys_log.debug("LoopTableTrade : elasped time, replace_table : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")  
                    # display(self.table_trade)

                
                #############################################
                # set order, order_way (for CLOSE)
                #############################################
                
                if row.order_way == 'OPEN':         
                    if (abs(float(self.order_info['executedQty']) / float(self.order_info['origQty'])) >= self.config.bank.quantity_open_exec_ratio) or self.order_info['status'] == 'FILLED':           
                        
                        self.sys_log.debug("------------------------------------------------")
                        start_time = time.time()
                        
                        self.table_trade.at[idx, 'order'] = 1
                        self.table_trade.at[idx, 'order_way'] = 'CLOSE'
                        
                        self.sys_log.debug("LoopTableTrade : elasped time, set order, order_way (CLOSE) : %.4fs" % (time.time() - start_time)) 
                        self.sys_log.debug("------------------------------------------------")
                        # display(self.table_trade)
                
                

                #############################################
                # check expiration for order_open.
                #     set price_reatlime
                #         for, open expiry
                #############################################
                
                # row.side_position has 2 state (LONG / SHORT) in status below 2 case.
                if row.order_way == 'OPEN' and self.order_info['status'] in ['NEW', 'PARTIALLY_FILLED']:
                
                    self.price_realtime = get_price_realtime(self, self.symbol)
                    self.sys_log.debug("self.price_realtime : {}".format(self.price_realtime))

                    if not pd.isnull(self.price_realtime):
                        
                        price_realtime_prev = self.table_trade.at[idx, 'price_realtime']
                        
                        slippage_margin = 0.1
                        lower_bound = price_realtime_prev * (1 - slippage_margin)
                        upper_bound = price_realtime_prev * (1 + slippage_margin)                                        
                        self.sys_log.debug(f"price_realtime_prev: {price_realtime_prev:.6f}, lower_bound: {lower_bound:.6f}, upper_bound: {upper_bound:.6f}")
                                                                                    
                        
                        if pd.isnull(price_realtime_prev) or (lower_bound <= self.price_realtime <= upper_bound):         
                                            
                            self.table_trade.at[idx, 'price_realtime'] = self.price_realtime                    
                    
                            self.sys_log.debug("------------------------------------------------")
                            start_time = time.time()  
                                    
                            self.expired = check_expiration(self.side_position, 
                                                            self.price_realtime, 
                                                            self.price_expiration)
                                
                            if self.expired:                                                         
                                quantity_unexecuted = get_quantity_unexecuted(self, 
                                                                            self.symbol,
                                                                            self.orderId)  
                                
                                self.table_account.loc[self.table_account['account'] == row.account, 'balance'] +=(quantity_unexecuted * row.price_entry / row.leverage)
                                
                                # Todo, messeage alertion needed ?
                                self.push_msg("{} has been expired.".format(self.code))
                                # self.user_text = 'watch'  # allowing message.
                                
                            self.sys_log.debug("LoopTableTrade : elasped time, check expiration for order_open : %.4fs" % (time.time() - start_time)) 
                            self.sys_log.debug("------------------------------------------------")
        
                    
                        # check stop_loss
                            # get order_market_on
                            # order_way == 'CLOSE' : stop_loss / liquidation condition.
                        # elif row.order_way == 'CLOSE':
                        #     self.sys_log.debug("------------------------------------------------")
                        #     start_time = time.time()  
                            
                        #     self.order_market_on = check_stop_loss(self,
                        #                                         self.side_open,
                        #                                         self.price_realtime,
                        #                                         self.price_liquidation,
                        #                                         self.price_stop_loss)
                            
                        #     self.sys_log.debug("LoopTableTrade : elasped time, check_stop_loss : %.4fs" % (time.time() - start_time)) 
                        #     self.sys_log.debug("------------------------------------------------")
            
        
        # order_limit
        if not self.config.bank.backtrade and not self.order_motion:

           # reflect updated row.
           if self.table_trade.at[idx, 'order'] == 1:

                self.sys_log.debug("------------------------------------------------")
                start_time = time.time()
               
                # public set.
                order_type = 'LIMIT' # fixed for order_limit.
                side_position = row.side_position
            
            
                ##############################
                # preprocessing
                ##############################
                                
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
            
            
                ##############################
                # ordering
                ##############################
                
                # order limit
                self.order_result, \
                self.error_code = order_limit(self, 
                                            self.symbol,
                                            side_order, 
                                            side_position, 
                                            price, 
                                            quantity)
               
                # normal state : self.error_code = 0
                if not self.error_code:
                    # self.table_trade.at[idx, 'orderId'] = str(self.order_result['orderId']) # this is key for updating order_info (change to OPEN / CLOSE orderId)
                    self.table_trade.at[idx, 'orderId'] = self.order_result['orderId'] # this is key for updating order_info (change to OPEN / CLOSE orderId)
                    
                else:
                    self.table_trade.at[idx, 'remove_row'] = 1
                    
                    # return sequence duplicated with under 'remove_row' == 1 phase, so we dont need below code.
                    # if self.table_trade.at[idx, 'order_way'] == 'OPEN': # if open fail, deposit withdrew margin.
                    #     self.table_account.loc[self.table_account['account'] == row.account, 'balance'] += row.margin
                    
                self.sys_log.debug("LoopTableTrade : elasped time, order_limit : %.4fs" % (time.time() - start_time)) 
                self.sys_log.debug("------------------------------------------------")
                
                
                # order_stop_market
                if self.table_trade.at[idx, 'order_way'] == 'CLOSE':
                    
                    self.sys_log.debug("------------------------------------------------")
                    start_time = time.time()
                                   
                    self.order_result, \
                    self.error_code = order_stop_market(self, 
                                                 self.symbol,
                                                 side_order,
                                                 side_position,
                                                 row.price_stop_loss,
                                                 quantity)
                    
                    # normal state : self.error_code = 0
                    if not self.error_code:
                        self.table_trade.at[idx, 'orderId'] = self.order_result['orderId'] # this is key for updating order_info (change to CLOSE orderId)       
                                                 
                    self.sys_log.debug("LoopTableTrade : elasped time, order_market : %.4fs" % (time.time() - start_time)) 
                    self.sys_log.debug("------------------------------------------------")
                    

        #############################################
        # remove_rows & calc. income, profit.
        #############################################
        
        if self.table_trade.at[idx, 'remove_row'] == 1: 
                 
            # drop rows
                # this phase should be placed in the most below, else order_ will set value as np.nan in invalid rows.                
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


            # calc. income, profit.
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
                                                    currency="USDT")
            
            self.table_account.loc[self.table_account['account'] == row.account, 'balance'] += row.margin
            self.table_account.loc[self.table_account['account'] == row.account, 'balance'] += self.income
            
            if self.config.bank.profit_mode == 'COMPOUND':
                self.table_account.loc[self.table_account['account'] == row.account, 'balance_origin'] += self.income
            
            self.config.bank.income_accumulated = self.income_accumulated
            self.config.bank.profit_accumulated = self.profit_accumulated
            
            
            code_split = self.code.split('_')
            condition = code_split[1]

            if len(code_split) == 3 and len(condition) > 5: # reject idx placed instead conndition.
                
                self.config.bank.condition_res.setdefault(condition, {
                    'income_accumulated': 0,
                    'profit_accumulated': 0
                })
                
                self.config.bank.condition_res[condition]['income_accumulated'] += self.income
                self.config.bank.condition_res[condition]['profit_accumulated'] += self.profit                
                
            
            with open(self.path_config, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            self.sys_log.debug("LoopTableTrade : elasped time, get_income_info : %.4fs" % (time.time() - start_time)) 
            self.sys_log.debug("------------------------------------------------")  
            # display(self.table_trade)


        # preventing losing orderId      
        self.sys_log.debug("------------------------------------------------")
        start_time = time.time()
        
        self.db_manager.replace_table(self.table_account, self.table_account_name)
        self.db_manager.replace_table(self.table_trade, self.table_trade_name, mode=['UPDATE', 'DELETE'])
        
        self.sys_log.debug("LoopTableTrade : elasped time, replace_table : %.4fs" % (time.time() - start_time)) 
        self.sys_log.debug("------------------------------------------------")  

        
        # # preventing API flood.
        time.sleep(self.config.term.loop_table_trade)

    # use send=True instead time.sleep for API flood.
    self.sys_log.debug("------------------------------------------------")
    start_time = time.time()
    
    self.db_manager.replace_table(self.table_account, self.table_account_name, send=True)
    self.db_manager.replace_table(self.table_trade, self.table_trade_name, send=True, mode=['UPDATE', 'DELETE'])
    self.db_manager.replace_table(self.table_log, self.table_log_name, send=True)
    
    self.sys_log.debug("LoopTableTrade : elasped time, replace_table (send=True) : %.4fs" % (time.time() - start_time)) 
    self.sys_log.debug("------------------------------------------------")  
