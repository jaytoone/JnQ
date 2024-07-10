import os 

from funcs.public.indicator_ import *
from funcs.public.broker import itv_to_number
from funcs.public.plot_check import *

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import re
import gc



def get_point_index(df, 
                    mode,
                    indicator,  
                    point_value,
                    interval,
                    return_bool=False):

    """
    v2.0
        rearrange later.
    v2.1
        integrate input, output.
        add return_bool for Bank.

    last confirmed at, 20240627 1957.
    """

    cross_over = []
    cross_under = []
    
    indicator_value = np.nan
    
    if mode == 'CROSS':
        if indicator == 'II':
            df = get_II(df, period=point_value)
            
            indicator_value = df.iiSource.to_numpy()
            indicator_value_back_1 = df.iiSource.shift(1).to_numpy()
            base_value = 0
              
            cross_over = np.where((indicator_value > base_value) & (base_value > indicator_value_back_1), 1, 0)
            cross_under = np.where((indicator_value < base_value) & (base_value < indicator_value_back_1), 1, 0)    

        
        elif indicator == 'DC':            
            df = get_DC(df, period=point_value, interval=interval)
            
            # indicator_value = df['DC_{}{}_upper'.format(interval, point_value).to_numpy()
            DC_upper_back_1 = df['DC_{}{}_upper'.format(interval, point_value)].shift(1).to_numpy()
            DC_upper_back_2 = df['DC_{}{}_upper'.format(interval, point_value)].shift(2).to_numpy()
            # indicator_value = df['DC_{}{}_lower'.format(interval, point_value).to_numpy()
            DC_lower_back_1 = df['DC_{}{}_lower'.format(interval, point_value)].shift(1).to_numpy()
            DC_lower_back_2 = df['DC_{}{}_lower'.format(interval, point_value)].shift(2).to_numpy()
            # base_value = df.close.to_numpy()
    
            close = df.close.to_numpy()
            close_back_1 = df.close.to_numpy()
            
            cross_over = np.where((close > DC_upper_back_1), 1, 0) # close cannot be higher / lower than DC.
            cross_under = np.where((close < DC_lower_back_1), 1, 0)

        
        elif indicator == 'CCI':  
            df = get_CCI(df, period=point_value, interval=interval) 

            indicator_name = "CCI_{}{}".format(interval, point_value)
            
            indicator_value = df[indicator_name].to_numpy()
            indicator_value_back_1 = df[indicator_name].shift(1).to_numpy()
            base_value = 0
              
            cross_over = np.where((indicator_value > base_value) & (base_value > indicator_value_back_1), 1, 0)
            cross_under = np.where((indicator_value < base_value) & (base_value < indicator_value_back_1), 1, 0)    

        
        elif indicator == 'BB':
            df = get_BB(df, point_value, 1, interval, level=2)
            
            BB_upper = df['BB_{}{}_upper'.format(interval, point_value)].to_numpy() 
            BB_lower = df['BB_{}{}_lower'.format(interval, point_value)].to_numpy()
    
            close = df.close.to_numpy()            
            price_open = df.open.to_numpy()
            
            cross_over = np.where((close > BB_upper) & (BB_upper > price_open), 1, 0)
            cross_under = np.where((close < BB_lower) & (BB_lower < price_open), 1, 0)
            

    if return_bool:
        return df, cross_over, cross_under
    else:
        point_index_long = np.argwhere(cross_over).ravel()
        point_index_short = np.argwhere(cross_under).ravel()    
        
        return df, point_index_long, point_index_short


def add_zone(df, 
             indicator,
             zone_value, 
             interval,
             point_index_long,
             point_index_short):

    if indicator == 'MA':
        df = get_MA(df, 
                   zone_value, 
                   interval)
    
    
        close = df['close'].to_numpy()
        MA1 = df['MA_{}{}'.format(interval, zone_value)].to_numpy()
        
        zone_bool_long = close[point_index_long] > MA1[point_index_long]
        zone_bool_short = close[point_index_short] < MA1[point_index_short] 
        
        point_index_long = point_index_long[zone_bool_long]
        point_index_short = point_index_short[zone_bool_short]

    return df, point_index_long, point_index_short


def get_priceBox(df,
                 indicator,
                 priceBox_value,
                 interval,
                 side_open):
    
    """
    v2.0
        remove spread_arr.
        add side_open BUY.
    v2.1
        divide by side_open
    
    last confirmed at, 20240626 0954.
    """
    
    # public.
    close = df.close.to_numpy()

    if indicator == 'DC':    
        df = get_DC(df, 
                     priceBox_value, 
                     interval=interval)    

        # public.
        priceBox_upper = df['DC_{}{}_upper'.format(interval, priceBox_value)].to_numpy()
        priceBox_lower = df['DC_{}{}_lower'.format(interval, priceBox_value)].to_numpy()

    
    elif indicator == 'BB':        
        # df = get_BB(df, priceBox_value, 1, interval, level=2) # temporary, cause using point_indiator 'BB'.

        if side_open == 'BUY':
            priceBox_upper = df['BB_{}{}_upper2'.format(interval, priceBox_value)].to_numpy() 
            priceBox_lower = df['BB_{}{}_base'.format(interval, priceBox_value)].to_numpy()
        else:
            priceBox_upper = df['BB_{}{}_base'.format(interval, priceBox_value)].to_numpy()
            priceBox_lower = df['BB_{}{}_lower2'.format(interval, priceBox_value)].to_numpy() 

    return priceBox_upper, priceBox_lower, close


def get_price_arr(side_open, 
                 box_upper, 
                 box_lower,
                 close,
                 point_index_short,
                 point_index_long):

    """
    v1.1
        add price_validation

    last confirmed at, 20240627 2032.
    """
    
    
    if side_open == 'SELL':    
        price_take_profit_arr = box_lower[point_index_short]
        price_entry_arr = close[point_index_short]
        price_stop_loss_arr = box_upper[point_index_short]
        
        index_valid_bool = (price_take_profit_arr < price_entry_arr) & (price_entry_arr < price_stop_loss_arr)        
    else:
        price_take_profit_arr = box_upper[point_index_long]
        price_entry_arr = close[point_index_long]
        price_stop_loss_arr = box_lower[point_index_long]     

        index_valid_bool = (price_take_profit_arr > price_entry_arr) & (price_entry_arr > price_stop_loss_arr)

    
    return price_take_profit_arr, price_entry_arr, price_stop_loss_arr, index_valid_bool


def get_leverage_limit_by_symbol(self, ):
    
    server_time = self.time()['serverTime']
    data  = self.leverage_brackets(
                         recvWindow=6000, 
                         timestamp=server_time)
    
    max_leverage = {}  # Dictionary to store max initialLeverage for each symbol    
    
    # Iterate over each symbol's data
    for item in data:
        symbol = item['symbol']
        brackets = item['brackets']
        
        # Initialize max_leverage with the first bracket's initialLeverage
        max_initial_leverage = brackets[0]['initialLeverage']
        
        # Iterate through brackets to find the maximum initialLeverage
        for bracket in brackets:
            if bracket['initialLeverage'] > max_initial_leverage:
                max_initial_leverage = bracket['initialLeverage']
        
        # Store the max initialLeverage for the symbol
        max_leverage[symbol] = max_initial_leverage
    
    return max_leverage
    



def get_table_trade_result(bank, point_index_short, point_index_long, priceBox_indicator, priceBox_value, interval):
   
    """
    v1.1
        add spread_arr
    v1.2
        remove quantity info from table.
        reject status=DataEnd
        add symbol for winRatio calibration.
        add timestamp for amount agg. check.
    v1.3
        add timestamp column and timestamp_entry & exit derives from it.
    v1.4
        add LONG position.
            LONG SHORT one table.
        apply functional mode.
        
        v1.4.1
            remove fee.        
        v1.4.2
            add get_priceBox (v2.1)
            modify logical miss for TP / SL in LONG.
            modify zip to list(zip) for price_validation.

        this funciton only allow 'bank', modify it...
    
    last confirmed at, 20240708 1053.
    """

    data_len = len(bank.df_res)
    high = bank.df_res.high.to_numpy()
    low = bank.df_res.low.to_numpy()

    data_list = []

    for side_open in ['BUY', 'SELL']:  # Iterate through both 'SELL' and 'BUY' sides

        if side_open == 'SELL':
            point_index = point_index_short
        else:
            point_index = point_index_long

        priceBox_upper, priceBox_lower, close = get_priceBox(bank.df_res,
                                                             priceBox_indicator,
                                                             priceBox_value,
                                                             interval,
                                                             side_open)

        price_take_profit_arr, price_entry_arr, price_stop_loss_arr, index_valid_bool = get_price_arr(side_open,
                                                                                                     priceBox_upper,
                                                                                                     priceBox_lower,
                                                                                                     close,
                                                                                                     point_index_short,
                                                                                                     point_index_long)

        trade_arr_valid = np.array(list(zip(point_index, price_take_profit_arr, price_entry_arr, price_stop_loss_arr)), 
                                   dtype=[('0', int), ('1', float), ('2', float), ('3', float)])[index_valid_bool]

        for idx_entry, price_take_profit, price_entry, price_stop_loss in trade_arr_valid:
            status = None
            idx_realtime = idx_entry + 1

            while True:
                if idx_realtime >= data_len:
                    status = 'DataEnd'
                    break

                if side_open == 'SELL':
                    if low[idx_realtime] < price_take_profit:
                        status = 'TP'
                        break
                    elif high[idx_realtime] >= price_stop_loss:
                        status = 'SL'
                        break
                else:
                    if high[idx_realtime] > price_take_profit:
                        status = 'TP'
                        break
                    elif low[idx_realtime] <= price_stop_loss:
                        status = 'SL'
                        break

                idx_realtime += 1
     
            if status in ['TP', 'SL']:
                position = 'SHORT' if side_open == 'SELL' else 'LONG'
                price_exit = price_take_profit if status == 'TP' else price_stop_loss
                
                row = [bank.symbol, position, status, idx_entry, idx_realtime, price_take_profit, price_entry, price_stop_loss, price_exit]
                
                data_list.append(row)

    columns = ['symbol', 'position', 'status', 'idx_entry', 'idx_exit', 'price_take_profit', 'price_entry', 'price_stop_loss', 'price_exit']
    table_trade_result = pd.DataFrame(data_list, columns=columns)

    timestamp = bank.df_res['timestamp'].to_numpy()
    table_trade_result['timestamp_entry'] = timestamp[table_trade_result.idx_entry]
    table_trade_result['timestamp_exit'] = timestamp[table_trade_result.idx_exit]

    return table_trade_result





def set_quantity(table_trade_result,                  
                 leverage_limits,
                 target_loss, 
                 target_loss_pct, 
                 target_leverage=None, 
                 fee_entry=0.0002, 
                 fee_exit=0.0005,
                 unit_RRratio_adj_fee=np.arange(0, 2, 0.1),
                 leverage_rejection=False,
                 ):

    """
    v1.1
        define ~ to_numpy() as a var. for keep usage
        add RRratio_adj_fee
        modify income & commission to income & commission
    v1.2
        move RRratio to anal phase.
        set target_loss.
            income calculated from quantity.
    v1.3
        modify 'income' to support LONG & SHORT one table.
        add loss_pct
        add profit.
    v1.4
        rearrange with to_numpy()
        
        v1.4.1
            "fee will be placed in here."
            apply leverage_limit_user = None.
        v1.4.2
            modify to functional.
            adj. leverage_limit_user min = 1.
            modify loss_pct (multiply 100)
            modify leverage_limit_user logic.
            modify concept of target_loss & target_loss_concept.
            modify input params to fee_entry & fee_exit.
            include RRratio by default.
                considering leverage_rejection & functional.
            add leverage_limit_server.
    
    last confirmed at, 20240630 1939.
    """

    # Convert relevant columns to numpy arrays for calculations
    price_take_profit = table_trade_result.price_take_profit.to_numpy()
    price_entry = table_trade_result.price_entry.to_numpy()
    price_stop_loss = table_trade_result.price_stop_loss.to_numpy()
    price_exit = table_trade_result.price_exit.to_numpy()
    position = table_trade_result.position.to_numpy()

    # Set fee values and include them in the DataFrame
    table_trade_result['fee_entry'] = fee_entry
    table_trade_result['fee_exit'] = fee_exit
    fee_entry = table_trade_result.fee_entry.to_numpy()
    fee_exit = table_trade_result.fee_exit.to_numpy()

    # Calculate loss and loss percentage for quantity = 1
    loss = abs(price_entry - price_stop_loss) + (price_entry * fee_entry + price_stop_loss * fee_exit)
    loss_pct = loss / price_entry * 100
    table_trade_result['loss'] = loss
    table_trade_result['loss (%)'] = loss_pct

    # Determine quantity
    if target_loss is None:
        amount_entry = 15  # default minimum amount
        quantity = amount_entry / price_entry
    else:
        quantity = target_loss / loss

    table_trade_result['quantity'] = quantity

    # Calculate entry and exit amounts
    amount_entry = price_entry * quantity
    amount_exit = price_exit * quantity

    # Calculate commission and income
    commission = amount_entry * fee_entry + amount_exit * fee_exit
    income = np.where(position == 'SHORT', amount_entry - amount_exit, amount_exit - amount_entry)

    table_trade_result['commission'] = commission
    table_trade_result['income'] = income
    table_trade_result['income - commission'] = income - commission
    table_trade_result['amount_entry'] = amount_entry

    
    leverage_limit_server = table_trade_result['symbol'].map(leverage_limits)
    table_trade_result['leverage_limit_server'] = leverage_limit_server

    # Calculate leverage limit and adjust leverage if needed
    leverage_limit_user = np.maximum(1, np.floor(100 / loss_pct).astype(int))
    table_trade_result['leverage_limit_user'] = leverage_limit_user

    leverage_limit = np.minimum(leverage_limit_server, leverage_limit_user)

    if target_loss_pct is None:
        leverage = np.minimum(target_leverage, leverage_limit) if target_leverage else leverage_limit
    else:
        leverage = np.minimum(np.maximum(1, np.floor(target_loss_pct / loss_pct).astype(int)), leverage_limit)

    table_trade_result['leverage'] = leverage
    table_trade_result['amount_entry_adj_leverage'] = amount_entry / leverage

    # Calculate profit and profit percentage
    profit = (income - commission) / amount_entry * leverage
    table_trade_result['profit'] = profit
    table_trade_result['profit (%)'] = profit * 100
    
        
    # zone (continuous value)
        # default
    table_trade_result['RRratio'] = abs(price_take_profit - price_entry) / abs(price_entry - price_stop_loss)
    table_trade_result['RRratio_adj_fee'] = (abs(price_take_profit - price_entry) - (price_entry * fee_entry + price_take_profit * fee_exit)) / (abs(price_entry - price_stop_loss) + (price_entry * fee_entry + price_stop_loss * fee_exit))
    table_trade_result['RRratio_adj_fee_category'] = pd.cut(table_trade_result['RRratio_adj_fee'], unit_RRratio_adj_fee, precision=0, duplicates='drop').astype(str)
    
    # Leverage rejection if needed
    if leverage_rejection:
        assert target_loss_pct is not None, "target_loss_pct must be provided if leverage_rejection is True."
        table_trade_result = table_trade_result[target_loss_pct > loss_pct]
        
    return table_trade_result




def add_datatype(table_trade_result, train_ratio=0.7):
    
    """
    Splits the table_trade_result DataFrame into train and test sets based on the timestamp.

    Parameters:
    # - table_trade_result (pd.DataFrame): DataFrame containing trade results with 'timestamp_entry' column.
    - train_ratio (float): Ratio of data to be used for training. Default is 0.7 (70%).

    Returns:
    # - pd.DataFrame: DataFrame with an additional 'dataType' column indicating 'TRAIN' or 'TEST'.
    """

    # Convert 'timestamp_entry' column to numpy array for calculations
    timestamp_entry = table_trade_result['timestamp_entry'].to_numpy()

    # Find minimum and maximum timestamps
    timestamp_min = timestamp_entry.min()
    timestamp_max = timestamp_entry.max()

    # Calculate the threshold timestamp for splitting into train and test sets
    timestamp_train = timestamp_min + (timestamp_max - timestamp_min) * train_ratio

    # Assign 'TRAIN' or 'TEST' labels based on the timestamp threshold
    table_trade_result['dataType'] = np.where(timestamp_entry < timestamp_train, 'TRAIN', 'TEST')

    return table_trade_result


def convert_to_position_single(df):

    """
    v2.0
        modify to timestamp mode.
            = using public index
                private index : idx_entry & exit.

    last confiremd at, 20240703 0732.
    """
    
    # Sort trades by entry index in ascending order
    df = df.sort_values(by='timestamp_entry').reset_index(drop=True)
    
    # List to store the result
    result = []
    
    # Variable to store the exit index of the current position
    current_timestamp_exit = -1
    
    for idx, row in df.iterrows():
        # Add trades that start after the current position's exit index
        if row['timestamp_entry'] > current_timestamp_exit:
            result.append(row)
            current_timestamp_exit = row['timestamp_exit']
    
    return pd.DataFrame(result)




def get_minimum_assets(trade_data):
    
    """
    v1.0
        arrange income-commission by timemap.
    v1.1
        simplified by gpt.
        
        v1.1.1
            vectorization. (lower latency)
            reorder output.
        
    last confirmed at, 20240708 1655.
    """    
    
    # Normalize timestamps
    timestamp_min = trade_data['timestamp_entry'].min()
    trade_data['timemap_entry'] = (trade_data['timestamp_entry'] - timestamp_min) // 60
    trade_data['timemap_exit'] = (trade_data['timestamp_exit'] - timestamp_min) // 60
       
    # Combine entry and exit events into a single DataFrame
    events = pd.concat([
        trade_data[['timemap_entry', 'amount_entry_adj_leverage']].rename(columns={'timemap_entry': 'timestamp', 'amount_entry_adj_leverage': 'amount'}).assign(event='entry'),
        trade_data[['timemap_exit', 'amount_entry_adj_leverage', 'income - commission']].rename(columns={'timemap_exit': 'timestamp', 'amount_entry_adj_leverage': 'amount'}).assign(event='exit')
    ]).sort_values(by='timestamp').reset_index(drop=True)

    
    current_assets = np.where(events['event'] == 'entry', -events['amount'], events['amount'] + events['income - commission'].fillna(0))
    events['current_assets'] = current_assets

    # Calculate minimum assets required and maximum drawdown
    assets_over_time = current_assets.cumsum()
    assets_min_required = assets_over_time.min()
    max_drawdown = (assets_over_time - np.maximum.accumulate(assets_over_time)).min()
    
    plt.step(range(len(assets_over_time)), assets_over_time)
    plt.show()

    return events, assets_over_time, assets_min_required, max_drawdown




def get_periodic_profit(table_trade_result_agg, profit, mode='SIMPLE'):
    """
    Calculate daily, monthly, and yearly profit rates based on simple or compound interest.
    
    Parameters:
    # table_trade_result_agg (pd.DataFrame): DataFrame containing trade results with timestamp entries and exits.
    profit (float): The total profit in percentage.
    mode (str): Mode of calculation ('SIMPLE' or 'COMPOUND').
    
    Returns:
    tuple: Daily, monthly, and yearly profit rates in percentage.
    """
    
    # Calculate the total duration in seconds
    duration_seconds = table_trade_result_agg.timestamp_exit.max() - table_trade_result_agg.timestamp_entry.min()
    
    # Define constants
    SECONDS_IN_A_DAY = 86400
    SECONDS_IN_A_MONTH = 30 * SECONDS_IN_A_DAY
    SECONDS_IN_A_YEAR = 365 * SECONDS_IN_A_DAY
    
    # Convert duration to days, months, and years
    duration_days = duration_seconds / SECONDS_IN_A_DAY
    duration_months = duration_seconds / SECONDS_IN_A_MONTH
    duration_years = duration_seconds / SECONDS_IN_A_YEAR
    
    # Convert profit from percentage to decimal
    profit_decimal = profit / 100
    
    if mode == 'SIMPLE':
        # Calculate simple interest rates
        profit_daily = profit_decimal / duration_days * 100
        profit_monthly = profit_decimal / duration_months * 100
        profit_yearly = profit_decimal / duration_years * 100
    else:
        # Calculate compound interest rates
        profit_daily = ((1 + profit_decimal) ** (1 / duration_days) - 1) * 100
        profit_monthly = ((1 + profit_decimal) ** (1 / duration_months) - 1) * 100
        profit_yearly = ((1 + profit_decimal) ** (1 / duration_years) - 1) * 100

    return profit_daily, profit_monthly, profit_yearly
    

def get_output_row(table_trade_result_agg, 
                symbol_extracted, 
                symbol_extracted_len, 
                symbol_extracted_len_pct, 
                interval, 
                target_loss,
                target_loss_pct,
                target_leverage, 
                position, 
                RRratio_adj_fee_category,
                threshold_winRatio, 
                threshold_frequency, 
                threshold_frequencyTotal, 
                path_dir_save_fig,
                mode_position='SINGLE',
                mode_profit='SIMPLE',
                show_figure=False,
                save_figure=False):
    """
    v1.1
        add target_loss_pct
    v1.2
        simplified by gpt.
        modify get_amount_agg algorithm which uses index_entry & exit instead timemap.
        
        v1.2.1
            add divide bar on title.
    v1.3
        modify get_amount_agg to get_minimum_assets.
        
        v1.3.1
            add mode_position.

    last confirmed at, 20240708 2251.
    """
    
    # Calculate win ratio
    frequencyTotal = len(table_trade_result_agg)    
    winRatio = len(table_trade_result_agg[table_trade_result_agg.status == 'TP']) / frequencyTotal
    
    # Calculate cumulative and final profit percentage
    profit_pct_cum = np.cumsum(table_trade_result_agg['profit (%)'].to_numpy()) if mode_profit == 'SIMPLE' else (np.cumprod(table_trade_result_agg['profit'].to_numpy() + 1) - 1) * 100
    profit_pct_final = profit_pct_cum[-1] if profit_pct_cum.size > 0 else 0

    frequencyMean = frequencyTotal / symbol_extracted_len if symbol_extracted_len != 0 else 0

    # Calculate Sharpe and Sortino Ratios
    profit = table_trade_result_agg.profit.to_numpy()
    mean_return = np.mean(profit) if profit.size > 0 else 0
    std_return = np.std(profit) if profit.size > 0 else 0
    sharpe_ratio = (mean_return - 0) / std_return if std_return != 0 else 0
    downside_returns = profit[profit < 0]
    downside_deviation = np.std(downside_returns) if downside_returns.size > 0 else 0
    sortino_ratio = (mean_return - 0) / downside_deviation if downside_deviation != 0 else 0

    # Calculate Maximum Drawdown
    cumulative_returns = np.cumsum(profit)
    drawdowns = cumulative_returns - np.maximum.accumulate(cumulative_returns)
    max_drawdown = np.min(drawdowns) if drawdowns.size > 0 else 0
    cumulative_returns_min, cumulative_returns_max = cumulative_returns.min(), cumulative_returns.max()
    cumulative_returns_scaled = (cumulative_returns - cumulative_returns_min) / (cumulative_returns_max - cumulative_returns_min) if cumulative_returns_max - cumulative_returns_min != 0 else np.zeros_like(cumulative_returns)
    drawdowns_scaled = cumulative_returns_scaled - np.maximum.accumulate(cumulative_returns_scaled)
    max_drawdown_scaled = np.min(drawdowns_scaled) if drawdowns_scaled.size > 0 else 0

    # Calculate Profit Factor
    total_profit = np.sum(profit[profit > 0])
    total_loss = -np.sum(profit[profit < 0])
    profit_factor = total_profit / total_loss if total_loss != 0 else 0

    # Calculate MSE and R2
    x = np.linspace(0, 1, cumulative_returns_scaled.size)
    y_actual = x
    y_predicted = cumulative_returns_scaled
    mse = mean_squared_error(y_actual, y_predicted) if y_actual.size > 0 and y_predicted.size > 0 else 0
    r2 = r2_score(y_actual, y_predicted) if y_actual.size > 0 and y_predicted.size > 0 else 0

    if mode_profit == 'SIMPLE':        
        events, \
        assets_over_time, \
        assets_min_required, \
        assets_max_drawdown = get_minimum_assets(table_trade_result_agg)
        
        profit_pct_final_min = abs(assets_over_time[-1] / assets_max_drawdown) * 100
        profit_pct_final_max = abs(assets_over_time[-1] / assets_min_required) * 100
    else:
        assets_min_required = np.nan
        assets_max_drawdown = np.nan
        profit_pct_final_min = profit_pct_final
        profit_pct_final_max = profit_pct_final   
        
    assets_min_required_const = abs(assets_min_required / target_loss)
    assets_max_drawdown_const = abs(assets_max_drawdown / target_loss)
    
        
    profit_pct_daily, \
    profit_pct_monthly, \
    profit_pct_yearly = get_periodic_profit(table_trade_result_agg, 
                                        profit_pct_final,
                                        mode_profit) 
    profit_pct_daily_min, \
    profit_pct_monthly_min, \
    profit_pct_yearly_min = get_periodic_profit(table_trade_result_agg, 
                                        profit_pct_final_min,
                                        mode_profit)
    profit_pct_daily_max, \
    profit_pct_monthly_max, \
    profit_pct_yearly_max = get_periodic_profit(table_trade_result_agg, 
                                        profit_pct_final_max,
                                        mode_profit)

    
    row = [
        interval,
        target_loss,
        target_loss_pct,
        target_leverage,
        position,
        RRratio_adj_fee_category,
        threshold_winRatio,
        threshold_frequency,
        threshold_frequencyTotal,
        mode_position,
        
        "{}".format(symbol_extracted.tolist()),
        symbol_extracted_len,
        symbol_extracted_len_pct,
        
        frequencyTotal,
        frequencyMean,
        
        winRatio,
        
        assets_min_required_const,
        assets_max_drawdown_const,
        profit_pct_final,
        profit_pct_daily, 
        profit_pct_monthly,
        profit_pct_yearly,
        profit_pct_final_min,
        profit_pct_daily_min, 
        profit_pct_monthly_min,
        profit_pct_yearly_min,
        profit_pct_final_max,
        profit_pct_daily_max, 
        profit_pct_monthly_max,
        profit_pct_yearly_max,
        
        mean_return,
        std_return,
        max_drawdown,
        max_drawdown_scaled,
        sharpe_ratio,
        sortino_ratio,
        profit_factor,
        mse,
        r2,
    ]
    
    # Generate title string
    title = (
        f"interval : {interval}\n"
        f"target_loss : {target_loss}\n"
        f"target_loss_pct : {target_loss_pct}\n"
        f"target_leverage : {target_leverage}\n"
        f"position : {position}\n"
        f"RRratio_adj_fee_category : {RRratio_adj_fee_category}\n"
        f"threshold_winRatio : {threshold_winRatio:.2f}\n"
        f"threshold_frequency : {threshold_frequency}\n"
        f"threshold_frequencyTotal : {threshold_frequencyTotal}\n"
        f"mode_position : {mode_position}\n"
        f"mode_profit : {mode_profit}\n"
        f"symbol_extracted_len : {symbol_extracted_len}\n"
        f"symbol_extracted_len_pct : {symbol_extracted_len_pct:.2f}\n"
        f"------------------------\n"
        f"frequencyTotal : {frequencyTotal}\n"
        f"frequencyMean : {frequencyMean:.2f}\n"
        f"------------------------\n"
        f"winRatio : {winRatio:.2f}\n"
        f"------------------------\n"
        f"assets_min_required_const : {assets_min_required_const:.2f}\n"
        f"assets_max_drawdown_const : {assets_max_drawdown_const:.2f}\n"
        f"profit_pct_final : {profit_pct_final:.2f} ({profit_pct_final_min:.2f}~{profit_pct_final_max:.2f})\n"
        f"profit_pct_daily : {profit_pct_daily:.2f} ({profit_pct_daily_min:.2f}~{profit_pct_daily_max:.2f})\n"
        f"profit_pct_monthly : {profit_pct_monthly:.2f} ({profit_pct_monthly_min:.2f}~{profit_pct_monthly_max:.2f})\n"
        f"profit_pct_yearly : {profit_pct_yearly:.2f} ({profit_pct_yearly_min:.2f}~{profit_pct_yearly_max:.2f})\n"
        f"------------------------\n"
        f"mean_return : {mean_return:.2f}\n"
        f"std_return : {std_return:.2f}\n"
        f"max_drawdown : {max_drawdown:.2f}\n"
        f"max_drawdown_scaled : {max_drawdown_scaled:.2f}\n"
        f"sharpe_ratio : {sharpe_ratio:.2f}\n"
        f"sortino_ratio : {sortino_ratio:.2f}\n"
        f"profit_factor : {profit_factor:.2f}\n"
        f"mse : {mse:.2f}\n"
        f"r2 : {r2:.2f}"
    )
    
    # # Mapping row indices to variable names for title generation
    # row_labels = [
    #         "interval",
    #         "target_loss",
    #         "target_loss_pct",
    #         "target_leverage",
    #         "position",
    #         "RRratio_adj_fee_category",
    #         "threshold_winRatio",
    #         "threshold_frequency",
    #         "threshold_frequencyTotal",
    #         "symbol_extracted_len",
    #         "symbol_extracted_len_pct",
    #         "frequencyTotal",
    #         "frequencyMean",
    #         "winRatio",
    #         "profit_pct_final",
    #         "mean_return",
    #         "std_return",
    #         "max_drawdown",
    #         "max_drawdown_scaled",
    #         "sharpe_ratio",
    #         "sortino_ratio",
    #         "profit_factor",
    #         "mse",
    #         "r2",
    #         "assets_max_drawdown"
    #         ]

        
    if show_figure:
    
        fig = plt.figure(figsize=(10, 5))
    
        # Create a GridSpec layout with 2 columns
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.1)
    
        # Create a subplot for the main plot
        ax_plot = fig.add_subplot(gs[0])
        ax_plot.step(np.arange(len(table_trade_result_agg)), profit_pct_cum)
    
        # Add vertical lines to mark the change points between TRAIN and TEST data        
        change_indices = np.where(table_trade_result_agg['dataType'].values[:-1] != table_trade_result_agg['dataType'].values[1:])[0] + 1
        for idx in change_indices:
            ax_plot.axvline(x=idx, color='r', linestyle='--', linewidth=0.5)
    
        # Create a subplot for the title
        ax_title = fig.add_subplot(gs[1])
        ax_title.axis('off')
        ax_title.text(0, 0.5, title, ha='left', va='center', fontsize=8, wrap=True)
    
        # Apply constrained_layout to the figure
        # fig.constrained_layout()
        # fig.tight_layout()

        if save_figure:
            # payload = f"{interval}_{target_loss}_{target_loss_pct}_{target_leverage}_{position}_{RRratio_adj_fee_category}_{threshold_winRatio:.2f}_{threshold_frequency}_{threshold_frequencyTotal}_{mode_position}"
            payload = (
                    f"{interval}_{target_loss}_{target_loss_pct}_{target_leverage}_"
                    f"{position}_{RRratio_adj_fee_category}_{threshold_winRatio:.2f}_"
                    f"{threshold_frequency}_{threshold_frequencyTotal}_{mode_position}_{mode_profit}"
                    )
            plt.savefig(os.path.join(path_dir_save_fig, f"{payload}.png"))
        # else:
        plt.show()
        
        plt.close('all')

    return row






