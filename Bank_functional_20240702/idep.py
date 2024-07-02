from funcs.public.indicator_ import *


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


def get_leverage_limit_by_symbol(data):
    
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
    


def set_quantity(table_trade_result, 
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

    
    leverage_limits = get_leverage_limit_by_symbol(data)
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
    - table_trade_result (pd.DataFrame): DataFrame containing trade results with 'timestamp_entry' column.
    - train_ratio (float): Ratio of data to be used for training. Default is 0.7 (70%).

    Returns:
    - pd.DataFrame: DataFrame with an additional 'dataType' column indicating 'TRAIN' or 'TEST'.
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



def get_amount_agg(trades_df):
    
    """
    Calculate the minimum amount required to execute all trades successfully, considering overlapping trades.

    Parameters:
    trades_df (pd.DataFrame): A DataFrame containing 'idx_entry', 'idx_exit', and 'amount_entry_adj_leverage' columns.

    Returns:
    float: The minimum amount required to execute all trades.
    """
    # Find the maximum exit index
    max_idx = trades_df["idx_exit"].max()

    # Initialize an array to store the required amount at each time unit
    amount_per_time_unit = np.zeros(max_idx + 1)

    # Accumulate the amount_entry_adj_leverage for each trade over its duration
    for _, row in trades_df.iterrows():
        amount_per_time_unit[row["idx_entry"]:row["idx_exit"] + 1] += row["amount_entry_adj_leverage"]

    # The minimum required amount to execute all trades is the maximum value in the amount_per_time_unit array
    min_required_amount = amount_per_time_unit.max()

    return min_required_amount


def get_periodic_profit(table_trade_result_agg, profit, mode='SIMPLE'):
    """
    Calculate daily, monthly, and yearly profit rates based on simple or compound interest.
    
    Parameters:
    table_trade_result_agg (pd.DataFrame): DataFrame containing trade results with timestamp entries and exits.
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
                mode_profit='SIMPLE',
                show_figure=False,
                save_figure=False):
    """
    v1.1
        add target_loss_pct
    v1.2
        simplified by gpt.
        modify get_amount_agg algorithm which uses index_entry & exit instead timemap.

    last confirmed at, 20240630 1805.
    """

    # # Helper function to convert interval to number of minutes
    # def itv_to_number(interval):
    #     interval_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    #     return interval_map.get(interval, 1)
    
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

    
    map_amount_agg_max = get_amount_agg(table_trade_result_agg)

    profit_pct_daily, \
    profit_pct_monthly, \
    profit_pct_yearly = get_periodic_profit(table_trade_result_agg, 
                                        profit_pct_final,
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
        "{}".format(symbol_extracted.tolist()),
        symbol_extracted_len,
        symbol_extracted_len_pct,
        frequencyTotal,
        frequencyMean,
        winRatio,
        profit_pct_final,
        profit_pct_daily,       # Changed to profit_pct_daily
        profit_pct_monthly,     # Changed to profit_pct_monthly
        profit_pct_yearly,      # Changed to profit_pct_yearly
        mean_return,
        std_return,
        max_drawdown,
        max_drawdown_scaled,
        sharpe_ratio,
        sortino_ratio,
        profit_factor,
        mse,
        r2,
        map_amount_agg_max,
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
        f"symbol_extracted_len : {symbol_extracted_len}\n"
        f"symbol_extracted_len_pct : {symbol_extracted_len_pct:.2f}\n"
        f"frequencyTotal : {frequencyTotal}\n"
        f"frequencyMean : {frequencyMean:.2f}\n"
        f"winRatio : {winRatio:.2f}\n"
        f"profit_pct_final : {profit_pct_final:.2f}\n"
        f"profit_pct_daily : {profit_pct_daily:.2f}\n"          # Changed to profit_pct_daily
        f"profit_pct_monthly : {profit_pct_monthly:.2f}\n"      # Changed to profit_pct_monthly
        f"profit_pct_yearly : {profit_pct_yearly:.2f}\n"        # Changed to profit_pct_yearly
        f"mean_return : {mean_return:.2f}\n"
        f"std_return : {std_return:.2f}\n"
        f"max_drawdown : {max_drawdown:.2f}\n"
        f"max_drawdown_scaled : {max_drawdown_scaled:.2f}\n"
        f"sharpe_ratio : {sharpe_ratio:.2f}\n"
        f"sortino_ratio : {sortino_ratio:.2f}\n"
        f"profit_factor : {profit_factor:.2f}\n"
        f"mse : {mse:.2f}\n"
        f"r2 : {r2:.2f}\n"
        f"map_amount_agg_max : {map_amount_agg_max:.2f}"
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
    #         "map_amount_agg_max"
    #         ]


    # # Generate title string from row and labels
    # title = "\n".join([f"{label} : {value}" if not isinstance(value, float) else f"{label} : {value:.2f}" for label, value in zip(row_labels, row)])
    # print(title)


    if show_figure:
        fig = plt.figure()
        plt.step(np.arange(len(table_trade_result_agg)), profit_pct_cum)
        
        # Add vertical lines to mark the change points between TRAIN and TEST data        
        change_indices = np.where(table_trade_result_agg['dataType'].values[:-1] != table_trade_result_agg['dataType'].values[1:])[0] + 1
        for idx in change_indices:
            plt.axvline(x=idx, color='r', linestyle='--', linewidth=0.5)
            
        plt.title(title, y=-0.1, fontsize=10)

        if save_figure:
            fig.tight_layout()
            # plt.subplots_adjust(left=0.1, right=0.1, bottom=0.0, top=0.0, wspace=0.2, hspace=0.2)
            payload = f"{interval}_{target_loss}_{target_loss_pct}_{target_leverage}_{position}_{RRratio_adj_fee_category}_{threshold_winRatio}_{threshold_frequency}_{threshold_frequencyTotal}"
            plt.savefig(os.path.join(path_dir_save_fig, f"{payload}.png"))
        else:
            plt.show()

        plt.close('all')

    return row




