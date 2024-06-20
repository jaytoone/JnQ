import numpy as np
import random


def adjust_time_span_v3(v, ts_train, bars_in_train, workers=20):
    """
    v2 --> v3
        1. return ts_idx_valid_sorted.
    """

    ts_train_close_ = ((ts_train - ts_train[0]) / 60).astype('uint64')[v]
    bars_in_train_close_ = bars_in_train.astype('uint32')[v]

    sorting_index = np.argsort(ts_train_close_)

    v_sorted_by_ts = np.array(v)[sorting_index]
    ts_train_close = ts_train_close_[sorting_index]
    bars_in_train_close = bars_in_train_close_[sorting_index]

    ts_idx_valid = []

    ts_map = np.zeros(int(ts_train_close[-1] + 1)).astype('uint64')

    for ts_i, (ts_start, ts_dur) in enumerate(zip(ts_train_close, bars_in_train_close)):
        ts_end = ts_start + ts_dur
        # print(ts_start, ts_end)
        # ts_map[ts_start:int(ts_end)] += 1
        if ts_map[ts_start:ts_end].max() < workers:
            ts_idx_valid.append(ts_i)
            ts_map[ts_start:ts_end] += 1

            # return v_sorted_by_ts[ts_idx_valid]
    return sorting_index[ts_idx_valid]


def adjust_time_span_v2(v, ts_train, bars_in_train, workers=20):
    """
    v1 --> v2
        1. allow workers recycling. = enable lower min_workers for getting higher pr.
    """

    ts_train_close_ = ((ts_train - ts_train[0]) / 60).astype('uint64')[v]
    bars_in_train_close_ = bars_in_train.astype('uint32')[v]

    sorting_index = np.argsort(ts_train_close_)

    v_sorted_by_ts = np.array(v)[sorting_index]
    ts_train_close = ts_train_close_[sorting_index]
    bars_in_train_close = bars_in_train_close_[sorting_index]

    ts_idx_valid = []

    ts_map = np.zeros(int(ts_train_close[-1] + 1)).astype('uint64')

    for ts_i, (ts_start, ts_dur) in enumerate(zip(ts_train_close, bars_in_train_close)):
        ts_end = ts_start + ts_dur
        # print(ts_start, ts_end)
        # ts_map[ts_start:int(ts_end)] += 1
        if ts_map[ts_start:ts_end].max() < workers:
            ts_idx_valid.append(ts_i)
            ts_map[ts_start:ts_end] += 1

    return v_sorted_by_ts[ts_idx_valid]


def adjust_time_span_v1(v, ts_train, workers=5, time_span_size=60 * 100):
    """
    algorithm validation.

    # print(sorted(v_ts_random)[:10])
    # print(sorted(v)[:10])

    # for vtr in v_ts_random:
    #     if vtr not in v:
    #         print("{} not in v".format(vtr))

    v1. v_concated 는 ts 가 다시 섞이기 때문에, 재정렬이 요구된다.
    """

    ts_train_close_ = ts_train[v]
    sorting_index = np.argsort(ts_train_close_)  # reindexed v_index
    # print(sorting_index.max(), sorting_index.min())
    # print("sorting_index :", sorting_index)

    v_sorted_by_ts = np.array(v)[sorting_index]  # reindexed v_index
    # print("v_sorted_by_ts : {}".format(v_sorted_by_ts))

    ts_train_close = ts_train_close_[sorting_index]
    ts_train_close -= min(ts_train_close)

    # plt.plot(ts_train_close)
    # plt.show()

    # break

    n = 0
    v_ts_random = []
    # ts_train_close_len = len(ts_train_close)

    while 1:
        # for ts_ in ts_train_close:
        n += 1
        time_span = time_span_size * n
        if time_span >= ts_train_close[-1]:
            break

        # 1. 한 time_span 내의 index 중 택 1 하는 알고리즘을 구성할 것.
        inspan_index = np.argwhere(ts_train_close < time_span).ravel()
        # print("inspan_index :", inspan_index)

        #    a. if time_span 내의 index 존재시,
        if len(inspan_index) > 0:
            inspan_index_last = inspan_index[-1]
            # print("inspan_index_last :", inspan_index_last)
            # print("ts_train_close[:inspan_index_last + 1] :", ts_train_close[:inspan_index_last + 1])

            # ts_train_session = ts_train_close[:inspan_index_lastes + 1]
            # sorting_index_inspan = sorting_index[:inspan_index_last + 1]
            v_sorted_by_ts_inspan = v_sorted_by_ts[:inspan_index_last + 1]
            # print("v_sorted_by_ts_inspan :", v_sorted_by_ts_inspan)

            # if inspan_index_lastes >= ts_train_close_len - 1:
            #     break

            #    i. sorting_index_inspan 중 택 1. (randomly)
            # if len(sorting_index_inspan) > 0:
            # choiced_index = np.random.choice(sorting_index_inspan)
            # choiced_index = np.random.choice(v_sorted_by_ts_inspan)
            # v_ts_random.append(np.random.choice(v_sorted_by_ts_inspan))
            # v_ts_random.append(v_sorted_by_ts_inspan[0])
            # v_ts_random += list(v_sorted_by_ts_inspan
            random.shuffle(v_sorted_by_ts_inspan)
            v_ts_random += list(v_sorted_by_ts_inspan[:workers])
            # v_ts_random += list(v_sorted_by_ts_inspan[0])

            #    ii. overlapping variable with remaining items.
            ts_train_close = ts_train_close[inspan_index_last + 1:]
            v_sorted_by_ts = v_sorted_by_ts[inspan_index_last + 1:]

            # print("inspan_index : {}".format(inspan_index))
            # print("choiced_index : {}".format(choiced_index))
            # # print(ts_train_close)
            # print()
            # break

    return v_ts_random


def set_dynamic_lvrg(out_data, en_data, ex_data, fee_data, target_pct=0.05, multiplier=100, log_base=10):
        
    loss_expected = abs((out_data / en_data) * (1 - 0.0006) - 1)
    # lvrg_expected = (target_pct / loss_expected).astype(int)  # flooring.
    lvrg_expected = (loss_expected * multiplier)#.astype(int)  # flooring.
    # lvrg_expected = 1
    print("lvrg_expected : {}".format(lvrg_expected))
    # plt.hist(lvrg_expected, bins=500)
    # plt.show()    
    lvrg_expected =  np.array(list(map(lambda x: math.log(x, log_base) if x > 0 else 1, lvrg_expected))).astype(int)
    print("lvrg_expected (log adj.) : {}".format(lvrg_expected))

    print("invalid leverage percentage : {:.2%}".format(len(loss_expected[lvrg_expected == 0]) / len(loss_expected)))
    
    y_data2 = (ex_data / en_data * (1 - fee_data) - 1) * lvrg_expected + 1
    y_data2[lvrg_expected == 0] = 1  # lvrg_rejection.
    return y_data2



def plot_pr(y_target_final, data_period):
    
    y_target_final[y_target_final <= 0] = 0
    y_target_final_len = len(y_target_final[y_target_final != 1])
    wr = (y_target_final > 1).sum() / y_target_final_len
    pr_cumprod = np.cumprod(y_target_final)

    plt.plot(pr_cumprod)    
    title_msg = "frq : {}\nwr : {:.2f}\npr : {:.2f}\nmdd : {:.2f}".format(y_target_final_len, wr, pr_cumprod[-1], mdd_v2(pr_cumprod))   
    title_msg += "\nday : {:.2f}\nmonth : {:.2f}\nyear : {:.2f}".format(*get_period_pr_v3(data_period, pr_cumprod[-1], pr_type="PROD", mode='CRYPTO'))    
    plt.title(title_msg, y=0.3, fontsize=10, loc='left')


def check_distribution_v2(spread_data, y_data, tr_data, v_concated, ts_data, bars_in_data, data_type='TRAIN'):
    
    """
    v1 --> v2
        1. add v_concated as input param.
    """
    
    plt.figure(figsize=(10,10))
    # plt.figure(figsize=(9,9))
    rows, cols = 4, 3
    
    
    spread_target = spread_data[v_concated]
    y_target = y_data[v_concated]
    tr_target = tr_data[v_concated]
    
    v_concated = np.array(v_concated)
    
    
    data_period = (ts_data[-1] - ts_data[0]) / 60  # minute
    # print("data_period :", data_period)


    # spread_range = spread_target - 1
    spread_range = spread_target
    
    # lvrg_needed_ = (1 / spread_range_scaled * lvrg_k)
    lvrg_needed_ = (1 / spread_range * lvrg_k)
    if lvrg_ceiling:
        lvrg_needed_[lvrg_needed_ > lvrg_max] = lvrg_max 
    lvrg_needed_int_ = lvrg_needed_.astype('uint32')
    lvrg_needed_int_[lvrg_needed_int_ < 1] = 1
       
        

    
    v3_01 = y_target.ravel() > 1
    v3_02 = y_target.ravel() < 1
    
    v3 = (spread_min <= spread_range) & (spread_range <= spread_max)    
    
    y_target_spread = y_target[v3].ravel()
    v3_1 = y_target_spread > 1
    v3_2 = y_target_spread < 1


    plt.subplot(rows, cols, 1)
    plt.scatter(np.arange(len(spread_range[v3_01])), spread_range[v3_01], color='g', s=5)
    plt.scatter(np.arange(len(spread_range[v3_02])), spread_range[v3_02], color='r', s=5)
    plt.axhline(spread_max, linestyle='--')
    plt.axhline(spread_min, linestyle='--')
    plt.title('spread', loc='right', fontsize=10)

    plt.subplot(rows, cols, 2)
    # plt.scatter(np.arange(len(spread_range_scaled)), spread_range_scaled, s=5)
    """
    # x_range 는 짧아지는게 정상 : v3_1 / v3_2 를 분리해서 표기했기 때문에.
    """
    plt.scatter(np.arange(len(spread_range[v3][v3_1])), spread_range[v3][v3_1], color='g', s=5)
    plt.scatter(np.arange(len(spread_range[v3][v3_2])), spread_range[v3][v3_2], color='r', s=5)
    # plt.show()

    plt.subplot(rows, cols, 3)
    
    v_concated_final = v_concated[v3]
    ts_valid_idx_sorted = adjust_time_span_v3(v_concated_final, ts_data, bars_in_data, workers=workers)    
    v_concated_ts = v_concated_final[ts_valid_idx_sorted]
    
    y_target_spread_ts = y_data[v_concated_ts].ravel()    
    plot_pr(y_target_spread_ts, data_period)    
    # plt.show()

    
        
        
    lvrg_needed = lvrg_needed_[v3]
    
    # v4 = lvrg_needed < lvrg_max    
#     v4_01 = y_target[v3].ravel() > 1
#     v4_02 = y_target[v3].ravel() < 1    
        
    # 1. threshing 우선. --> loc_set.
    # v4 = (lvrg_min < lvrg_needed)   
    v4 = (lvrg_min <= lvrg_needed) & (lvrg_needed <= lvrg_max2)
    
    # 2. int화, min_value validation 나중. --> lvrg_liqd()
    #    a. 해당 phase 를 vectorize 하기 위해 우선하게 되면, threshing 기능이 약화됨 (섬세하게 접근할 수 없게된다.)    
    #        i. utils --> public 의 순서를 가지고 있기 때문.
    lvrg_needed_int = lvrg_needed[v4].astype('uint32')
    # lvrg_needed_int = lvrg_needed[v4].astype('uint8')
    lvrg_needed_int[lvrg_needed_int < 1] = 1

    y_target_lvrg = y_target[v3][v4].ravel()
    v4_1 = y_target_lvrg > 1
    v4_2 = y_target_lvrg < 1
    
    

    plt.subplot(rows, cols, 4)

    # plt.scatter(np.arange(len(spread_range_scaled)), lvrg_needed, s=5)
    plt.scatter(np.arange(len(lvrg_needed[v3_1])), lvrg_needed[v3_1], color='g', s=5)
    plt.scatter(np.arange(len(lvrg_needed[v3_2])), lvrg_needed[v3_2], color='r', s=5)
    # plt.bar(np.arange(len(spread_range_scaled)), lvrg_needed)
    plt.axhline(lvrg_max, alpha=.5, linestyle='--')
    plt.axhline(lvrg_max2, alpha=1, linestyle='--')
    plt.axhline(lvrg_min, linestyle='--')
    plt.title('lvrg', loc='right', fontsize=10)
    # plt.show()

    plt.subplot(rows, cols, 5)
    # plt.bar(np.arange(len(lvrg_needed_int)), lvrg_needed_int)
    plt.scatter(np.arange(len(lvrg_needed_int[v4_1])), lvrg_needed_int[v4_1], color='g', s=5)
    plt.scatter(np.arange(len(lvrg_needed_int[v4_2])), lvrg_needed_int[v4_2], color='r', s=5)
    plt.title('lvrg_int', loc='right', fontsize=10)
    # plt.show()

    plt.subplot(rows, cols, 6)
    
    v_concated_final = v_concated[v3][v4]
    ts_valid_idx_sorted = adjust_time_span_v3(v_concated_final, ts_data, bars_in_data, workers=workers)    
    v_concated_ts = v_concated_final[ts_valid_idx_sorted]
    
    y_target_lvrg_ts = y_data[v_concated_ts].ravel()
    plot_pr(y_target_lvrg_ts, data_period)
    # plt.show()


    


    tr_range = tr_target[v3][v4]

    v5 = (tr_min <= tr_range) & (tr_range <= tr_max)
    
    y_target_tr = y_target[v3][v4][v5].ravel()
    v5_1 = y_target_tr > 1
    v5_2 = y_target_tr < 1
    
    

    # plt.subplot(1,1,3)
    plt.subplot(rows, cols, 7)
    # plt.scatter(np.arange(len(tr_range)), tr_range, s=5)
    plt.scatter(np.arange(len(tr_range[v4_1])), tr_range[v4_1], color='g', s=5)
    plt.scatter(np.arange(len(tr_range[v4_2])), tr_range[v4_2], color='r', s=5)
    plt.axhline(tr_max, linestyle='--')
    plt.axhline(tr_min, linestyle='--')
    # # plt.show()
    plt.title('tr', loc='right', fontsize=10)

    plt.subplot(rows, cols, 8)
    # plt.scatter(np.arange(len(tr_range[v5])), tr_range[v5])
    plt.scatter(np.arange(len(tr_range[v5][v5_1])), tr_range[v5][v5_1], color='g', s=5)
    plt.scatter(np.arange(len(tr_range[v5][v5_2])), tr_range[v5][v5_2], color='r', s=5)

    # v4 = lvrg_needed < lvrg_max
    plt.subplot(rows, cols, 9)
    
    v_concated_final = v_concated[v3][v4][v5]
    ts_valid_idx_sorted = adjust_time_span_v3(v_concated_final, ts_data, bars_in_data, workers=workers)    
    v_concated_ts = v_concated_final[ts_valid_idx_sorted]
    
    y_target_tr_ts = y_data[v_concated_ts].ravel()
    plot_pr(y_target_tr_ts, data_period)
    
    
        
    plt.subplot(rows, cols, 12) 
    
    if lvrg_type == 'STATIC':
        y_target_final = (y_target_tr_ts - 1) * lvrg_k + 1
    else:
        lvrg_needed_int_ts = lvrg_needed_int_[v3][v4][v5][ts_valid_idx_sorted]
        y_target_final = (y_target_tr_ts - 1) * lvrg_needed_int_ts + 1
    # print("np.sum(y_target_final < 0) :", np.sum(y_target_final < 0))
    
    plot_pr(y_target_final, data_period)

    
    
    plt.suptitle("{}".format(data_type))
    # save_digits = ''.join(['_' + str(val).split('.')[-1] for val in [spread_max, spread_min, lvrg_max, lvrg_min, lvrg_k, tr_max, tr_min, workers, datetime.now().timestamp()]])
    # save_digits = ''.join(['_' + str(val).split('.')[-1] for val in [spread_max, spread_min, lvrg_max, lvrg_min, lvrg_k, tr_max, tr_min, workers]])
    save_digits = ''.join(['_' + str(val) for val in [spread_max, spread_min, lvrg_max, lvrg_min, lvrg_k, tr_max, tr_min, workers]])
    
    if use_robust_key:
        plt.savefig(os.path.join(res_save_path, "robust_res/{}_{}.png".format(save_digits, data_type)))
    else:
        plt.savefig(os.path.join(res_save_path, "res/{}_{}.png".format(save_digits, data_type)))
    plt.show()
    
    
def check_distribution(spread_target, y_target, tr_target, ts_target, data_type='TRAIN'):
    
    plt.figure(figsize=(10,10))
    # plt.figure(figsize=(9,9))
    rows, cols = 4, 3
    
    
    data_period = (ts_target[-1] - ts_target[0]) / 60  # minute
    # print("data_period :", data_period)


    spread_range = spread_target - 1

    v3_01 = y_target.ravel() > 1
    v3_02 = y_target.ravel() < 1
    
    # v3 = (spread_min < spread_range) & (spread_range < spread_max)
    v3 = (spread_min <= spread_range) & (spread_range <= spread_max)
    # spread_range_scaled = min_max_scaler(spread_range[v3])
    # spread_range_scaled = (spread_range[v3] - spread_min) / (spread_max - spread_min)
    # spread_range_scaled = spread_range[v3]  # / 0.5
    
    y_target_spread = y_target[v3].ravel()
    v3_1 = y_target_spread > 1
    v3_2 = y_target_spread < 1


    plt.subplot(rows, cols, 1)
    plt.scatter(np.arange(len(spread_range[v3_01])), spread_range[v3_01], color='g', s=5)
    plt.scatter(np.arange(len(spread_range[v3_02])), spread_range[v3_02], color='r', s=5)
    plt.axhline(spread_max, linestyle='--')
    plt.axhline(spread_min, linestyle='--')
    plt.title('spread', loc='right', fontsize=10)

    plt.subplot(rows, cols, 2)
    # plt.scatter(np.arange(len(spread_range_scaled)), spread_range_scaled, s=5)
    """
    # x_range 는 짧아지는게 정상 : v3_1 / v3_2 를 분리해서 표기했기 때문에.
    """
    plt.scatter(np.arange(len(spread_range[v3][v3_1])), spread_range[v3][v3_1], color='g', s=5)
    plt.scatter(np.arange(len(spread_range[v3][v3_2])), spread_range[v3][v3_2], color='r', s=5)
    # plt.show()

    plt.subplot(rows, cols, 3)
    plot_pr(y_target_spread, data_period)
    
    # plt.show()


    # lvrg_needed = (1 / spread_range_scaled * lvrg_k)
    lvrg_needed = (1 / spread_range[v3] * lvrg_k)
    if lvrg_ceiling:
        lvrg_needed[lvrg_needed > lvrg_max] = lvrg_max 
    
    # v4 = lvrg_needed < lvrg_max
    
#     v4_01 = y_target[v3].ravel() > 1
#     v4_02 = y_target[v3].ravel() < 1
    
        
    # 1. threshing 우선. --> loc_set.
    # v4 = (lvrg_min < lvrg_needed)   
    v4 = (lvrg_min <= lvrg_needed) & (lvrg_needed <= lvrg_max2)
    
    # 2. int화, min_value validation 나중. --> lvrg_liqd()
    #    a. 해당 phase 를 vectorize 하기 위해 우선하게 되면, threshing 기능이 약화됨 (섬세하게 접근할 수 없게된다.)    
    #        i. utils --> public 의 순서를 가지고 있기 때문.
    lvrg_needed_int = lvrg_needed[v4].astype('uint32')
    # lvrg_needed_int = lvrg_needed[v4].astype('uint8')
    lvrg_needed_int[lvrg_needed_int < 1] = 1

    y_target_lvrg = y_target[v3][v4].ravel()
    v4_1 = y_target_lvrg > 1
    v4_2 = y_target_lvrg < 1

    plt.subplot(rows, cols, 4)

    # plt.scatter(np.arange(len(spread_range_scaled)), lvrg_needed, s=5)
    plt.scatter(np.arange(len(lvrg_needed[v3_1])), lvrg_needed[v3_1], color='g', s=5)
    plt.scatter(np.arange(len(lvrg_needed[v3_2])), lvrg_needed[v3_2], color='r', s=5)
    # plt.bar(np.arange(len(spread_range_scaled)), lvrg_needed)
    plt.axhline(lvrg_max, alpha=.5, linestyle='--')
    plt.axhline(lvrg_max2, alpha=1, linestyle='--')
    plt.axhline(lvrg_min, linestyle='--')
    plt.title('lvrg', loc='right', fontsize=10)
    # plt.show()

    plt.subplot(rows, cols, 5)
    # plt.bar(np.arange(len(lvrg_needed_int)), lvrg_needed_int)
    plt.scatter(np.arange(len(lvrg_needed_int[v4_1])), lvrg_needed_int[v4_1], color='g', s=5)
    plt.scatter(np.arange(len(lvrg_needed_int[v4_2])), lvrg_needed_int[v4_2], color='r', s=5)
    plt.title('lvrg_int', loc='right', fontsize=10)
    # plt.show()

    plt.subplot(rows, cols, 6)
    plot_pr(y_target_lvrg, data_period)
    

    # plt.show()




    tr_range = tr_target[v3][v4]

    # v5_01 = y_target[v3][v4].ravel() > 1
    # v5_02 = y_target[v3][v4].ravel() < 1

    v5 = (tr_min <= tr_range) & (tr_range <= tr_max)
    y_target_tr = y_target[v3][v4][v5].ravel()
    v5_1 = y_target_tr > 1
    v5_2 = y_target_tr < 1

    # plt.subplot(1,1,3)
    plt.subplot(rows, cols, 7)
    # plt.scatter(np.arange(len(tr_range)), tr_range, s=5)
    plt.scatter(np.arange(len(tr_range[v4_1])), tr_range[v4_1], color='g', s=5)
    plt.scatter(np.arange(len(tr_range[v4_2])), tr_range[v4_2], color='r', s=5)
    plt.axhline(tr_max, linestyle='--')
    plt.axhline(tr_min, linestyle='--')
    # # plt.show()
    plt.title('tr', loc='right', fontsize=10)

    plt.subplot(rows, cols, 8)
    # plt.scatter(np.arange(len(tr_range[v5])), tr_range[v5])
    plt.scatter(np.arange(len(tr_range[v5][v5_1])), tr_range[v5][v5_1], color='g', s=5)
    plt.scatter(np.arange(len(tr_range[v5][v5_2])), tr_range[v5][v5_2], color='r', s=5)

    # v4 = lvrg_needed < lvrg_max
    plt.subplot(rows, cols, 9)
    plot_pr(y_target_tr, data_period)

    # plt.show()

    # plt.subplot(rows, 1, 1)
    # plt.subplot(rows, 1, 1)
    plt.subplot(rows, cols, 12)
    # print("np.sum(lvrg_needed_int< 1) :", np.sum(lvrg_needed_int< 1))
    if lvrg_type == 'STATIC':
        y_target_final = (y_target[v3][v4][v5].ravel() - 1) * lvrg_k + 1
    else:
        y_target_final = (y_target[v3][v4][v5].ravel() - 1) * lvrg_needed_int[v5] + 1
    # print("np.sum(y_target_final < 0) :", np.sum(y_target_final < 0))
    
    plot_pr(y_target_final, data_period)
    
    plt.suptitle("{}".format(data_type))
    save_digits = ''.join(['_' + str(val).split('.')[-1] for val in [spread_max, spread_min, lvrg_max, lvrg_min, lvrg_k, tr_max, tr_min, workers, time_span_size]])
    # plt.savefig(r"D:\Projects\System_Trading\JnQ\result/{}_res{}_{}.png".format(data_type, save_digits, str(datetime.now().timestamp()).split('.')[0]))
    plt.savefig(r"D:\Projects\System_Trading\JnQ\result/{}_res{}.png".format(data_type, save_digits))
    plt.show()
    

def adjust_time_span_v2_invalid(v, ts_train, workers=5, time_span_size=60 * 100):
    """
    algorithm validation.

    # print(sorted(v_ts_random)[:10])
    # print(sorted(v)[:10])

    # for vtr in v_ts_random:
    #     if vtr not in v:
    #         print("{} not in v".format(vtr))
    """

    ts_train_close_ = ts_train[v]
    sorting_index = np.argsort(ts_train_close_)  # reindexed v_index
    # print(sorting_index.max(), sorting_index.min())
    # print("sorting_index :", sorting_index)

    v_sorted_by_ts = np.array(v)[sorting_index]  # reindexed v_index
    # print("v_sorted_by_ts : {}".format(v_sorted_by_ts))

    # ts_train_close = sorted(ts_train_close_)  # 60s * 100 bars = 6000
    ts_train_close = ts_train_close_[sorting_index]
    ts_train_close -= min(ts_train_close)
    print("len(v_sorted_by_ts) :", len(v_sorted_by_ts))
    print("len(ts_train_close) :", len(ts_train_close))

    v_sorted_by_ts = v
    ts_train_close = ts_train[v]
    ts_train_close -= min(ts_train_close)

    plt.plot(ts_train_close)
    plt.show()
    print("len(v_sorted_by_ts) :", len(v_sorted_by_ts))
    print("len(ts_train_close) :", len(ts_train_close))

    # break

    n = 0
    # time_span_size = 60 * 50
    # time_span_size = 5
    v_ts_random = []
    # ts_train_close_len = len(ts_train_close)

    while 1:
        # for ts_ in ts_train_close:
        n += 1
        time_span = time_span_size * n
        if time_span >= ts_train_close[-1]:
            break

        # 1. 한 time_span 내의 indexes 중 택 1 하는 알고리즘을 구성할 것.
        inspan_index = np.argwhere(ts_train_close < time_span).ravel()
        # print("inspan_index :", inspan_index)

        #    a. if time_span 내의 indexes 존재시,
        if len(inspan_index) > 0:
            inspan_index_last = inspan_index[-1]
            # print("inspan_index_last :", inspan_index_last)
            # print("ts_train_close[:inspan_index_last + 1] :", ts_train_close[:inspan_index_last + 1])

            # ts_train_session = ts_train_close[:inspan_index_lastes + 1]
            # sorting_index_inspan = sorting_index[:inspan_index_last + 1]
            v_sorted_by_ts_inspan = v_sorted_by_ts[:inspan_index_last + 1]
            # print("v_sorted_by_ts_inspan :", v_sorted_by_ts_inspan)

            # if inspan_index_lastes >= ts_train_close_len - 1:
            #     break

            #    i. sorting_index_inspan 중 택 1. (randomly)
            # if len(sorting_index_inspan) > 0:
            # choiced_index = np.random.choice(sorting_index_inspan)
            # choiced_index = np.random.choice(v_sorted_by_ts_inspan)
            # v_ts_random.append(np.random.choice(v_sorted_by_ts_inspan))
            # v_ts_random.append(v_sorted_by_ts_inspan[0])
            # v_ts_random += list(v_sorted_by_ts_inspan
            random.shuffle(v_sorted_by_ts_inspan)
            v_ts_random += list(v_sorted_by_ts_inspan[:workers])
            # v_ts_random += list(v_sorted_by_ts_inspan[0])

            #    ii. overlapping variable with remaining items.
            ts_train_close = ts_train_close[inspan_index_last + 1:]
            v_sorted_by_ts = v_sorted_by_ts[inspan_index_last + 1:]

            # print("inspan_index : {}".format(inspan_index))
            # print("choiced_index : {}".format(choiced_index))
            # # print(ts_train_close)
            # print()
            # break

    return v_ts_random


def adjust_time_span_v1(v, ts_train, workers=5, time_span_size=60 * 100):
    """
    algorithm validation.

    # print(sorted(v_ts_random)[:10])
    # print(sorted(v)[:10])

    # for vtr in v_ts_random:
    #     if vtr not in v:
    #         print("{} not in v".format(vtr))

    v1. v_concated 는 ts 가 다시 섞이기 때문에, 재정렬이 요구된다.
    """

    ts_train_close_ = ts_train[v]
    sorting_index = np.argsort(ts_train_close_)  # reindexed v_index
    # print(sorting_index.max(), sorting_index.min())
    # print("sorting_index :", sorting_index)

    v_sorted_by_ts = np.array(v)[sorting_index]  # reindexed v_index
    # print("v_sorted_by_ts : {}".format(v_sorted_by_ts))

    ts_train_close = ts_train_close_[sorting_index]
    ts_train_close -= min(ts_train_close)

    # plt.plot(ts_train_close)
    # plt.show()

    # break

    n = 0
    v_ts_random = []
    # ts_train_close_len = len(ts_train_close)

    while 1:
        # for ts_ in ts_train_close:
        n += 1
        time_span = time_span_size * n
        if time_span >= ts_train_close[-1]:
            break

        # 1. 한 time_span 내의 indexes 중 택 1 하는 알고리즘을 구성할 것.
        inspan_index = np.argwhere(ts_train_close < time_span).ravel()
        # print("inspan_index :", inspan_index)

        #    a. if time_span 내의 indexes 존재시,
        if len(inspan_index) > 0:
            inspan_index_last = inspan_index[-1]
            # print("inspan_index_last :", inspan_index_last)
            # print("ts_train_close[:inspan_index_last + 1] :", ts_train_close[:inspan_index_last + 1])

            # ts_train_session = ts_train_close[:inspan_index_lastes + 1]
            # sorting_index_inspan = sorting_index[:inspan_index_last + 1]
            v_sorted_by_ts_inspan = v_sorted_by_ts[:inspan_index_last + 1]
            # print("v_sorted_by_ts_inspan :", v_sorted_by_ts_inspan)

            # if inspan_index_lastes >= ts_train_close_len - 1:
            #     break

            #    i. sorting_index_inspan 중 택 1. (randomly)
            # if len(sorting_index_inspan) > 0:
            # choiced_index = np.random.choice(sorting_index_inspan)
            # choiced_index = np.random.choice(v_sorted_by_ts_inspan)
            # v_ts_random.append(np.random.choice(v_sorted_by_ts_inspan))
            # v_ts_random.append(v_sorted_by_ts_inspan[0])
            # v_ts_random += list(v_sorted_by_ts_inspan
            random.shuffle(v_sorted_by_ts_inspan)
            v_ts_random += list(v_sorted_by_ts_inspan[:workers])
            # v_ts_random += list(v_sorted_by_ts_inspan[0])

            #    ii. overlapping variable with remaining items.
            ts_train_close = ts_train_close[inspan_index_last + 1:]
            v_sorted_by_ts = v_sorted_by_ts[inspan_index_last + 1:]

            # print("inspan_index : {}".format(inspan_index))
            # print("choiced_index : {}".format(choiced_index))
            # # print(ts_train_close)
            # print()
            # break

    return v_ts_random
