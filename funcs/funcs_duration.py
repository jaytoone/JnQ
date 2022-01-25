from funcs.funcs_indicator_candlescore import *
from funcs.funcs_trader import *
import logging

sys_log3 = logging.getLogger()


class OrderSide:
    BUY = "BUY"
    SELL = "SELL"
    INVALID = None


def mr_calc(mr_const_cnt, mr_score, const_str, const_bool, var_left, var_right, show_detail):
    # splited_const_str = const_str.split(' ')
    # assert len(splited_const_str) == 3, "assert len(splited_const_str) == 3"

    # const_bool = eval(const_str)
    if show_detail:
        sys_log3.warning("{} : {:.5f} {:.5f} ({})".format(const_str, var_left, var_right, const_bool))

    mr_const_cnt += 1
    if const_bool:
        mr_score += 1

    # else:
    #     if show_detail:
    #         pass
    #     else:
    #         return res_df, open_side, zone

    return mr_const_cnt, mr_score


def candle_game0(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    strat_version = config.strat_version
    if open_side == OrderSide.SELL:
        if strat_version in ['v5_2', '1_1']:  # Todo - 언젠가 분리 시켜주어야할 것
            front_wick_score = res_df['front_wick_score'].iloc[i]
            const_ = "front_wick_score <= -config.loc_set.zone.front_wick_score0"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, front_wick_score,
                                             -config.loc_set.zone.front_wick_score0, show_detail)
    else:
        if strat_version in ['v5_2', '1_1']:
            front_wick_score = res_df['front_wick_score'].iloc[i]
            const_ = "front_wick_score >= config.loc_set.zone.front_wick_score0"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, front_wick_score, config.loc_set.zone.front_wick_score0,
                                             show_detail)

    return mr_const_cnt, mr_score


#   Todo - 현재, func_naming 은 short 기준으로 진행중
def candle_game(open_side, res_df, config, i, score_idx, prev_hclose_idx, mr_const_cnt, mr_score, show_detail):
    strat_version = config.strat_version
    if i < 0 or (i >= 0 and prev_hclose_idx >= 0):  # trader or colab env.
        h_front_wick_score = res_df['h_front_wick_score'].iloc[prev_hclose_idx]
        h_body_score = res_df['h_body_score'].iloc[prev_hclose_idx]
        h_back_wick_score = res_df['h_back_wick_score'].iloc[prev_hclose_idx]

        if open_side == OrderSide.SELL:
            if config.loc_set.zone.front_wick_score[score_idx] != "None":
                if strat_version in ['v7_3']:
                    total_ratio = h_front_wick_score + h_body_score / 100
                    if config.loc_set.zone.front_sign[score_idx] == "up":
                        const_ = "total_ratio <= -config.loc_set.zone.front_wick_score[score_idx]"
                    else:
                        const_ = "total_ratio >= -config.loc_set.zone.front_wick_score[score_idx]"
                    const_bool_ = eval(const_)
                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, total_ratio,
                                                     -config.loc_set.zone.front_wick_score[score_idx], show_detail)

                else:
                    if config.loc_set.zone.front_sign[score_idx] == "up":
                        const_ = "h_front_wick_score <= -config.loc_set.zone.front_wick_score[score_idx]"
                    else:
                        const_ = "h_front_wick_score >= -config.loc_set.zone.front_wick_score[score_idx]"
                    const_bool_ = eval(const_)
                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_front_wick_score,
                                                     -config.loc_set.zone.front_wick_score[score_idx], show_detail)

            #       h_body_score 가 public 으로 정의하는게 좋을까     #
            if config.loc_set.zone.body_score[score_idx] != "None":
                if config.loc_set.zone.body_sign[score_idx] == "up":
                    const_ = "h_body_score >= config.loc_set.zone.body_score[score_idx]"
                else:
                    const_ = "h_body_score <= config.loc_set.zone.body_score[score_idx]"
                const_bool_ = eval(const_)
                mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_body_score,
                                                 config.loc_set.zone.body_score[score_idx],
                                                 show_detail)

            if config.loc_set.zone.back_wick_score[score_idx] != "None":
                if config.loc_set.zone.back_sign[score_idx] == "up":
                    const_ = "h_back_wick_score >= config.loc_set.zone.back_wick_score[score_idx]"
                else:
                    const_ = "h_back_wick_score <= config.loc_set.zone.back_wick_score[score_idx]"
                const_bool_ = eval(const_)
                mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_back_wick_score,
                                                 config.loc_set.zone.back_wick_score[score_idx],
                                                 show_detail)
        else:
            if config.loc_set.zone.front_wick_score[score_idx] != "None":
                if strat_version in ['v7_3']:
                    total_ratio = h_front_wick_score + h_body_score / 100
                    if config.loc_set.zone.front_sign[score_idx] == "up":
                        const_ = "total_ratio >= config.loc_set.zone.front_wick_score[score_idx]"
                    else:
                        const_ = "total_ratio <= config.loc_set.zone.front_wick_score[score_idx]"
                    const_bool_ = eval(const_)
                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, total_ratio,
                                                     config.loc_set.zone.front_wick_score[score_idx], show_detail)

                else:
                    if config.loc_set.zone.front_sign[score_idx] == "up":
                        const_ = "h_front_wick_score >= config.loc_set.zone.front_wick_score[score_idx]"
                    else:
                        const_ = "h_front_wick_score <= config.loc_set.zone.front_wick_score[score_idx]"
                    const_bool_ = eval(const_)
                    mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_front_wick_score,
                                                     config.loc_set.zone.front_wick_score[score_idx], show_detail)

            if config.loc_set.zone.body_score[score_idx] != "None":
                if config.loc_set.zone.body_sign[score_idx] == "up":
                    const_ = "h_body_score >= config.loc_set.zone.body_score[score_idx]"
                else:
                    const_ = "h_body_score <= config.loc_set.zone.body_score[score_idx]"
                const_bool_ = eval(const_)
                mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_body_score,
                                                 config.loc_set.zone.body_score[score_idx],
                                                 show_detail)

            if config.loc_set.zone.back_wick_score[score_idx] != "None":
                if config.loc_set.zone.back_sign[score_idx] == "up":
                    const_ = "h_back_wick_score <= -config.loc_set.zone.back_wick_score[score_idx]"
                else:
                    const_ = "h_back_wick_score >= -config.loc_set.zone.back_wick_score[score_idx]"
                const_bool_ = eval(const_)
                mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_back_wick_score,
                                                 -config.loc_set.zone.back_wick_score[score_idx],
                                                 show_detail)

    return mr_const_cnt, mr_score


def candle_game2(order_side, res_df, config, i, shared_ticks, mr_const_cnt, mr_score, show_detail):
    ho = res_df['hopen_H'].iloc[i]
    hh = res_df['high'].iloc[i + 1 - shared_ticks: i + 1].max()
    hl = res_df['low'].iloc[i + 1 - shared_ticks: i + 1].min()
    hc = res_df['close'].iloc[i]
    h_front_wick_score, h_body_score, h_back_wick_score = get_candle_score(ho, hh, hl, hc)
    if order_side == OrderSide.SELL:
        # ------ public const_ ------ #
        # const_ = "ho > hc"
        # const_bool_ = eval(const_)
        # if show_detail:
        #     sys_log3.warning("ho > hc : {:.5f} {:.5f} ({})".format(ho, hc, const_))
        # mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, res_df, open_side, show_detail)

        if config.loc_set.zone.front_wick_score2 != "None":
            const_ = "h_front_wick_score <= -config.loc_set.zone.front_wick_score2"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_front_wick_score, -config.loc_set.zone.front_wick_score2,
                                             show_detail)

        if config.loc_set.zone.body_score2 != "None":
            const_ = "h_body_score >= config.loc_set.zone.body_score2"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_body_score, config.loc_set.zone.body_score2, show_detail)

        if config.loc_set.zone.back_wick_score2 != "None":
            const_ = "h_back_wick_score >= config.loc_set.zone.back_wick_score2"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_back_wick_score, config.loc_set.zone.back_wick_score2,
                                             show_detail)
    else:
        if config.loc_set.zone.front_wick_score2 != "None":
            const_ = "h_front_wick_score >= config.loc_set.zone.front_wick_score2"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_front_wick_score, config.loc_set.zone.front_wick_score2,
                                             show_detail)

        if config.loc_set.zone.body_score2 != "None":
            const_ = "h_body_score >= config.loc_set.zone.body_score2"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_body_score, config.loc_set.zone.body_score2, show_detail)

        if config.loc_set.zone.back_wick_score2 != "None":
            const_ = "h_back_wick_score <= -config.loc_set.zone.back_wick_score2"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, h_back_wick_score, -config.loc_set.zone.back_wick_score2,
                                             show_detail)

    return mr_const_cnt, mr_score


def spread(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    tp_fee, out_fee = calc_tp_out_fee(config)

    if open_side == OrderSide.SELL:
        spread = (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] - tp_fee * res_df['bb_base_5m'].iloc[
            # spread = (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] - out_fee * res_df['bb_base_5m'].iloc[
            # i]) / (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + tp_fee *
            i]) / (res_df['bb_base_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + out_fee *
                   res_df['bb_base_5m'].iloc[i])
        # spread = (res_df['bb_base_15m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] - tp_fee * res_df['bb_base_15m'].iloc[
        #     i]) / (res_df['bb_base_15m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + out_fee *
        #             res_df['bb_base_15m'].iloc[i])

        # spread = (res_df['dc_upper_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] - tp_fee * res_df['bb_lower_5m'].iloc[
        #     i]) / (res_df['dc_upper_5m'].iloc[i] - res_df['bb_lower_5m'].iloc[i] + out_fee *
        #             res_df['bb_lower_5m'].iloc[i])

        # spread = (res_df['dc_upper_15m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] - tp_fee * res_df['dc_lower_5m'].iloc[
        #     i]) / (res_df['dc_upper_15m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] + out_fee *
        #             res_df['dc_lower_5m'].iloc[i])
        # spread = ((res_df['dc_upper_15m'].iloc[i] - res_df['dc_lower_5m'].iloc[i])/2 - tp_fee * res_df['dc_lower_5m'].iloc[
        #     i]) / ((res_df['dc_upper_15m'].iloc[i] - res_df['dc_lower_5m'].iloc[i])/2 + out_fee *
        #             res_df['dc_lower_5m'].iloc[i])

        const_ = "spread >= config.loc_set.zone.short_spread"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, spread, config.loc_set.zone.short_spread, show_detail)
    else:
        spread = (res_df['bb_upper_5m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] - tp_fee * res_df['bb_upper_5m'].iloc[
            # spread = (res_df['bb_upper_5m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] - out_fee * res_df['bb_upper_5m'].iloc[
            # i]) / (res_df['bb_upper_5m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] + tp_fee *
            i]) / (res_df['bb_upper_5m'].iloc[i] - res_df['dc_lower_5m'].iloc[i] + out_fee *
                   res_df['bb_upper_5m'].iloc[i])

        # spread = (res_df['dc_upper_5m'].iloc[i] - res_df['dc_lower_15m'].iloc[i] - tp_fee * res_df['dc_upper_5m'].iloc[
        #     i]) / (res_df['dc_upper_5m'].iloc[i] - res_df['dc_lower_15m'].iloc[i] + out_fee *
        #             res_df['dc_upper_5m'].iloc[i])
        # spread = ((res_df['dc_upper_5m'].iloc[i] - res_df['dc_lower_15m'].iloc[i])/2 - tp_fee * res_df['dc_upper_5m'].iloc[
        #     i]) / ((res_df['dc_upper_5m'].iloc[i] - res_df['dc_lower_15m'].iloc[i])/2 + out_fee *
        #             res_df['dc_upper_5m'].iloc[i])

        const_ = "spread >= config.loc_set.zone.long_spread"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, spread, config.loc_set.zone.long_spread, show_detail)

    return mr_const_cnt, mr_score


def dtk(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    strat_version = config.strat_version
    if open_side == OrderSide.SELL:
        # if res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i] >= res_df['short_rtc_1'].iloc[i] - res_df['h_short_rtc_gap'].iloc[i] * config.loc_set.zone.dt_k:
        #     dtk_v1 & v2 platform     #
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_lower_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                   res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = "dc >= dt_k"  # short 이라, tp_line 이 dc 보다 작아야하는게 맞음 (dc 가 이미 tp_line 에 다녀오면 안되니까)
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, dc, dt_k, show_detail)

            #     dc_v2   #
        else:
            dc = res_df['dc_lower_v2'.format(strat_version)].iloc[i]
            dt_k = res_df['short_dtk_1_{}'.format(strat_version)].iloc[i] - \
                   res_df['short_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = "dc >= dt_k"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, dc, dt_k, show_detail)
    else:
        if config.loc_set.zone.dtk_dc_itv != "None":
            dc = res_df['dc_upper_%s' % config.loc_set.zone.dtk_dc_itv].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                   res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = "dc <= dt_k"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, dc, dt_k, show_detail)

        else:
            #     dc_v2     #
            dc = res_df['dc_upper_v2'.format(strat_version)].iloc[i]
            dt_k = res_df['long_dtk_1_{}'.format(strat_version)].iloc[i] + \
                   res_df['long_dtk_gap_{}'.format(strat_version)].iloc[i] * config.loc_set.zone.dt_k

            const_ = "dc <= dt_k"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, dc, dt_k, show_detail)

    return mr_const_cnt, mr_score


def bb_lower2_close(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    if open_side == OrderSide.SELL:
        close = res_df['close'].iloc[i]
        bb_upper2_ = res_df['bb_upper2_%s' % config.loc_set.zone.bbz_itv].iloc[i]

        const_ = "bb_upper2_ < close"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_upper2_, close, show_detail)
    else:
        close = res_df['close'].iloc[i]
        bb_lower2_ = res_df['bb_lower2_%s' % config.loc_set.zone.bbz_itv].iloc[i]

        const_ = "bb_lower2_ > close"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_lower2_, close, show_detail)

    return mr_const_cnt, mr_score


def bb_base_4h_close(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    close = res_df['close'].iloc[i]
    bb_base_4h = res_df['bb_base_4h'].iloc[i]
    if open_side == OrderSide.SELL:
        # const_ = "close > bb_base_4h"
        const_ = "close < bb_base_4h"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, close, bb_base_4h, show_detail)
    else:
        # const_ = "close < bb_base_4h"
        const_ = "close > bb_base_4h"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, close, bb_base_4h, show_detail)

    return mr_const_cnt, mr_score


def bb_upper_5m_bb_base_(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    if open_side == OrderSide.SELL:
        bb_upper_5m = res_df['bb_upper_5m'].iloc[i]
        bb_base_ = res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]

        const_ = "bb_upper_5m < bb_base_"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_upper_5m, bb_base_, show_detail)
    else:
        bb_lower_5m = res_df['bb_lower_5m'].iloc[i]
        bb_base_ = res_df['bb_base_%s' % config.loc_set.zone.bbz_itv].iloc[i]

        const_ = "bb_lower_5m > bb_base_"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_lower_5m, bb_base_, show_detail)

    return mr_const_cnt, mr_score


def bb_upper_5m_ep_(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    strat_version = config.strat_version
    if open_side == OrderSide.SELL:
        bb_upper_5m = res_df['bb_upper_5m'].iloc[i]
        short_ep_ = res_df['short_ep_{}'.format(strat_version)].iloc[i]

        const_ = "bb_upper_5m > short_ep_"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_upper_5m, short_ep_, show_detail)
    else:
        bb_lower_5m = res_df['bb_lower_5m'].iloc[i]
        long_ep_ = res_df['long_ep_{}'.format(strat_version)].iloc[i]

        const_ = "bb_lower_5m < long_ep_"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_lower_5m, long_ep_, show_detail)

    return mr_const_cnt, mr_score


def dc_upper_5m_bb_upper_15m(open_side, res_df, config, i, prev_hopen_idx, mr_const_cnt, mr_score, show_detail):
    if i < 0 or (i >= 0 and prev_hopen_idx >= 0):  # considering ide & trader platform
        # if res_df['dc_upper_5m'].iloc[prev_hopen_idx] < res_df['bb_upper_15m'].iloc[i]:
        if open_side == OrderSide.SELL:
            dc_upper_5m = res_df['dc_upper_5m'].iloc[prev_hopen_idx]
            bb_upper_15m = res_df['bb_upper_15m'].iloc[prev_hopen_idx]

            const_ = "bb_upper_15m > dc_upper_5m"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_upper_15m, dc_upper_5m, show_detail)
        else:
            dc_lower_5m = res_df['dc_lower_5m'].iloc[prev_hopen_idx]
            bb_lower_15m = res_df['bb_lower_15m'].iloc[prev_hopen_idx]

            const_ = "bb_lower_15m < dc_lower_5m"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_lower_15m, dc_lower_5m, show_detail)

    return mr_const_cnt, mr_score


def dc_upper_5m_ema_5m(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    if open_side == OrderSide.SELL:
        dc_upper_5m = res_df['dc_upper_5m'].iloc[i]
        ema_5m = res_df['ema_5m'].iloc[i]

        const_ = "dc_upper_5m < ema_5m"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, dc_upper_5m, ema_5m, show_detail)
    else:
        dc_lower_5m = res_df['dc_lower_5m'].iloc[i]
        ema_5m = res_df['ema_5m'].iloc[i]

        const_ = "dc_lower_5m > ema_5m"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, dc_lower_5m, ema_5m, show_detail)

    return mr_const_cnt, mr_score


def close_ema_5m(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    close = res_df['close'].iloc[i]
    ema_5m = res_df['ema_5m'].iloc[i]
    if open_side == OrderSide.SELL:
        const_ = "close < ema_5m"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, close, ema_5m, show_detail)
    else:
        const_ = "close > ema_5m"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, close, ema_5m, show_detail)

    return mr_const_cnt, mr_score


def bb_upper_30m_ema_5m(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    if open_side == OrderSide.SELL:
        bb_upper_30m = res_df['bb_upper_30m'].iloc[i]
        ema_5m = res_df['ema_5m'].iloc[i]

        const_ = "bb_upper_30m > ema_5m"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_upper_30m, ema_5m, show_detail)
    else:
        bb_lower_30m = res_df['bb_lower_30m'].iloc[i]
        ema_5m = res_df['ema_5m'].iloc[i]

        const_ = "bb_lower_30m < ema_5m"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, bb_lower_30m, ema_5m, show_detail)

    return mr_const_cnt, mr_score


def degree(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    norm_period = 120
    if i < 0 or (i > norm_period):  # considering idep & trader platform
        norm_res = res_df['target_diff'].iloc[i] / res_df['norm_range'].iloc[i]
        if open_side == OrderSide.SELL:
            const_ = "norm_res < -config.loc_set.zone.degree"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, norm_res, -config.loc_set.zone.degree, show_detail)
        else:
            const_ = "norm_res > config.loc_set.zone.degree"
            const_bool_ = eval(const_)
            mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, norm_res, config.loc_set.zone.degree, show_detail)

    return mr_const_cnt, mr_score


# def hopen_H_ema_5m(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail): # Todo - hopen 잘사용해야함 (i)
#     hopen_H = res_df['hopen_H'].iloc[i]
#     ema_5m = res_df['ema_5m'].iloc[i]
#     if open_side == OrderSide.SELL:
#         const_ = "hopen_H < ema_5m"
#         const_bool_ = eval(const_)
#         mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, hopen_H, ema_5m, show_detail)
#     else:
#         const_ = "hopen_H > ema_5m"
#         const_bool_ = eval(const_)
#         mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, hopen_H, ema_5m, show_detail)
#
#     return mr_const_cnt, mr_score


def descending_dc(open_side, res_df, config, i, mr_const_cnt, mr_score, show_detail):
    if open_side == OrderSide.SELL:
        dc_lower_5m = res_df['dc_lower_5m'].iloc[i]
        dc_lower_5m_min = res_df['dc_lower_5m'].iloc[i - 50:i].min()  # Todo - public 에서 rolling 해놓는게 latency 좋음

        const_ = "dc_lower_5m <= dc_lower_5m_min"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, dc_lower_5m, dc_lower_5m_min, show_detail)
    else:
        dc_upper_5m = res_df['dc_upper_5m'].iloc[i]
        dc_upper_5m_max = res_df['dc_upper_5m'].iloc[i - 50:i].max()  # Todo - public 에서 rolling 해놓는게 latency 좋음

        const_ = "dc_upper_5m >= dc_upper_5m_max"
        const_bool_ = eval(const_)
        mr_const_cnt, mr_score = mr_calc(mr_const_cnt, mr_score, const_, const_bool_, dc_upper_5m, dc_upper_5m_max, show_detail)

    return mr_const_cnt, mr_score


def zoned_tr_set(open_side, res_df, config, i, zone, show_detail):
    strat_version = config.strat_version
    if open_side == OrderSide.SELL:
        zone_dc_upper_v2_ = res_df['zone_dc_upper_v2'.format(strat_version)].iloc[i]
        long_dtk_plot_ = res_df['long_dtk_plot_1'].iloc[i] + res_df['long_dtk_plot_gap'].iloc[
            i] * config.loc_set.zone.zone_dt_k

        const_ = "zone_dc_upper_v2_ > long_dtk_plot_"
        const_bool_ = eval(const_)
        if show_detail:
            sys_log3.warning("{} : {:.5f} {:.5f} ({})".format(const_, zone_dc_upper_v2_, long_dtk_plot_, const_bool_))

        if const_bool_:  # c_zone
            if config.ep_set.static_ep:
                res_df['short_ep_{}'.format(strat_version)].iloc[i] = res_df['short_ep2_{}'.format(strat_version)].iloc[i]
            else:
                res_df['short_ep_{}'.format(strat_version)] = res_df['short_ep2_{}'.format(strat_version)]
            if config.out_set.static_out:
                res_df['short_out_{}'.format(strat_version)].iloc[i] = \
                    res_df['short_out_org_{}'.format(strat_version)].iloc[i]
            else:
                res_df['short_out_{}'.format(strat_version)] = res_df['short_out_org_{}'.format(strat_version)]
            zone = 'c'
        else:  # t_zone
            if config.ep_set.static_ep:
                res_df['short_ep_{}'.format(strat_version)].iloc[i] = \
                    res_df['short_ep_org_{}'.format(strat_version)].iloc[i]
            else:
                res_df['short_ep_{}'.format(strat_version)] = res_df['short_ep_org_{}'.format(strat_version)]
            if config.out_set.static_out:
                res_df['short_out_{}'.format(strat_version)].iloc[i] = \
                    res_df['short_out2_{}'.format(strat_version)].iloc[i]
            else:
                res_df['short_out_{}'.format(strat_version)] = res_df['short_out2_{}'.format(strat_version)]
            zone = 't'
    else:
        zone_dc_lower_v2_ = res_df['zone_dc_lower_v2'.format(strat_version)].iloc[i]
        short_dtk_plot_ = res_df['short_dtk_plot_1'].iloc[i] - res_df['short_dtk_plot_gap'].iloc[
            i] * config.loc_set.zone.zone_dt_k

        const_ = "zone_dc_lower_v2_ < short_dtk_plot_"
        const_bool_ = eval(const_)
        if show_detail:
            sys_log3.warning("{} : {:.5f} {:.5f} ({})".format(const_, zone_dc_lower_v2_, short_dtk_plot_, const_bool_))

        if const_bool_:
            if config.ep_set.static_ep:
                res_df['long_ep_{}'.format(strat_version)].iloc[i] = res_df['long_ep2_{}'.format(strat_version)].iloc[i]
            else:
                res_df['long_ep_{}'.format(strat_version)] = res_df['long_ep2_{}'.format(strat_version)]
            if config.out_set.static_out:
                res_df['long_out_{}'.format(strat_version)].iloc[i] = \
                    res_df['long_out_org_{}'.format(strat_version)].iloc[i]
            else:
                res_df['long_out_{}'.format(strat_version)] = res_df['long_out_org_{}'.format(strat_version)]
            zone = 'c'
        else:
            if config.ep_set.static_ep:
                res_df['long_ep_{}'.format(strat_version)].iloc[i] = \
                    res_df['long_ep_org_{}'.format(strat_version)].iloc[i]
            else:
                res_df['long_ep_{}'.format(strat_version)] = res_df['long_ep_org_{}'.format(strat_version)]
            if config.out_set.static_out:
                res_df['long_out_{}'.format(strat_version)].iloc[i] = res_df['long_out2_{}'.format(strat_version)].iloc[
                    i]
            else:
                res_df['long_out_{}'.format(strat_version)] = res_df['long_out2_{}'.format(strat_version)]
            zone = 't'
    return res_df, zone


def legacy():  # 위에가 short
    # ------------------ tr scheduling ------------------ #
    # if config.loc_set.zone.tr_thresh != "None":
    #   tr = ((done_tp - ep_list[0] - tp_fee * ep_list[0]) / (ep_list[0] - done_out + out_fee * ep_list[0]))

    # -------------- candle_dt_k -------------- #
    # mr_const_cnt += 1
    # # if res_df['dc_lower_1m'].iloc[i] >= res_df['hclose60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # if res_df['dc_lower_1m'].iloc[i] >= res_df['hlow_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   mr_score += 1

    # mr_const_cnt += 1
    # if res_df['dc_upper_1m'].iloc[i] <= res_df['hhigh_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # # if res_df['dc_upper_1m'].iloc[i] <= res_df['hopen_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   mr_score += 1

    # -------------- candle_dt_k -------------- #
    # mr_const_cnt += 1
    # # if res_df['dc_upper_1m'].iloc[i] <= res_df['hclose60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # if res_df['dc_upper_1m'].iloc[i] <= res_df['hhigh_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   mr_score += 1

    # mr_const_cnt += 1
    # if res_df['dc_lower_1m'].iloc[i] >= res_df['hlow_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    # # if res_df['dc_lower_1m'].iloc[i] >= res_df['hopen_60'].iloc[i - (np_timeidx[i] % config.loc_set.zone.c_itv_ticks + 1)]:
    #   mr_score += 1

    return
