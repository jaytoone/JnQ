# from numba import jit, njit, float64, float32, int8, int64, vectorize, guvectorize, prange
import numpy as np
# import torch

#   Todo    #
#       1. 아직 local 에서는 사용안함


def candle_ratio_v2(hrel, habs, hrel_v, habs_v, np_ones):
    hrel_mr = hrel == hrel_v if hrel_v >= 0 else np_ones
    habs_mr = habs == habs_v if habs_v >= 0 else np_ones

    return hrel_mr & habs_mr


def fast_candle_game_v2(fws, bs, fws_v, bs_v, np_ones):
    fws_mr = fws == fws_v if fws_v >= -10 else np_ones
    bs_mr = bs == bs_v if bs_v >= 0 else np_ones

    return fws_mr & bs_mr


# def candle_ratio_torch_v2(hrel, habs, hrel_v, habs_v):
#     hrel_mr = hrel == hrel_v if hrel_v >= 0 else torch.ones_like(hrel).bool()
#     habs_mr = habs == habs_v if habs_v >= 0 else torch.ones_like(hrel).bool()
#
#     return hrel_mr & habs_mr
#
#
# def fast_candle_game_torch_v2(fws, bs, fws_v, bs_v):
#     fws_mr = fws == fws_v if fws_v >= -10 else torch.ones_like(fws).bool()
#     bs_mr = bs == bs_v if bs_v >= 0 else torch.ones_like(fws).bool()
#
#     return fws_mr & bs_mr
#
#
# def candle_ratio_torch(hrel, habs, hrel_v, habs_v, hrel_gap=10, habs_gap=10):
#     hrel_mr = (hrel_v < hrel) & (hrel < hrel_v + hrel_gap) if hrel_v >= 0 else torch.ones_like(hrel).bool()
#     habs_mr = (habs_v < habs) & (habs < habs_v + habs_gap) if habs_v >= 0 else torch.ones_like(hrel).bool()
#
#     return hrel_mr & habs_mr
#
#
# def fast_candle_game_torch(fws, bs, fws_v, bs_v, fws_gap=10, bs_gap=10):
#     fws_mr = (fws_v < fws) & (fws < fws_v + fws_gap) if fws_v >= -100 else torch.ones_like(fws).bool()
#     bs_mr = (bs_v < bs) & (bs < bs_v + bs_gap) if bs_v >= 0 else torch.ones_like(fws).bool()
#
#     return fws_mr & bs_mr

# # @vectorize([int8(float64, float64, float64, float64, float64, float64)])
# @vectorize([int8(float64, float64, int8, int8, int8, int8)])
def candle_ratio_nbvec(hrel, habs, hrel_v, habs_v, hrel_gap=10, habs_gap=10):
  hrel_mr = (hrel_v < hrel) & (hrel < hrel_v + hrel_gap) if hrel_v >= 0 else 1
  habs_mr = (habs_v < habs) & (habs < habs_v + habs_gap) if habs_v >= 0 else 1

  return hrel_mr & habs_mr
#
#
# # @vectorize([int8(float64, float64, int8, int8, int8, int8)])
# @vectorize([int8(float64, float64, float64, float64, float64, float64)])
def fast_candle_game_nbvec(fws, bs, fws_v, bs_v, fws_gap=10, bs_gap=10):
  fws_mr = (fws_v < fws) & (fws < fws_v + fws_gap) if fws_v >= -100 else 1
  bs_mr = (bs_v < bs) & (bs < bs_v + bs_gap) if bs_v >= 0 else 1

  return fws_mr & bs_mr


# @jit
def candle_ratio_nb(input_data, hrel_v, habs_v, hrel_gap=10, habs_gap=10):
  hrel, habs = np.split(input_data, input_data.shape[-1], axis=1)
  hrel_mr = (hrel_v < hrel) & (hrel < hrel_v + hrel_gap) if hrel_v >= 0 else np.ones_like(hrel).astype(np.int8)
  habs_mr = (habs_v < habs) & (habs < habs_v + habs_gap) if habs_v >= 0 else np.ones_like(habs).astype(np.int8)

  return (hrel_mr & habs_mr).reshape(-1,)
#
#
# @jit
def fast_candle_game_nb(input_data, fws_v, bs_v, fws_gap=10, bs_gap=10):
  fws, bs, bws = np.split(input_data, input_data.shape[-1], axis=1)
  fws_mr = (fws_v < fws) & (fws < fws_v + fws_gap) if fws_v >= -100 else np.ones_like(fws).astype(np.int8)
  bs_mr = (bs_v < bs) & (bs < bs_v + bs_gap) if bs_v >= 0 else np.ones_like(bs).astype(np.int8)

  return (fws_mr & bs_mr).reshape(-1,)