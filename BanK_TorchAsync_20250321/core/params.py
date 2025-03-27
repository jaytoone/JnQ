"""
v0.4.1
  - sync. vars name to PineScript. 20250317 1008.
"""

params = {
    'modes': ['adjDCOpen', 
              'adjBBOpen',
              
              'adjCCIMAGapStarter', 
              
              'adjBBCandleBodyAnchor', 
              'adjCandleBodyAnchor', 
              'adjCandleWickAnchor', 
              'adjHHLLAnchor',
              'adjBBWAnchor', 
               
              # 'adjDCBBSR',
              # 'adjDC2SafeSR', 
              
              # 'adjSARmmt',
              'adjDCmmt',
              
              'adjSARdir'
            ],


    ###### Indicator ######
    # Fisher Indicator
    'FisherLens': [30, 60, 120, 180, 240],  # 피셔 변환 길이
    'FisherBands': [1.0, 0.5, 0.0, 0.0, 0.0],  # 피셔 밴드
    'Fisher_lookback_len': 3,  # 피셔 방향성 체크 시 사용할 캔들 수

    # Donchian Channel (DC)
    'DC_period': 5,  # 기본 DC 기간
    'DC_safe_const': 0.05,  # DC 안전 범위 비율
    'DC2_period': 60,  # 장기 DC2 기간
    'DC2_safe_const': 0.1,  # DC2 안전 범위 비율

    # Commodity Channel Index (CCI)
    'CCI_period': 21,  # CCI 계산을 위한 기간
    'CCIMA_period': 14,  # CCI 이동 평균 기간

    # Bollinger Bands (BB)
    'BB_period': 20,  # BB 기간
    'BB_multiple': 1,  # 표준편차 배율
    'BB_level': 2,  # 추가 확장 레벨
    
    # ADX & DI (Directional Movement)
    'ADX_period': 14,  # ADX 계산 기간
    
    # SAR (Parabolic Stop and Reverse)
    'SAR_start': 0.02,  # SAR 시작값
    'SAR_increment': 0.02,  # 증가값
    'SAR_maximum': 0.2,  # 최대값
    
    
    
    ###### priceChannel ######
    'priceChannelMethod': 'BB1',
    
    
  
    ###### Starter ######
    'CCIMAGapBandLong': 145,
    'CCIMAGapBandShort': 145,

    ###### Anchor ######
    # BBCandleBodyRatio (%)
    'BBCandleBodyRatio_upper_long': 170,
    'BBCandleBodyRatio_lower_long': 100,
    'BBCandleBodyRatio_upper_short': 200,
    'BBCandleBodyRatio_lower_short': 100,

    # Candle Body (%)
    'bodyPctUpper_long': 100,
    'bodyPctLower_long': 75,
    'bodyPctUpper_short': 100,
    'bodyPctLower_short': 60,

    # Candle Wick (%)
    'wickPctUpper_long': 10,
    'wickPctLower_long': 0,
    'wickPctUpper_short': 10,
    'wickPctLower_short': 0,
    
    # BBW 한계값 (공통)
    'BBW_const': 0.9,

    ###### Momentum ######
    'DCmmt_const_long_min': 1,
    'DCmmt_const_long_max': 8,
    'DCmmt_const_short_min': 1,
    'DCmmt_const_short_max': 10,
    


    ###### TR ######
    'entry_gap_multi': 0,  # 
    'take_profit_gap_multi': 2,  # 익절 기준 배율
    'stop_loss_gap_multi': 2,  # 손절 기준 배율
    'expiry_gap_multi': 2,  # 만료 기준 배율 (Bank 는 take_profit 과 동일 사용 가능)
    
    
    'use_limit': True,
    'use_stop_loss': True,
    'use_stop_loss_barclose': False, # wait_barClose 미사용, asyncio 작업 대기 중
    'use_closer': True,
    # 'use_closer': False,
    'ncandle_game': 2,
    

    ###### Risk Management ######
    # 'target_loss': 100,  # 목표 손실 금액
    # 'target_loss_pct': 100,  # 목표 손실 비율 (%)
    # 'target_leverage': 5,  # 레버리지 목표값
    # 'fee_limit': 0.0002,  # 지정가 수수료
    # 'fee_market': 0.0005,  # 시장가 수수료
    # 'leverage_rejection': False,  # 레버리지 거부 여부
    # 'leverage_brackets': leverage_brackets  # 레버리지 제한 값
}
