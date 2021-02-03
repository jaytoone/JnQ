## 인공지능 가상화폐 주문 비서 개발



## Intro

최적의 매수 / 매도 포인트를 찾기 위해 CNN 모델을 사용해 **거래로직**이 적용된 차트 이미지를 학습하고 최적의 포인트를 도출한다. **Bithumb** API를 사용해 자동매매의 형식을 택했다.



## Description

* Make_X (...) : **<u>파일명이 Make_X로 시작하는</u>** 파일은 CNN 모델에 학습하기 위해 차트로부터 거래로직이 적용된 이미지 데이터를 생산하기 위한 파일이다. **분류 모델**이기 때문에 라벨링된 y값이 생성된다. 

* Funcs_ (...) : 거래에 필요한 데이터를 제작하기 위해 필요한 함수들을 모아놓은 곳이다. **Bithumb API에서 제공하는 기능**들은 물론 **각종 지표**가 존재하며 **보유한 거래 알고리즘을 기반으로 수익을 확인하는 함수** 또한 존재한다.
* Make_pred_ (...) : 학습된 인공지능 모델 파일 .hdf5을 로드해 **실시간 차트에 적용해보는 파일**이다. 
* BVC_ (...) : BestValueCheck로 거래 알고리즘을 개발하는 과정에서 최적의 수익률을 도출하는 최적의 세팅을 선별하기위해 만들었다.
* System_ (..) : 보유한 거래 로직으로 **실제로 자동매매**를 하는 파일이다. <u>Bithumb API를 사용해 자동매매 하는 방식이 궁금한 사람은 이 파일을 참고하면 좋다.</u>



## Result

![Figure_1](https://user-images.githubusercontent.com/50652715/81029875-92be5180-8ec1-11ea-88ed-e3c64a2f3423.png)

**인공지능을 사용해 매수 / 매도 포인트를 포착한 예시** 

가장 상단의 청록색 선이 매수 포인트를 의미하며 밑에서 두번째 그림의 자주색 선이 매도 포인트를 의미한다.



**각 모델에 대한 상세한 세팅과 설명 그리고 결과를 담은 Markdown 파일 리스트들을 아래와 같이 정리하였다.** 

1. 고저점 찾기.md
2. 고저점 찾기2.md
3. EMA_RIBBON.md
4. MACD_OSC.md
5. Quant.md
6. Gaussian.md
7. Support_Resistance_Line.md



## 고저점 찾기.md 열어본 예시

## Tune Results.

| Model No.      | Description                                                  |                                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 54_0           | y data 전전 데이터                                           |                                                              |
| 54_1           | y data 전 데이터                                             |                                                              |
| 54_2           | 54_1 버전 60만개 데이터                                      |                                                              |
| 54_3           | 54_1 버전 고저점을 뒷 데이터만으로 만든 경우 / check_span=60 | 훨씬 안정적 : 입력데이터의 라벨양이 좀 있어야한다.           |
| ~~54_4~~       | 54_3 버전 개선 / + CMO                                       |                                                              |
| 54_5           | 54_3 버전 check_span=40                                      |                                                              |
| 54_6           | 54_3 버전 check_span=30                                      | **30이 적당함**                                              |
| 54_7           | 54_6 버전 / + MA15                                           |                                                              |
| 54_8           | 54_3 버전 check_span=20                                      |                                                              |
| 54_9           | 54_6 버전 / + OBV                                            |                                                              |
| 54_10          | 54_6 버전 / - MA60 / + OBV                                   |                                                              |
| 54_11          | 54_6 버전 / - MA60 / + CMO                                   | CMO 데이터는 예측에 안맞는다..                               |
| 54_12          | 54_10 버전 / - CNN / + GAN                                   | 보류 : GAN 을 사용하는 게 효율적인지에 대해 고민해보아야한다. |
| 96_0           | 54_10 버전 / - 54 / + 96                                     |                                                              |
| 54_10_copied   | LIMIT_LINE 을 낮춰보기 / 저점은 높게 고점은 낮게 / 저점 : 0.9 고점 : 0.8 |                                                              |
| 54_13          | 54_10 버전 / 54 개의 데이터를 자른 후 scaling 하기 (price & volume) | MADE_X 는 crop 안하는게 좋음.. 무분별해짐                    |
| 54_14          | 54_10 버전 / crop_size >> (all)                              |                                                              |
| 54_15          | 54_10 ver. / - Volume                                        |                                                              |
| 54_16          | 54_10 ver. / 1월 이후의 데이터로 Made_X                      |                                                              |
| 54_17          | 54_16 ver. / - Volume                                        | 왜 떨어지지.. OHLC 는 올랐는데                               |
| 54_18          | 54_10 ohlcvobv 급등 예측                                     |                                                              |
| 54_19          | 54_16 1월 데이터 급등 예측                                   | 예측이 안됌..                                                |
| **54_20**      | y_value 가 4개 (진입, 저점, 고점, 거래 안함)                 |                                                              |
| 54_16_copied   | 54_16 ver. / np.argmax 수정                                  | 수익률 낮아짐                                                |
| **54_21**      | y_value 가 3개 (저점, 고점, 거래 안함) / max_value * limit_line 으로 |                                                              |
| 54_22          | 54_21 - OBV + CMO                                            |                                                              |
| **54_23**      | only ohlc                                                    | **Best  Model**                                              |
| 54_24          | 54_21 - OBV + RSI                                            |                                                              |
| 54_25          | 54_21 - OBV + MACD                                           | 안해봄                                                       |
| 54_26          | ohlcv                                                        |                                                              |
| 54_21_copy     | 54_21 * 0.7                                                  | 영향력이 적다.                                               |
| 54_27          | model_ensemble                                               | 흐..                                                         |
| 54_21_copy2    | 54_21 + 1.0                                                  | 영향력이 적다.                                               |
| 54_28          | 54_23 input_data_length = 30 ver.                            | 그냥 그럼..                                                  |
| 54_23          | 정확도 상승 / kfold                                          | 여러개를 돌려서 가장 높은 정확도를 얻을 수 있고 수익률과 비례하는 성향을 띈다. **좋다** |
| ~~54_29 & 30~~ | 54_23 MADE_X WITH CROP_SIZE 500 & 100                        | Model 만든게 부끄럽다.. <u>그냥 엉망임</u>                   |
| 54_31          | 코인별 모델                                                  | 데이터의 양이 부족해 정확도는 높게 나오지만 실 예측이 떨어진다. |
| 54_32          | 저점 / 고점 모델만                                           | 정확도 떨어진다.                                             |
| 54_33          | OHLC + MA60                                                  | **장단점이 있는거 같음** (좀더 안정적이다?)                  |
| 54_21_copied   | 54_21 ver. 2월 데이터 추가                                   | 뭐랄까.. 예측 분포가 조금 떨어진달까 중간 고점들을 지나치는 경우들이 존재함 / 고점 저점에 대한 정확도도 떨어짐 |
| 54_35          | 고점과 저점의 폭이 큰 이미지만 사용                          | y_label 양이 적어서 그런지 잘 못찾음.. 딱히 급상승을 예측하는 것두 아니고 |
| 54_36          | 데이터의 양을 줄여본다. 어떤 걸 제외시킬까                   |                                                              |
| 54_37          | CROP_SIZE 제거하고 OHLC 로 거래 진행 기존에 OBV 가 문제가 되었지만 OBV 제거 했으니 무관하다. | 최고 저점에서만 진입하는 문제점.. 중간 진입이 안된다는 소리 / 고점도 마찬가지로 차트의 최대고점을 찍으면 바로 팔아버린다. = 해당 스케일 차트의 최저와 최고점을 찾아준다 |
| 100_38         | input_data_length 가 길어질수록 중간 봉우리를 예측하는 확률이 올라가는 거같음 | 고점 먹기가 힘들어짐 idl 이 올라갈 수록                      |
| 300_39         | IDL 300                                                      | origin : 중간 고저점이 좀 더 나온다 / crop : 350 으로 잘랐을 경우 고점에서 저점이 잡히는 불상사.. IDL 을 조정해야한다. CROP_SIZE 는 IDL 보다 같거나 커야하기 때문에 |
| 100_40         | rapid_ascend ohlc                                            | 40 > 41                                                      |
| 100_41         | rapid_ascend ohlcv                                           |                                                              |
| 100_42         | rapid_ascend ohlcma                                          | 42 > 40                                                      |
| 100_43         | rapid_ascend ohlcobv                                         | 10-09 PPT 찾기 **BEST?**                                     |
| 100_44         | rapid_ascend ohlcvrsi                                        |                                                              |
| 100_45         | rapid_ascend ohlccmo                                         |                                                              |
| 100_46         | rapid_ascend ohlcmacd                                        |                                                              |
| 300_47         | CROP_SIZE = IDL                                              | 저점을 잘 찾지 못한다. 이전 저점을 기준으로 crop 하는게 낫지 않나 |

>  #### Made_X (GPU) >> Make_model (GPU) >> Make_pred (GPU) >> BVC_CNN2



##  DATASETS

* #### 애초에 데이터를 만들때 추세와 수익저점, 고점을 반영한다. 상승 저점만 라벨링

* ### spectrogram 

  https://kaen2891.tistory.com/39

* 거래 로직을 다 때려박아놓는다.

* 내가 만들고 싶은 이미지파일이 무엇인지

  * y 라벨값이 1개인 파일은 데이터셋에 포함하지 않는다. (이유 : 실제로 fluc_close 가 1인 데이터지만 잘려서 1로 표시되지 않아 데이터 혼선을 가져올 수 있다.)
  * 0.4 RANGE_FLUC

  * np.max >> 일단은 수용해야할 오차범위인가.. (데이터 스케일 사이즈와는 무관하지만 y_pred_fluc_close 의 최대값)



### 보정

> #### CROP_SIZE 와 LIMIT_LINE 의 조합이 중요
>
> 1. MODEL_ACC
> 2. ~~CROP_SIZE (부분 확대하는 느낌)~~
> 3. LIMIT_LINE
> 4. ~~SUDDEN_DEATH~~
>    * ~~찾아야 되는 패턴 : 저점이 갱신되었을때 고점의 반응~~
>
> ---
>
> #### ~~일정 틱 이후(CHECK_SPAN) 지지선이 갱신되면 손절모드로 전환~~
>
> #### 익절모드 : 최대 고점을 노린다.
>
> > crop_size_high=300
> >
> > limit_line_high=0.65
>
> 최대 고점을 노리다 보면, 고점이 안잡힌다. 어떻게 할 건가
>
> * ##### <u>고점이 안잡히는 경우 보정</u>
>
> #### 손절모드 : 최저 고점을 노린다. >> 바로 팔아버리기
>
> > crop_size_sudden_death=100
> >
> > limit_line_sudden_death=0.45
>
> ---
>
> ~~LIMIT_LINE 을 사용할 때,  지정값을 사용하는 방법말고, MAX 값부터 N 개를 표시하는 방법은 어떨까~~



## Tuning

### * Data Tune

> 데이터를 좀더 이쁘게 잘라준다
>
> check_span을 줄여보기

### * Model Tune

#### Best Parameter.

| Best Model : softmax / categorical_crossentropy / batch_size=128 / rotation=60 / h_flip=True / width_shift=0.6 / height_shift=0.6 / fill_mode='nearest / filter_amt=100 / dense_relu / kernel_size=3,1 / |      |
| ------------------------------------------------------------ | ---- |
| 베이지언 하이퍼 파라미터 최적화 검색해보기 (automl 관련해서) |      |

* #### Model Ensemble

  * https://towardsdatascience.com/how-i-used-transfer-learning-and-ensemble-learning-to-get-90-accuracy-in-kaggle-competition-5a5e4c7e63e
  * not_trainable : [https://inspiringpeople.github.io/data%20analysis/Ensemble_Stacking/](https://inspiringpeople.github.io/data analysis/Ensemble_Stacking/)

* #### ParamSearchCV

  Link : https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

  optimizer

  epoch 늘리기 (크게 의미가 없을거 같다, 100 안에서 꿑나고 있으니까)

  batch size 

  

### 거래 코드

#### 1. 진입 코인 선정

##### <u>결국 마지막 IDL 을 이용해 현재 포인트가 저점인지 아닌지를 판단한다.</u>

> 이전 저점보다 현재 저점의 갭이 0.1 이상이면 진입
>
> DATA_CHECK_SPAN = 20
>
> * REAL_TIME_CROP 적용
> * 이전 저점 포인트와 현재 포인트의 거리가 DATA_CHECK_SPAN 이상
> * **최근 저점을 가장 잘 예측하는 SLICE or CROP_SIZE 를 찾는다.**
> * 저점으로 예상해서 진입하려고 하면, 이전 저점을 찾아서 (현재로부터 적어도 CHECK_SPAN 만큼 떨어져있다.), 이전 저점 보다 0.1 이상 크면 진입한다.

#### 2. 매수 시점 = 저가

#### 3. 매수가 = 이전 종가

#### 4. 매도 시점 = 고가

#### 5. 매도가 = 55초 시장가 매도 (매도법 생각해보기)

#### 6. 이탈에 대한 보정 작업 : 손절 모드

> 이전 저점과 현재 포인트 종가 기울기가 손절 기울기(음수값) 보다 작아지면 시장가 매도

#### +a. 매수가 / 매도가 선정 코드

* 종가를 기준으로 예측하는 것이니, 55초 이후의 현재가격으로 매수 / 매도하면 될까
* 저가는 그렇게 차이가 나지는 않네



## Reference

CNN 기반 주가 예측 : Taewook Kim, Ha Young Kim, Forecasting stock prices with a feature fusion LSTM-CNN 