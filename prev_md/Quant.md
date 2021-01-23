## Quant Algorithm



### 1. Low

```markdown
Support Line
저점 지표
CMO가 최저점을 찍었을때 상향추세인 경우
# MACD OSC, SUPERTREND, CMO
```



### 2. High

```markdown
Resistance Line
[최고점/고점]까지 갈 수 있는 높은 지표
+ 지표를 여러개 사용해 하나라도 만족하면 매도하는 방식 이용
# MA 종류
# 아닌 것 같은 지표 : CMO, BB, CHANDE KROLL STOP 
```



### 3. Exit

```markdown
최고점까지는 아니지만 안정적인 손절 지표
# SUPERTREND
```





## Strategy

| Low                                               | High         | Exit             |
| ------------------------------------------------------------ | ---------------------- | -------------------- |
| MACD_OSC > 0, ST(100, 1) RED2GREEN(실거래 = 조건 완성시, 매분 55초 종가 진입) 현재 OSC, 현재 ST | ST OFFSET OR SOMETHING | ST(100, 3) GREEN2RED |
|MACD_OSC > 0, ST(100, 3) RED2GREEN|ST OFFSET|-|
|**ST(X, Y) GREEN  + 인공지능 저점 예측**|**인공지능 고점 예측**|**ST(X, Y) RED**|
|상향추세 인공지능 저점 예측 = 거대한 상승추세의 시작점|인공지능 고점 예측|거대한 상승추세의 끝점|



## Consideration

```markdown
AI를 이용한 고저점 예측의 데이터셋 정의를 어떤식으로 변경해 결과치를 상향시킬 수 있을까?
```

| Model Num. | Description                                                  | Result                                                       |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| _102       | 30_57 수정/ check_span=60                                    | 첨예함이 떨어진 이유 : 같은 저점/고점인데 저점/고점으로 라벨링하지 않아서 헷갈려하는걸로 추측됀다. |
|            | ohlccmo                                                      | 첨예함이 떨어진 이유 : 일단은 최종 고점까지 가지 않고, cmo 영향을 받아 0라벨 영역에서 저점으로 결론짓는 경향이 생김 |
|            | ohlcmacd                                                     | 고점/저점 예측 정확도가 상승한다. : macd가 결과에 영향을 준다. (잔 저점이 사라진다.) >> **고저점과 맞물리는 macd를 설정해주면 좋다** |
| _103       | 30_57 수정/ dataset realtime crop : 데이터셋 형태에 대한 확실한 규명 작업 | **저점을 고점으로 인식한다. 끝**                             |
| _104       | **close_regression**                                         | 여기도 **crop_size**가 존재한다. 가장 높은 수익을 낼 수 있는 최적의 **crop_size**를 찾는게 관건임 |
| _105       | **osc_regression**                                           | **<u>데이터의 학습량을 늘리면 가능성이 보인다.</u>**         |



## SVM Regression

수익률을 close_reg 가 더 잘나오긴 하는데 실시간 reg와 커브 매칭이 잘 되는지가 관건 (close vs osc)

| Setting                                                      | Result                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| realtime_crop 두번째 방법 : crop_size를 정해주는게 아니라 max | 별로 않좋음..                                                |
| 한번에 reg_result를 구하고 매수 매도 하는거는 안맞지 않나    | 안맞는다! >> 수정완료                                        |
| **처음에 했던, 데이터누적 SVR 결과 잘 나오지 않았나?**       | 잘 안나온다.                                                 |
| **실시간 SVR로 거래 가능한가**                               | **crop_size=30, c, gamma, epsilon=0.1, 100, 0.5** 이런식의 실시간 SVR이 잘나온다. **gamma값이 크면 안됀다.** |
| **MACD_OSC로 최종 SVR 예측가능한가?**                        | **<u>유망주</u>** / 학습량이 많아지면 정확도가 올라가는지 파악한다. |
| **reg로 다음 reg를 예측하는 모델을 적용가능한가?**           | 불가하다.                                                    |



## MACD_OSC Regression

| Setting                                   | Result                                                       |
| ----------------------------------------- | ------------------------------------------------------------ |
| 데이터 학습량 증가시켜보기                | macd와 더 닮아 가려해서 않좋아지는것 같다.. >> **학습을 적게할 수 록 좋은거같은데??** |
| std scaler vs max abs scaler              | **std_scaler**                                               |
| 여러개의 feature 사용                     | 곡선의 feature를 사용해야할것임                              |
| 모델 깊이, 노드 수                        | 깊이는 영향 거의 없고, 노드수가 조금 있음                    |
| idl                                       | **낮은 idl**이 차트가 뒤로 밀리지 않는다.                    |
| dropout                                   | 잘 모르겠음.. = 영향력이 거의 없다는 거다.                   |
| batch size                                | 크면 **학습률이 떨어진다. 좋은거같기도**                     |
| macd_osc svr 세팅을 변경해서 학습시켜보기 | 1000, 10, 0.1                                                |



## 실시간 SVR vs Deep Learning SVR

```
Pycharm과 colab의 SVR조건 세팅이 다르다.
```





| 라벨링 결과                                                  |
| ------------------------------------------------------------ |
| 지지선                                                       |
| 저점                                                         |
| osc_reg, osc_dev **Robust Regression** : https://scipy-cookbook.readthedocs.io/items/robust_regression.html **Autogui** : https://lentner.io/2018/06/14/autogui-for-curve-fitting-in-python.html, Better SVR : https://stackoverflow.com/questions/48033510/tuning-parameters-for-svm-regression |
| osc_reg의 0기준선을 아래로 내려서 그 기준선 이상의 상황에서 하향 >> 상향하면 들어가고 상향 >> 하향하면 나온다. **gamma, C, epsilon의 영향을 파악한다.** |
|                                                              |

