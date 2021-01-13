## SupportLine & ResistanceLine

### **<span style= "color:red">  = Oversold, Overbought </span>**

1. <u>내가 정한 매수가(선)</u>이 **지지선**의 **최저가**인가
2. <u>내가 정한 매도가(선)</u>이 **저항선**의 **최고가 / 고가** 인가

| Finding Signal | Setting Method                                               | Remark                                   |
| -------------- | ------------------------------------------------------------ | ---------------------------------------- |
| Support        | 인공지능 / **RSI** /                                         | **다양성은 없어도 정확도가 높아야한다.** |
| Resistance     | **지정 수익제** + 비지정 수익제 / 인공지능 + 퀀트 / 인공지능은 다양성을 보장하지 못한다.  / 지정 수익제로 n% 수익을 꾀하고 | **다양성과 정확도가 높아야한다.**        |

| Model_Num | Setting                                      | Result                             |
| --------- | -------------------------------------------- | ---------------------------------- |
| _124      | **crop_size_scale 데이터셋 사용** 작은 idl값 |                                    |
| _125      | 변곡점 다음을 라벨링한다.                    | crop_size_scale 정확도가 떨어진다. |
| _126      | 60 + none max pooling                        |                                    |
| _127      | _125 max pooling                             |                                    |

* 지지선을 다 찾을 필요가 없다. 가장 정확한 지지선만 찾는다.

  

## Humachine

손절가 = 매수가의 1% 이하로 떨어지면 / 들어가는 시점에서 이전 저점의 밑으로 떨어지면

매도가 = **이전 고점 언저리**

인공지능을 이용해 지지 저항을 찾을 수 없다면 **Human + Machine**으로 가자



* **PyQT UI를 사용한 자동거래 ㄱㄱ, 거래 완료될 때까지 못기다리겄다..** 
  * 항상 7개의 후보 코인을 유지 (7개의 탭)
  * 매수 등록 취소는 가능한데 / **<u>매도 등록 취소하고 재등록</u>**이 안돼네?



## Logic

|      | 매수                                                         | 매수가 | 매도                   | 매도가 | 손절                      |
| ---- | ------------------------------------------------------------ | ------ | ---------------------- | ------ | ------------------------- |
| 1    | 전 아랫 봉우리 > 전전 아랫 봉우리 < 전전전 아랫 봉우리 **(<span style="color:blue">V shape</span>)** |        |                        |        |                           |
| 2    | Stochastic Setting을 변경해서 지지 고저에 가장 최적화시킨다. |        |                        |        |                           |
| 3    | Fisher crossunder ~ over peak                                |        | crossover ~ under peak |        | 매도가로 부터 다단계 매도 |
| 4    |                                                              |        |                        |        |                           |



## Logic Feedback

| Logic Number | Feedback                                                     |
| ------------ | ------------------------------------------------------------ |
| 3            | 1. 저점은 잘 찾지만 고점이 빨리 도래한다. 최종 고점까지 smooth하게 가능 방법을 고안한다. = 저점 **period 고점 period 따로 존재한다.**  2. 저점을 찍고 쭈욱 하강하는 경우 3. 추세가 하강하다가도 상승하는 경우 |
|              |                                                              |
|              |                                                              |



## Find Best Set

1. 일별로 데이터셋을 구축한다. (여러가지 파라미터를 일별로 적용) 
2. **일별 데이터를 비교해 가장 잦은 분포의 인자값을 검출한다.**
   * 왜 nan값이 생기는지 확인
   * min_profit이 1.0이 나오는데 확인하기 (차트 데이터는 끝났지만 거래가 끝나지 않은 경우라고 판단된다.)
3. 손절가를 정해야한다. **손절된 코인은 short_value 찍은 후에 거래한다. **"손절가를정해하는이유.png파일 확인"
4. 고점을 지났는지 확인한다. (**checked**)
5. 변동성 최저점을 찾아서 진입하기에는 체결이 잘 안됀다.. (**checked**)
   * **early entry 가 대체적으로 수익률이 높다.** (손해율도 조금 높지만 전체적으로 고려해보았을때)
   * **<u>최저점을 찾아서 적절히 진입하는게 맞는거 같은데</u>**
6. 지속적 하락패턴을 예고하는 지표가 있나?
   * 선행지표인 fisher로서 후행지표인 트렌드 지표를 선택하는 건 알맞지 않은 것 같다.
   * 보통 진입할때 하향 트렌드인데 상향 트렌드를 기다리다 손절한다.
7. **<span style="color:magenta">fisher가 너무 낮아지면 하락세를 예고하는 듯한데?</span>**
8. 

## Reference

웹크롤링 - iframe 처리하기 : https://dejavuqa.tistory.com/198

Stochastic RSI : https://school.stockcharts.com/doku.php?id=technical_indicators:stochrsi

fisher : https://www.prorealcode.com/prorealtime-indicators/fisher-transform/

fisher code = ?https://www.prorealcode.com/prorealtime-indicators/kuskus-starlight/



