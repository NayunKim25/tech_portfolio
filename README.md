# tech_portfolio


### 1. In-vehicle Coupon Recommendation Data Analysis

#### "데이터사이언스 공통 중 Python, SQL, R, Java, Scala, Go, C/C++, Javascript 등 데이터 처리 언어 활용 능력"

1) 사용 데이터 : Amazon Mechanical Turk 설문조사를 통해 수집된 데이터
  - 차량 내 추천 시스템에서 추천에 대한 고객의 반응을 파악하기 위해 수행
  - 목적지, 현재 시간, 날씨 등의 다양한 운전 상황에서 쿠폰 수락 여부를 묻는 설문조사

2) 분석 목표
  - 고객의 특징과 성향을 고려한 쿠폰 수락 여부를 예측하여 쿠폰을 제공함으로써 효과 극대화
  - 더 많은 고객이 서비스를 이용하도록 유도함으로써 마케팅 및 비즈니스 성장에 기여

3) 분석 내용
  - 데이터 전처리(결측치, 중복 데이터 제거)
  - EDA(변수 분포 파악, 쿠폰 수락 여부와 변수의 관계 파악(상관관계 분석))
  - classification 모델 비교(Decision Tree, Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes, KNN)

4) 분석 결과
  - 쿠폰의 수락 여부에 있어서 coupon(쿠폰의 종류), occupation(직업), income(수입)의 순서대로  
중요한 영향을 미치는 것을 알 수 있음
  - ColorDT의 Rule을 통해 탐색한 결과, 긴급하게 향하고 있는 목적지가 없고 평소에 카페를 자주 가는 사람은 20달러 미만 레스토랑 쿠폰을 수락할 확률이 높음
  - 또한 쿠폰의 유효기간이 짧고, 뚜렷하게 향하고 있는 목적지가 있으며 결혼한 상태의 사람은 쿠폰을 수락할 확률이 떨어짐
  - 이러한 Classification 결과와 EDA 결과를 종합해보면 20대 학생이 친구와 함께 있는 경우에 쿠폰을 수락할 확률이 가장 높으며, 평소 카페나 레스토랑을 자주 이용하는 사람이 뚜렷한 목적지가 없을 때 20달러 미만의 레스토랑 쿠폰을 제공하는 것이 가장 좋은 효과를 볼 수 있음
