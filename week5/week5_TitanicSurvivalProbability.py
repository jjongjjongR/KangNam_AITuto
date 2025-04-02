import pandas as pd

#데이터셋 로드 > 전처리 > 나누기 > 훈련 > 예측 >평가

#타이타닉 데이터셋 로드
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

#데이터셋 확인
print(data.head())

#필요한 열만 선택
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

#결측값 처리
data['Age'].fillna(data['Age'].mean(),inplace=True)

#성별을 숫자로 변환
data['Sex'] = data['Sex'].map({'male':0, 'female':1})

#데이터 확인
print(data.head())

from sklearn.model_selection import train_test_split

#데이터셋을 훈련 데이터와 테스트로 나누기
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#데이터 분할 결과 확인
print("훈련 데이터 수: ", X_train.shape[0])
print("테스트 데이터 수: ", X_test.shape[0])

from sklearn.linear_model import LogisticRegression

#로지스틱 회귀 모델 생성
model = LogisticRegression()

#모델 훈련
model.fit(X_train, y_train)

#훈련 완료 메세지 출력
print("모델 훈련이 완료되었습니다.")

#테스트 데이터로 예측하기
predictions = model.predict(X_test)

#예측 결과 출력
print("예측 결과: ", predictions)

from sklearn.metrics import accuracy_score

#정확도 계산
accuracy = accuracy_score(y_test, predictions)

#결과 출력
print("모델의 정확도: ", accuracy)
