import numpy as np
from sklearn.linear_model import LinearRegression

#데이터 준비
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([1,4,9,16,25])

#모델 생성
model = LinearRegression()

#모델 학습
model.fit(x,y)

#회귀 계수와 절편 출력
print(model.coef_)
print(model.intercept_)

#예측
x_new = np.array([[6]])
#print(f"6에 대한 예측 값: {x_new[0]}")
y_pred = model.predict(x_new)
print(y_pred)
