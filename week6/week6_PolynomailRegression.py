import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#데이터 준비
x = np.array([[1],[2],[3],[4],[5]])
y = np.array([1,4,9,16,25])

#다항식 특성 생성
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)

#모델 학습
model = LinearRegression()
model.fit(X_poly, y)

#모델 저장
joblib.dump((model, poly), 'Poly_model.joblib', compress=3)

#모델 불러오기
loaded_model, loaded_poly = joblib.load('Poly_model.joblib')

#예측
#X_new = np.array([[6]])
X_test = poly.transform([[6], [7], [8]])

X_new_poly = poly.transform([[6]])
y_pred = model.predict(X_new_poly)
y_pred2 = model.predict(X_test)

print(f"다수 입력에 대한 예측 값: {[f'{x:.2f}' for x in y_pred]}")

for i, p in enumerate([6, 7, 8]):
    print(f"{p}에 대한 예측 값: {y_pred2}")