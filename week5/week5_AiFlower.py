import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

#데이터 준비
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df.head())

#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#모델 훈련
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#예측하기
predictions = model.predict(X_test)
print("예측결과: ", predictions)
print("실제 값: ", y_test)

#모델 저장
joblib.dump(model, 'decision_tree_model.joblib', compress=3)

#모델 불러오기
loaded_model = joblib.load('decision_tree_model.joblib')