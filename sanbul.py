import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1-1: 데이터 불러오기 및 로그 변환
url = "https://github.com/samkeun/data/raw/refs/heads/main/sanbul2/sanbul2.csv"
fires = pd.read_csv(url)

# 1-2: 기초 정보 출력
print(fires.head())
print(fires.info())
print(fires.describe())

print("\nValue counts for 'month':\n", fires['month'].value_counts())
print("\nValue counts for 'day':\n", fires['day'].value_counts())
print("1-2 끝")

# 1-3: 데이터 시각화 - burned_area 히스토그램
fires["burned_area"].hist(bins=50)
plt.title("Histogram of Log-Transformed Burned Area")
plt.xlabel("log(burned_area + 1)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
print("1-3 끝")

# 1-5: Stratified Shuffle Split (month 기준)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

print("\nStratified Test Set (month 비율):\n",
      strat_test_set["month"].value_counts(normalize=True))
print("\n전체(month 비율):\n",
      fires["month"].value_counts(normalize=True))
print("1-5 끝")

# 1-6: scatter_matrix 출력
selected_features = ['avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
scatter_matrix(fires[selected_features], figsize=(10, 8))
plt.suptitle("Scatter Matrix of Selected Features")
plt.show()
print("1-6 끝")
# 1-7: 지역 시각화 (원의 크기는 max_temp, 색은 burned_area)
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
           s=fires["max_temp"] * 5, label="max_temp",
           c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.title("Fire Location and Burned Area")
plt.show()
print("1-7 끝")
# 1-8: 카테고리 특성 처리 (OneHotEncoder)
fires = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()

fires_num = fires.drop(["month", "day"], axis=1)
cat_attribs = ["month", "day"]
num_attribs = list(fires_num)
print("1-8 끝")

# 1-9: Pipeline 구축
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires)

print("\n데이터 전처리 완료. ")

import joblib

joblib.dump(fires_prepared, "fires_prepared.pkl")
joblib.dump(fires_labels, "fires_labels.pkl")
joblib.dump(full_pipeline, "full_pipeline.pkl")
joblib.dump(strat_test_set, "strat_test_set.pkl")

print("🔥 전처리 결과 저장 완료 (pkl 파일들)")
