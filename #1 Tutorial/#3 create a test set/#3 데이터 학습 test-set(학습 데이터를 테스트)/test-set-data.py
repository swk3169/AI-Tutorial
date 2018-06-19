import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# f = open("formula2.csv", "w")
# f.write("w,x,y,z\n",)
#
# np.random.seed(40)
#
# w = ["young"] * 50 # 50개 들어가있는 리스트
# x = 4*np.random.rand(50) -2
# y = 2000*np.random.rand(50) -1000
# z = x + y + np.random.randn(50) # 노이즈를 주는 이유는 더 정확한 이유를 위해
#
# for v1, v2, v3, v4 in zip(w,x,y,z): # 4개의 튜블로 만들어진 것
#     f.write(v1 + "," + str(v2) + "," + str(v3) + "," + str(v4) + "\n")
#
# w = ["old"] * 50
# x = 4*np.random.rand(50) -2
# y = 2000*np.random.rand(50) -1000
# z = 3*x + 5*y + np.random.randn(50) # 노이즈를 주는 이유는 더 정확한 이유를 위해
#
# for v1, v2, v3, v4 in zip(w,x,y,z): # 4개의 튜블로 만들어진 것
#     f.write(v1 + "," + str(v2) + "," + str(v3) + "," + str(v4) + "\n")
#
# f.close()

data = pd.read_csv("formula2.csv")
# print(data)

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)  # 분리

# print(test_set)
# print("========================================================================================================")
# print(train_set)

# 전처리, 비어있는것 채워넣은 처리

m = train_set.median()

train_set = train_set.fillna(m)  # 자체가 바뀌는것이 아닌 새로운 값 추가

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
data_cat = train_set["w"]
data_cat_encoded = encoder.fit_transform(data_cat)
#print(data_cat_encoded)
#encoder = OneHotEncoder()
#housing_cat_1hot = encoder.fit_transform(data_cat_encoded.reshape(-1,1))

train_set = train_set.drop("w",axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(train_set)
train_set = pd.DataFrame(scaler.transform(train_set), columns=["x","y","z"])

train_set["w"] = data_cat_encoded

trainset_input = train_set.drop("z", axis=1)
trainset_output = train_set["z"]

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#lin_reg = LinearRegression()
#lin_reg = DecisionTreeRegressor()
from sklearn.ensemble import RandomForestRegressor
lin_reg = RandomForestRegressor()
lin_reg.fit(trainset_input, trainset_output)

# 테스트 셋 전처리
m = test_set.median()
test_set = test_set.fillna(m)

encoder = LabelEncoder()
data_cat = test_set["w"]
data_cat_encoded = encoder.fit_transform(data_cat)

test_set = test_set.drop("w",axis=1)
# StandardScaler 정규화, 평균을 제거하고 단위 분산에 맞게 스케일링하여 피쳐를 표준화 하십시오.
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

# 추후의 스케일링에 사용될 평균 및 표준 편차를 계산합니다.
scaler.fit(test_set)

# 변환
test_set = pd.DataFrame(scaler.transform(test_set), columns =["x","y","z"])

test_set["w"] = data_cat_encoded

testset_input = test_set.drop("z", axis=1)
testset_output = test_set["z"]

predicted = lin_reg.predict(testset_input)

from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(testset_output, predicted)
lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)

