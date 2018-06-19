import numpy as np
import pandas as pd
# import scipy

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

# print(train_set)

# 저리2 숫자로 변환
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder = LabelEncoder()
data_cat = train_set["w"]
data_cat_encoded = encoder.fit_transform(data_cat)
# print(data_cat_encoded)

#encoder = OneHotEncoder() # 원핫인코딩 에러 발생
#data_cat_1hot = encoder.fit_transform(data_cat_encoded.reshape(-1, 1))
#data_cat_1hot.toarray()
# print(data_cat_1hot.toarray())

# x에 비해 y의 범위가 다르다. 스케일이 다르다. y값에 의해 x값이 필요없어짐. 크기를 비슷하게 변환

from sklearn.preprocessing import StandardScaler

train_set = train_set.drop("w", axis=1)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(train_set);
# print(scaler.mean_);
# print(scaler.var_);
train_set = scaler.transform(train_set);
# print(scaled_data)

# =======================================튜플 추가
train_set["w"] = data_cat_encoded

# print(train_set)

# 학습시키기 위한 처리, input데이터와 학습 데이터 분리
train_set_input = train_set.drop("z", axis=1)
train_set_output = train_set["z"]

# 선형식
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_set_input, train_set_output)



print(train_set)

#카테코리카인코더