import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

learning_data = pd.read_csv("formula1.csv")
print(learning_data.describe())

train_set, test_set = train_test_split(learning_data, test_size=0.2, random_state=42)
train_input = train_set.drop("y", axis=1)
train_output = train_set["y"]

lin_reg = LinearRegression() # 값을 선형화
lin_reg.fit(train_input, train_output)

t = [[1,1], [2,2], [3,3]]
predicted_values = lin_reg.predict(t)
print(predicted_values)

testset_output = test_set["y"]
testset_input = test_set.drop("y", axis=1)
predicted_values = lin_reg.predict(testset_input)

from sklearn.metrics import mean_squared_error
import numpy as np

tree_mse = mean_squared_error(testset_output, predicted_values)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse) # 학습 데이터에 노이즈를 주게 되면 오차가 커짐

from sklearn.metrics import mean_squared_error
import numpy as np

# import pandas as pd
#
# a = [[1,2,3],[4,5,6],[7,8,9]]
# data = pd.DataFrame(a, columns=('x','y','z'))
#
# z = data["z"]
# z = z*2
# print(z)
#
# data = data.drop("z", axis = 1)
# data["z"] = z
# print(data)