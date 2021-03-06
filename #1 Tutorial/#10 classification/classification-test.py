import pandas as pd
import numpy as np
import tensorflow as tf

iris = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(iris, sep=",")
# 속성 정의
attributes = ["sepal_length", "sepal_width", "petal_width", "petal_length", "class"]
# 입력 속성값 할당
input_attributes = ["sepal_length", "sepal_width", "petal_width", "petal_length", "class"]
df.columns = attributes

print(df)
df["class"] = df["class"].map({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2})

df = df.reindex(np.random.permutation(df.index))

train_size = int(len(df)*0.8)
train_set = df[:train_size]
test_set = df[train_size:]

# 분리
X_train, y_train, X_test, y_test = train_set[input_attributes], train_set["class"], test_set[input_attributes], test_set["class"]

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
# classifier
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                         feature_columns=feature_columns)   # hidden_units (계층수 [x,y])
classifier = tf.contrib.learn.SKCompat(classifier)
classifier.fit(x=X_train, y=y_train, batch_size=50, steps=4000)

#new_samples = np.array([6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7], dtype=float)
pred_y = classifier.predict(X_test) # test_set[input_attributes]
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred_y['classes']))