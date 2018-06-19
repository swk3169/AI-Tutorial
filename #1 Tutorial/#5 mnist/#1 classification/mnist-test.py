from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='/media/Vancouver/apps/mnist_dataset/')
print(mnist)

X, y = mnist["data"], mnist["target"]
print(X.shape)  # (70000, 784)
print(y.shape)  # (70000,)

# import matplotlib
# import matplotlib.pyplot as plt
# some_digit = X[36000]
some_digit = X[0]
# some_digit_image = some_digit.reshape(28, 28)  # 배열의 차원을 나타내줌 28 x 28
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()
# print(y[0])

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# training a binary classifier
# create data
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# train the model
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# predict a new instance
# print(sgd_clf.predict([some_digit]))

# 학습 데이터를 가지고 성능을 측정
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
print(accuracy)

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_train_pred))

sgd_clf.fit(X_train, y_train)  # y_train, not y_train_5

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
mat = confusion_matrix(y_train, y_train_pred)
print(mat)

import matplotlib.pyplot as plt
row_sums = mat.sum(axis=1, keepdims=True)
norm_conf_mx = mat / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()