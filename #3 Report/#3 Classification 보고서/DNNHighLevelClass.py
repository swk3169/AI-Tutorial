import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

import DNNload_data as data

(x_train, y_train), (x_test, y_test) = data.load_data()
# 고수준 API
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
# DNNClassifier의 객체를 생성
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], feature_columns=feature_columns, n_classes=2, dropout=0.5)
# 사이킷런 인터페이스와 호환되게 만들어주는 SKCompat
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
dnn_clf.fit(x=x_train, y=y_train, batch_size=100, steps=10*10)  # 사이즈와 훈련 횟수를 조정

y_pred = dict(dnn_clf.predict(x_test))

print(str(accuracy_score(y_test, y_pred["classes"])*100))