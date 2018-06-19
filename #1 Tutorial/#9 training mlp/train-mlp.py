from tensorflow.examples.tutorials.mnist import input_data
# from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
# mnist data training
mnist = input_data.read_data_sets("C:/Temp/xxx")
X_train = mnist.train.images
y_train = mnist.train.labels
y_train = y_train.astype(np.int32)
X_test = mnist.test.images
y_test = mnist.test.labels
y_test = y_test.astype(np.int32)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
# classifier
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                         feature_columns=feature_columns)
classifier = tf.contrib.learn.SKCompat(classifier)
classifier.fit(x=X_train, y=y_train, batch_size=50, steps=40000)

y_pred = list(classifier.predict(X_test))
# accuracy_score(y_test, y_pred)

print(y_pred['classes'])