import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

import DNNload_data as data

(x_train, y_train), (x_test, y_test) = data.load_data()
#저수준 API
n_inputs = len(x_train)
n_hidden1 = 300
n_hidden2 = 100
n_output = 2
# 실행 횟수
n_epochs = 4300
# 미니배치 크기 정의
batch_size = 100
# 훈련 데이터와 타겟
x = tf.placeholder(tf.float32, shape=(batch_size, 6), name="x")  # x_train 입력층 역할
y = tf.placeholder(tf.int64, shape=(batch_size), name="y")  # y_train

# 심층 신경망 생성
# dense() 함수는 모든 입력이 은닉층에 있는 모든 뉴런과 연결된 완전 연결 층을 만든다.
with tf.name_scope("dnn"):
    # activation 매개변수로 활성화 함수를 지정
    hidden1 = tf.layers.dense(x, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_output, name="outputs") # softmax 활성화 함수로 들어가기 직전의 신경망 출력
    softmax = tf.nn.softmax(logits)
# 비용 함수 생성
with tf.name_scope("loss"):
    # 로짓을 기반으로 크로스 엔트로피를 계산, 0에서 '클래스 수 - 1'(여기서는 0에서 9사이) 사이의 정수로 된 레이블을 기대
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss") # 모든 샘플에 대한 크로스 엔트로피 평균을 계산

learning_rate = 0.1
# 비용 함수를 최소화
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
# 신경망의 전체 정확도
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# 모든 변수를 초기화하는 노드를 만들고 훈련된 모델 파라미터를 디스크에 저장하기 위한 Saver 객체를 생성
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# 실행 단계
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(n_inputs // batch_size): # 훈련 데이터의 크기를 미니배치 크기로 나눈 횟수만큼 반복
            x_batch = x_train[iteration * batch_size: (iteration + 1) * batch_size]
            y_batch = y_train[iteration * batch_size: (iteration + 1) * batch_size]

            y_batches = np.reshape(y_batch, (batch_size))
            sess.run(training_op, feed_dict={x: x_batch, y: y_batches})
        acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batches})
        acc_test = accuracy.eval(feed_dict={x: x_batch, y: y_batch})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "C:/Temp/my_model_final_100.ckpt") # 모델 파라미터를 디스크에 저장
# # 신경망 사용하기
# with tf.Session() as sess:
#     saver.restore(sess, "c:/Temp/my_model_final.ckpt")
#     x_new_scaled = x_test # some new images (scaled from 0 to 1)
#     Z = logits.eval(feed_dict={x: x_new_scaled})
#     y_pred = np.argmax(Z, axis=1)
#     print(y_pred)

# 한 번에 한 개 층씩 만들어 준다.
# 입력, 뉴런수, 층 이름, 활성화 함수
# 신경망 층을 만드는 함수, tf.layers.dens() 함수로 대체 가능
def neuron_layer(x, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])  # previous layer의 뉴런의 수를 파악
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        w = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="bases")
        z = tf.matmul(x, w) + b

        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

