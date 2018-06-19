import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

import DNNload_data as data

(x_train, y_train), (x_test, y_test) = data.load_data()

n_inputs = len(x_train)
n_hidden1 = 300
n_hidden2 = 100
n_output = 2

n_epochs = 100
batch_size = 100

x = tf.placeholder(tf.float32, shape=(batch_size, 6), name="x")  # x_train
y = tf.placeholder(tf.int64, shape=(batch_size), name="y")  # y_train

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(x, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_output, name="outputs")
    softmax = tf.nn.softmax(logits)

with tf.name_scope("loss"):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(entropy, name="loss")

learning_rate = 0.1
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(n_inputs // batch_size):
            x_batch = x_train[iteration * batch_size: (iteration + 1) * batch_size]
            y_batch = y_train[iteration * batch_size: (iteration + 1) * batch_size]

            y_batches = np.reshape(y_batch, (batch_size))
            sess.run(training_op, feed_dict={x: x_batch, y: y_batches})
        acc_train = accuracy.eval(feed_dict={x: x_batch, y: y_batches})
        acc_test = softmax.eval(feed_dict={x: x_batch, y: y_batch})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "c:/Temp/my_model_final.ckpt")

# with tf.Session() as sess:
#     saver.restore(sess, "c:/Temp/my_model_final.ckpt")
#     x_new_scaled = x_test # some new images (scaled from 0 to 1)
#     Z = logits.eval(feed_dict={x: x_new_scaled})
#     y_pred = np.argmax(Z, axis=1)
#     print(y_pred)

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

