import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from tensorflow.keras.datasets import mnist
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

a_0 = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


middle = 30

w_1 = tf.Variable(tf.truncated_normal([784, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 10]))
b_2 = tf.Variable(tf.truncated_normal([1, 10]))


def sigmoid(x):
	return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

# feedforward
z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigmoid(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigmoid(z_2)

diff = tf.subtract(a_2, y)

def sigmoidprime(x):
	return tf.multiply(sigmoid(x), tf.subtract(tf.constant(1.0), sigmoid(x)))

# backpropagation

d_z_2 = tf.multiply(diff, sigmoidprime(z_2))
d_b_2 = d_z_2
d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
d_z_1 = tf.multiply(d_a_1, sigmoidprime(z_1))
d_b_1 = d_z_1
d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)


#updating network

eta = tf.constant(0.5)
step = [
	tf.assign(w_1, tf.subtract(w_1, tf.multiply(eta, d_w_1))),
	tf.assign(b_1, tf.subtract(b_1, tf.multiply(eta, tf.reduce_mean(d_b_1, axis=[0])))),
	tf.assign(w_2, tf.subtract(w_2, tf.multiply(eta, d_w_2))),
	tf.assign(b_2, tf.subtract(b_2, tf.multiply(eta, tf.reduce_mean(d_b_2, axis=[0]))))
]


accuracy_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
accuracy_res = tf.reduce_sum(tf.cast(accuracy_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

def get_one_hot(x):
	temp = []
	for _ in x:
		temp_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		temp_arr[_] = 1
		temp.append(temp_arr)
	return np.array(temp)

for i in range(1000):
	batch_xs, batch_ys = x_train[i*10: (i+1)*10], y_train[i*10: (i+1)*10]
	batch_xs = batch_xs.reshape(10, -1)
	batch_ys = get_one_hot(batch_ys)
	sess.run(step, feed_dict={
		a_0: batch_xs,
		y: batch_ys
	})
	
	if i % 100 == 0 :
		res = sess.run(accuracy_res, feed_dict={
			a_0: x_test[:1000].reshape(1000, -1),
			y: get_one_hot(y_test[:1000])
		})
		print(res)
