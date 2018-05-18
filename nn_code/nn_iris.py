#as3:/usr/local/lib/python2.7/site-packages# cat sitecustomize.py
# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt




# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

i_start = int(len(x_data)*0.7) #70% de los datos
i_valid = int(len(x_data)*0.85) #85% - 70% de los datos

x_train_data = x_data[:i_start]
y_train_data = y_data[:i_start]

x_valid_data = x_data[i_start + 1:i_valid]
y_valid_data = y_data[i_start + 1:i_valid]

x_test_data = x_data[i_valid + 1:]
y_test_data = y_data[i_valid + 1:]


print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.02).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
valid_list = []
epoch_list = range(0, 1500)
menorError = 9999
epoh = 0;
for epoch in xrange(5000):
    for jj in xrange(int(i_start / batch_size)):


        batch_xs = x_train_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train_data[jj * batch_size: jj * batch_size + batch_size]

        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    error = sess.run(loss, feed_dict={x: x_valid_data, y_: y_valid_data})

    valid_list.append(error)


    print np.absolute(menorError-error), "-->", (menorError*0.05)
    print menorError, "***", error
    if np.absolute(menorError-error) < (0.0001):
        epoh = epoch
        break
    else:
        menorError = error

    """print epoch, "-->", error
    if epoch > 19:
        if error < menorError:
            menorError = error
        else:
            epoh = epoch
            break;"""


plt.plot(valid_list)
plt.show()

result = sess.run(y, feed_dict={x: x_test_data})
bad = 0
for b, r in zip(y_test_data, result):
    if np.argmax(b) == np.argmax(r):

        print b, "-->", r, " *** Se ha clasificado bien"
    else:
        print b, "-->", r, " *** Se ha clasificado mal"
        bad = bad + 1
print "\nSe han clasificado mal", bad, "muestras."
print "\n Se ha parado el la epoca", epoh
print "----------------------------------------------------------------------------------"
