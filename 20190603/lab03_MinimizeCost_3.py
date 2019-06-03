import tensorflow as tf
import matplotlib.pyplot as plt

W = tf.Variable(5.)
X = [1,2,3]
Y = [1,2,3]

hypothesis = X * W

# Manual gradient
gradient = tf.reduce_mean((W*X-Y) * X) * 2


cost = tf.reduce_mean(tf.square(hypothesis-Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Get gradients
gvs = optimizer.compute_gradients(cost, [W])

# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)
