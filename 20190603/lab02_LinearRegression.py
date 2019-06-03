import tensorflow as tf

# X , Y ë°ì´í°
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# ê°ì  xW + b
hypothesis = x_train * W + b

# cost/loss function
# reduce_mean : í¹ì  ì°¨ìì ì ê±°íê³  íê· ì êµ¬í¨
cost = tf.reduce_mean(tf.square(hypothesis-y_train))

# Minimize
# Learning rate a ì ê°ì´ í°ê²½ì° ë°ì°í  ì ìì¼ë©°, ëë¬´ ìì ê²½ì° ìë ´íë ìëê° ì§ëì¹ê² ëë¦´ ì ìì
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0 :
        print(step, sess.run(cost), sess.run(W), sess.run(b))