# 1. 유현모

LinearRegression 이론 관련 공부 및 기초 실습, 실행
hypothesis, cost 
Wx + b 로 가정하고 실습 진행 hypothesis = x_train * W + b

reduce_mean : 특정 차원을 제거하고 평균을 구함
Learning rate a 의 값이 큰경우 발산할 수 있으며, 너무 작은 경우 수렴하는 속도가 지나치게 느릴 수 있음

Minimize : Gradient Descent using derivative : W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W*X-Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

Manual gradient
gradient = tf.reduce_mean((W*X-Y) * X) * 2