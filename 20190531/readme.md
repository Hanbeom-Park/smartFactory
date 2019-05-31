#20190531 유현모
import tensorflow as tf
# tf.__version

# 연산(op, operation) : 텐서 객체에 (또는 텐서 객체를 사용하여) 계산을 수행하는 점이다
# 0개 이상의 입력을 받아 0개 이상의 텐서를 변환.
# constant op 생성
# 이 명령어로 default graph 가 노드에 더해진다.
hello = tf.constant("Hello, TensorFlow!!!")

# 텐서플로우 세션 시작
sess = tf.Session()

# op 를 실행하고 결과를 얻는 부분
print(sess.run(hello))

# 텐서는 배열로서 어떤 값이든 될 수 있음
# 랭크(Rank), 모양(Shape), 종류(type)를 사용한다
# 랭크 : 몇 차원 배열인가? s = 448 , v = [1.1, 2.2, 3.3] , m  = [[1,2,3], [4,5,6]]
# 모양 : 각각의 원소에 몇 개가 들어 있는 가? [] - zero , [D0] - 1-D, [D0,D1] - 2-D
# 종류 : 데이터의 종류, 대부분 float32 사용.

# Computational Graph

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1 : ", node1, "node2: ", node2)
print("node3 : ", node3)

print("sess.run(node1, node2) : ", sess.run([node1, node2]))
print("sess.run(node3)", sess.run(node3))

# placeholder 다른 텐서를 할당하는 것
#  전달 파라미터는 dtype : 데이터 타입을 의미하며 반드시 적어주어야 한다
#  shape : 입력 데이터의 형태를 의미한다. 상수 값이 될 수 도 잇고 다차원 배열의 정보가 들어올 수도 있다
#  name : 해당 placeholder의 이름을 부여하는 것으로 적지 않아도 됨 ( 디폴트 파라미터 None)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# + provides a shortcut fo ft.add(a,b)
adder_node =  a + b

# feed_dict 변수로 데이터를 입력한다.
print(sess.run(adder_node, feed_dict={a :3, b:4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2,4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))