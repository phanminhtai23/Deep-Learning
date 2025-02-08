# Xây dựng một mạng nơ ron perceptron đơn giản mô phỏng phép toán AND

import tensorflow as tf

X = tf.constant([[0.0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y = tf.constant([[0.0],
                 [0],
                 [0],
                 [1]])

# Khởi tạo ngẫu nhiên các trọng số W và b thay vì gán sẵn
W = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))

print("W khởi tạo ngẫu nhiên ban đầu:\n", W.numpy())
print("b khởi tạo ngẫu nhiên ban đầu:\n", b.numpy())

# Hàm dự đoán
@tf.function
def predict(X, W, b):
    return tf.nn.sigmoid(tf.matmul(X, W) + b)

# Hàm mất mát
@tf.function
def binary_crossentropy_loss(y, y_hat):
    epsilon = 1e-7  # Để tránh log(0)
    y_hat = tf.clip_by_value(y_hat, epsilon, 1. - epsilon)
    return -tf.reduce_mean(y * tf.math.log(y_hat) + (1 - y) * tf.math.log(1 - y_hat))


# Làm tròn so với 0.5, nếu >= 0.5 thì là 1, ngược lại là 0
@tf.function
def processed_prediction(X, W, b):
    return tf.round(predict(X, W, b))


alpha = 0.1
for it in range(500):
    with tf.GradientTape() as t:
        current_loss = binary_crossentropy_loss(y, predict(X, W, b))

    print(f"it: {it}, loss = {current_loss}")
    dW, db = t.gradient(current_loss, [W, b])
    W.assign_sub(alpha * dW)
    b.assign_sub(alpha * db)

print("W sau khi huấn luyện:\n", W.numpy())
print("b sau khi huấn luyện:\n", b.numpy())

y_hat = processed_prediction(X, W, b)

print("Kết quả thực tế:\n", y.numpy())
print("Kết quả dự đoán:\n", y_hat.numpy())
