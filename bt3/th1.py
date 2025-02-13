import tensorflow as tf

# Khoi tao bien
X = tf.constant([[0.0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y = tf.constant([[1.0, 0, 0],
                 [0, 1, 0],
                 [0, 1, 0],
                 [0, 0, 1]])

W = tf.Variable(tf.random.normal([2, 3]))
b = tf.Variable([0., 0., 0.])

print("W khởi tạo ngẫu nhiên ban đầu:\n", W.numpy())
print("b khởi tạo ngẫu nhiên ban đầu:\n", b.numpy())

# Hàm dự đoán
@tf.function
def predict(X, W, b):
    return tf.nn.softmax(tf.matmul(X, W) + b)

# Hàm mất mát
@tf.function
def MSE_loss(y, y_hat):
    return tf.reduce_mean(tf.reduce_sum((y - y_hat)**2, axis=1))

learning_rate = 0.1
for it in range(500):
    with tf.GradientTape() as t:
        y_hat = predict(X, W, b)
        current_loss = MSE_loss(y, y_hat)
    print(f"it: {it}, loss = {current_loss}")
    dW, db = t.gradient(current_loss, [W, b])
    W.assign_sub(learning_rate * dW)
    b.assign_sub(learning_rate * db)

predicted_y = predict(X, W, b)

print("Kết quả thực tế:\n", y.numpy())
print("Kết quả dự đoán:\n", tf.round(predicted_y).numpy())
