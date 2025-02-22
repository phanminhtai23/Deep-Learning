import tensorflow as tf


# Khoi tao bo du lieu
X = tf.constant([[0.0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y = tf.constant([[1.0, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

# khoi tao bien ngau nhien W va b
W1 = tf.Variable(tf.random.normal([2, 2]))
b1 = tf.Variable(tf.random.normal([2]))
W2 = tf.Variable(tf.random.normal([2, 3]))
b2 = tf.Variable(tf.random.normal([3]))

print("W1 khoi tao ngau nhien:\n", W1.numpy())
print("W1 khoi tao ngau nhien:\n", W2.numpy())
print("b1 khoi tao ngau nhien:\n", b1.numpy())
print("b2 khoi tao ngau nhien:\n", b2.numpy())


@tf.function
def layer1(X1, W1, b1):
    return tf.nn.relu(tf.matmul(X1, W1) + b1)


@tf.function
def layer2(X2, W2, b2):
    return tf.nn.softmax(tf.matmul(X2, W2) + b2)


@tf.function
def predict(X, W1, b1, W2, b2):
    return layer2(layer1(X, W1, b1), W2, b2)


@tf.function
def categorical_crossentropy(y_true, y_hat):
    epsilon = 1e-7
    y_hat = tf.clip_by_value(y_hat, epsilon, 1. - epsilon)
    return -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(y_hat), axis=1))


a = 0.1
it = 500
for i in range(it):
    with tf.GradientTape() as t:
        y_hat = predict(X, W1, b1, W2, b2)
        current_loss = categorical_crossentropy(y, y_hat)
    print(f"it: {i}, loss: {current_loss}")
    dW1, db1, dW2, db2 = t.gradient(current_loss, [W1, b1, W2, b2])
    W1.assign_sub(a * dW1)
    b1.assign_sub(a * db1)
    W2.assign_sub(a * dW2)
    b2.assign_sub(a * db2)

print("W1 sau khi cap nhat:\n", W1.numpy())
print("W2 sau khi cap nhat:\n", W2.numpy())
print("b1 sau khi cap nhat:\n", b1.numpy())
print("b2 sau khi cap nhat:\n", b2.numpy())


y_predict = predict(X, W1, b1, W2, b2)
print("Ket qua thuc te:\n", y.numpy())
print("Ket qua du doan:\n", tf.round(y_predict).numpy())

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_predict, axis=1)), tf.float32))
print("Do chinh xac cua mo hinh:", acc.numpy())
