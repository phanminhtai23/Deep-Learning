import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler

# Đọc tệp Excel
file_path = 'C:/Users/MINH TAI/Desktop/Deep learning/assignment-10%/bt4/speaker+accent+recognition/accent-mfcc-data-1.csv'

df = pd.read_csv(file_path)

print("Dữ liệu đọc từ file csv:\n", df.head(5))

data = df.replace(
    ['ES', 'FR', 'GE', 'IT', 'UK', 'US'], [0, 1, 2, 3, 4, 5])
data = data.sample(frac=1).reset_index(drop=True)

print("Dữ liệu sau khi thay thế nhãn:\n", data.head(5))

# Lấy X là từ cột 2 đến cột cuối cùng, y là cột đầu tiên
X = data.iloc[:, 1:].values
y = data.iloc[:, 0:1].values.reshape(-1,)

y = tf.one_hot(y, depth=6).numpy()
print("Nhãn y sau khi one-hot:\n", y)

# Chia dữ liệu thành 2 tập train và test theo tỉ lệ 80-20
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Chuyển dữ liệu pandas sang tensor
x_train = tf.constant(x_train, dtype=tf.float32)
x_test = tf.constant(x_test, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.float32)

# Khởi tạo ngẫu nhiên các trọng số W và b, mạng gồm layer1 có 12 perceptron, layer2 có 6 perceptron
W1 = tf.Variable(tf.random.normal([12, 12]))
b1 = tf.Variable(tf.random.normal([12]))
W2 = tf.Variable(tf.random.normal([12, 6]))
b2 = tf.Variable(tf.random.normal([6]))

print("W1 khoi tao ngau nhien:\n", W1.numpy())
print("W2 khoi tao ngau nhien:\n", W2.numpy())
print("b1 khoi tao ngau nhien:\n", b1.numpy())
print("b2 khoi tao ngau nhien:\n", b2.numpy())


@tf.function
def layer1(X, W1, b1):
    return tf.nn.leaky_relu(tf.matmul(X, W1) + b1)


@tf.function
def layer2(X, W2, b2):
    return tf.nn.softmax(tf.matmul(X, W2) + b2)


@tf.function
def predict(X, W1, b1, W2, b2):
    return layer2(layer1(X, W1, b1), W2, b2)


@tf.function
def categorical_crossentropy_loss(y, y_hat):
    epsilon = 1e-7
    y_hat = tf.clip_by_value(y_hat, epsilon, 1. - epsilon)
    return -tf.reduce_mean(tf.reduce_sum(y*tf.math.log(y_hat), axis=1))


a = 0.01
it = 1000
for i in range(it):
    with tf.GradientTape() as t:
        y_hat = predict(x_train, W1, b1, W2, b2)
        cur_loss = categorical_crossentropy_loss(y_train, y_hat)
    if i % 10 == 0:
        print(f"it: {i}, loss = {cur_loss}")
    dW1, db1, dW2, db2 = t.gradient(cur_loss, [W1, b1, W2, b2])
    W1.assign_sub(a*dW1)
    b1.assign_sub(a*db1)
    W2.assign_sub(a*dW2)
    b2.assign_sub(a*db2)

print("W1 sau khi cap nhat:\n", W1.numpy())
print("W2 sau khi cap nhat:\n", W2.numpy())
print("b1 sau khi cap nhat:\n", b1.numpy())
print("b2 sau khi cap nhat:\n", b2.numpy())

predicted_y = predict(x_test, W1, b1, W2, b2)
predicted_y = tf.argmax(predicted_y, axis=1)
y_test = tf.argmax(y_test, axis=1)

print("y thực tế:\n", y_test.numpy())
print("y dự đoán:\n", predicted_y.numpy())

acc = tf.reduce_mean(tf.cast(
    tf.equal(y_test, predicted_y), tf.float32))

print("Độ chính xác: ", acc.numpy())
