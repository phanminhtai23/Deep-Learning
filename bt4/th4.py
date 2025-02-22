import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Đọc tệp Excel
file_path = './assignment-10%/bt4/data.xlsx'
df = pd.read_excel(file_path, engine='openpyxl', sheet_name=1)
print("Dữ liệu đọc từ file:\n", df.head())

# Trộn dữ liệu
data = df.sample(frac=1).reset_index(drop=True)

# Chia dữ liệu, dữ liệu X từ cột 1 đên cột cuối cùng, dữ liệu y là cột cuối cùng
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

print("X:\n", X[:5])
print("Y:\n", y[:5])

# Chia dữ liệu thành 2 tập train và test theo tỉ lệ 80-20
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("x_train:\n", x_train[:5])

# Chuyển dữ liệu pandas sang tensor
x_train = tf.constant(x_train, dtype=tf.float32)
x_test = tf.constant(x_test, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.float32)

# Khởi tạo ngẫu nhiên các trọng số W và b, mạng gồm layer1 có 4 perceptrons, layer2 có 1 perceptron
W1 = tf.Variable(tf.random.normal([2, 4]))
b1 = tf.Variable(tf.random.normal([4]))
W2 = tf.Variable(tf.random.normal([4, 1]))
b2 = tf.Variable(tf.random.normal([1]))

print("W1 khoi tao ngau nhien:\n", W1.numpy())
print("W2 khoi tao ngau nhien:\n", W2.numpy())
print("b1 khoi tao ngau nhien:\n", b1.numpy())
print("b2 khoi tao ngau nhien:\n", b2.numpy())


@tf.function
def layer1(X, W1, b1):
    return tf.nn.relu(tf.matmul(X, W1) + b1)


@tf.function
def layer2(X, W2, b2):
    return tf.nn.sigmoid(tf.matmul(X, W2) + b2)


@tf.function
def predict(X, W1, b1, W2, b2):
    return layer2(layer1(X, W1, b1), W2, b2)


@tf.function
def binary_crossentropy_loss(y, y_hat):
    epsilon = 1e-7
    y_hat = tf.clip_by_value(y_hat, epsilon, 1. - epsilon)
    return -tf.reduce_mean(y*tf.math.log(y_hat) + (1-y)*tf.math.log(1-y_hat))


a = 0.01
it = 1000

for i in range(it):
    with tf.GradientTape() as t:
        y_hat = predict(x_train, W1, b1, W2, b2)
        cur_loss = binary_crossentropy_loss(y_train, y_hat)
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

acc = tf.reduce_mean(tf.cast(
    tf.equal(y_test, tf.round(predicted_y)), tf.float32))

print("Real y:\n", y_test.numpy())
print("Predicted y:\n", tf.round(predicted_y).numpy())

print("Accuracy: ", acc.numpy())
