from ucimlrepo import fetch_ucirepo
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dữ liệu iris từ UCI
iris = fetch_ucirepo(id=53)

Data = iris.data.original

# Biến đổi nhãn Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2
Data = Data.replace(
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])

# Xáo trộn dữ liệu và chia thành 2 tập train và test
Data = Data.sample(frac=1).reset_index(drop=True)

X = Data.iloc[:, :-1].values
y = Data.iloc[:, -1].values

# One hot encoding y, vơi 3 nhãn: 0, 1, 2
y = tf.one_hot(y, depth=3).numpy()

# print("\nDữ liệu X: \n", X[:5])
# print("type x: ", type(X))
print("\nNhãn y: \n", Data.iloc[:, -1].values[:5])
print("\nNhãn y one hot: \n", y[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Chuyển dữ liệu pandas sang tensor
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Khởi tạo ngẫu nhiên các trọng số W và b thay vì gán sẵn
W = tf.Variable(tf.random.normal([4, 3]))
b = tf.Variable(tf.random.normal([3]))

print("\nW khởi tạo ngẫu nhiên ban đầu: \n", W.numpy())
print("\nb khởi tạo ngẫu nhiên ban đầu: \n", b.numpy())

# Hàm dự đoán
@tf.function
def predict(X, W, b):
    return tf.nn.softmax(tf.matmul(X, W) + b)

# Hàm mất mát Categorial crossentropy
@tf.function
def categorial_crossentropy_loss(y, y_hat):
    epsilon = 1e-7
    y_hat = tf.clip_by_value(y_hat, epsilon, 1. - epsilon)
    return -tf.reduce_mean(tf.reduce_sum(y*tf.math.log(y_hat), axis = 1))


print("\nCập nhật trọng số W và b:\n")
alpha = 0.1
for it in range(500):
    with tf.GradientTape() as t:
        y_hat = predict(X_train, W, b)
        current_loss = categorial_crossentropy_loss(
            y_train, y_hat)

    print(f"it: {it}, loss = {current_loss}")
    dW, db = t.gradient(current_loss, [W, b])
    W.assign_sub(alpha * dW)
    b.assign_sub(alpha * db)

print("\nW sau khi huấn luyện: \n", W.numpy())
print("\nb sau khi huấn luyện: \n", b.numpy())


predicted_y = predict(X_test, W, b)

print("\nKết quả thực tế: \n", y_test.numpy()[:5])
print("\nKết quả dự đoán: \n", predicted_y.numpy()[:5])

# Chuyển về dạng nhãn gốc
y_test = tf.argmax(y_test, axis=1)
processed_y_hat = tf.argmax(predicted_y, axis=1)

# Tính độ chính xác
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_test, processed_y_hat), tf.float32))

print("\n Y thực tế:\n", y_test.numpy())
print("\n Y dự đoán:\n", processed_y_hat.numpy())

print("\nAccuracy: {:.2f}".format(accuracy.numpy()))