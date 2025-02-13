from ucimlrepo import fetch_ucirepo
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dữ liệu iris từ UCI
iris = fetch_ucirepo(id=53)

# Giữ lại 100 dòng đầu tiên
Data = iris.data.original.head(100)

# Biến đổi nhãn Iris-setosa -> 0, Iris-versicolor -> 1
Data = Data.replace(['Iris-setosa', 'Iris-versicolor'], [0, 1])

# Xáo trộn dữ liệu và chia thành 2 tập train và test
Data = Data.sample(frac=1).reset_index(drop=True)

X = Data.iloc[:, :-1].values
y = Data.iloc[:, -1].values.reshape(-1, 1)

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
W = tf.Variable(tf.random.normal([4, 1]))
b = tf.Variable(tf.random.normal([1]))

print("\nW khởi tạo ngẫu nhiên ban đầu: \n", W.numpy())
print("\nb khởi tạo ngẫu nhiên ban đầu: \n", b.numpy())

# Hàm dự đoán


@tf.function
def predict(X, W, b):
    return tf.nn.sigmoid(tf.matmul(X, W) + b)

# Hàm mất mát binary crossentropy


@tf.function
def binary_crossentropy_loss(y, y_hat):
    epsilon = 1e-7
    y_hat = tf.clip_by_value(y_hat, epsilon, 1. - epsilon)
    return -tf.reduce_mean(y * tf.math.log(y_hat) + (1 - y) * tf.math.log(1 - y_hat))


print("\nCập nhật trọng số W và b:\n")
alpha = 0.01
for it in range(250):
    with tf.GradientTape() as t:
        y_hat = predict(X_train, W, b)
        current_loss = binary_crossentropy_loss(
            y_train, y_hat)

    print(f"it: {it}, loss = {current_loss}")
    dW, db = t.gradient(current_loss, [W, b])
    W.assign_sub(alpha * dW)
    b.assign_sub(alpha * db)

print("\nW sau khi huấn luyện: \n", W.numpy())
print("\nb sau khi huấn luyện: \n", b.numpy())


y_hat = predict(X_test, W, b)

# Phân ngưỡng dự đoán so với 0.5
processed_y_hat = tf.round(y_hat)

print("\nKết quả thực tế: \n", y_test.numpy())
print("\nKết quả dự đoán: \n", processed_y_hat.numpy())

# Loại bỏ chiều thừa
processed_y_hat = tf.reshape(processed_y_hat, [-1])
y_test = tf.reshape(y_test, [-1])

# Chuyển tensor sang NumPy cho dễ tính toán
y_test_toNumPy = y_test.numpy()
processed_y_hat_toNumPy = processed_y_hat.numpy()

print("\nKết quả thực tế: \n", y_test_toNumPy)
print("\nKết quả dự đoán: \n", processed_y_hat_toNumPy)

# Tính độ chính xác
test_elements_len = len(y_test_toNumPy)

true_test_predictions = 0
for i in range(len(y_test_toNumPy)):
    if processed_y_hat_toNumPy[i] == y_test_toNumPy[i]:
        true_test_predictions += 1

accuracy = float(true_test_predictions / test_elements_len)

print(f"\nĐộ chính xác: {accuracy:.2f}")
