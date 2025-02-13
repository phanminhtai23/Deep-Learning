from ucimlrepo import fetch_ucirepo
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load dữ liệu iris từ UCI
iris = fetch_ucirepo(id=53)

# Giữ lại 100 dòng đầu tiên
Data = iris.data.original

# Biến đổi nhãn Iris-setosa -> 0, Iris-versicolor -> 1
Data = Data.replace(
    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])

# Xáo trộn dữ liệu và chia thành 2 tập train và test
Data = Data.sample(frac=1).reset_index(drop=True)

X = Data.iloc[:, :-1].values
y = Data.iloc[:, -1].values

# One hot encoding y, chuyển y thành ma trận
y = tf.one_hot(y, depth=3).numpy()

# print("\nDữ liệu X: \n", X[:5])
# print("type x: ", type(X))
# print("\nNhãn y one hot: \n", y[:5])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nDữ liệu X sau chuẩn hóa: \n", X_train[:5])
print("\nNhãn y sau chuẩn hóa: \n", y_train[:5])