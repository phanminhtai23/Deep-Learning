from ucimlrepo import fetch_ucirepo
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load dữ liệu iris từ UCI
iris = fetch_ucirepo(id=53)

print(iris.data.original)
# Giữ lại 100 dòng đầu tiên
Data = iris.data.original.head(100)

print(Data)
# Biến đổi nhãn Iris-setosa -> 0, Iris-versicolor -> 1
Data = Data.replace(['Iris-setosa', 'Iris-versicolor'], [0, 1])

print(Data)
# Xáo trộn dữ liệu và chia thành 2 tập train và test
Data = Data.sample(frac=1).reset_index(drop=True)

X = Data.iloc[:, :-1]
y = Data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Xây dựng mô hình
model = Sequential()
model.add(Dense(1, input_shape=(4,), use_bias=True, activation='sigmoid'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)

# Đánh giá mô hình
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
