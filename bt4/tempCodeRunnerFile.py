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